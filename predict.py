# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import shutil
import torch
import numpy
import uuid
from typing import List

sys.path.extend(["/IP-Adapter"])
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

from cog import BasePredictor, Input, Path

from diffusers import (
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForInpainting,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)

from diffusers.utils import load_image

from controlnet_aux.processor import Processor

import yolo
from utils import (
    get_face_embedding,
    download_weights,
    inpaint_masked,
    resize_and_crop,
)

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_ENABLE_SIGNED_URL = os.getenv("GCS_ENABLE_SIGNED_URL")
GCS_ENABLE_SIGNED_BY_KSA = os.getenv("GCS_ENABLE_SIGNED_BY_KSA")

MODEL_ID = "Lykon/dreamshaper-xl-v2-turbo"
DEPTH_MODEL_ID = "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"
FACE_ID_MODEL_CACHE = "./faceid-cache"
FACE_ID_MODEL_URL = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin?download=true"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the uploader info"""
        self.gcs_signing_service_account = None
        if GCS_ENABLE_SIGNED_BY_KSA:
            import gke

            self.gcs_signing_service_account = gke.get_ksa_service_account()

        """Load the model into memory to make running multiple predictions efficient"""
        download_weights(FACE_ID_MODEL_URL, FACE_ID_MODEL_CACHE)
        yolo.downloads()

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            varient="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")

        self.txt2img_pipe = AutoPipelineForText2Image.from_pipe(self.pipe)

        self.inpaint_pipe = AutoPipelineForInpainting.from_pipe(self.pipe).to("cuda")

        self.adapter = T2IAdapter.from_pretrained(
            DEPTH_MODEL_ID,
            torch_dtype=torch.float16,
            varient="fp16",
        ).to("cuda")

        self.adapter_pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            MODEL_ID,
            adapter=self.adapter,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        self.depth_processor = Processor("depth_midas")

        # create a dedicated sdxl pipeline for IP Adapter to prevent the txt2img pipeline from corrupting
        ip_pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            varient="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        )
        ip_pipe.enable_model_cpu_offload()

        self.ip_pipe = IPAdapterFaceIDXL(ip_pipe, FACE_ID_MODEL_CACHE, "cuda")

    def load_image(self, path):
        print(f"load_image from {path}")
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        ref_image: Path = Input(
            description="Image for reference",
            default=None,
        ),
        faceid_image: Path = Input(
            description="Image with a face",
            default=None,
        ),
        adapter_scale: float = Input(
            description="Scale for the adapter conditioning",
            ge=0,
            le=1,
            default=0.6,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "width": width,
            "height": height,
            "num_inference_steps": 7,
            "guidance_scale": 2,
        }

        restore_faces = False

        if ref_image:
            ref_image = self.load_image(ref_image)
            depth_image = self.depth_processor(ref_image, to_pil=True)
            depth_image = resize_and_crop(depth_image, width, height)
            images = self.adapter_pipe(
                image=[depth_image] * num_outputs,
                adapter_conditioning_scale=adapter_scale,
                **args,
            ).images

            restore_faces = True

        elif faceid_image:
            face_image = self.load_image(faceid_image)
            face_image = numpy.asarray(face_image)
            face_embeddings = get_face_embedding(face_image)

            kwargs = {
                **args,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }

            images = self.ip_pipe.generate(
                faceid_embeds=face_embeddings,
                **kwargs,
            )
        else:
            images = self.txt2img_pipe(
                **args,
            ).images

            restore_faces = True

        if restore_faces:
            fixed_images = []
            for image in images:
                fixed_image = self.restore_faces(self.inpaint_pipe, image, **args)
                fixed_images.append(fixed_image)

            images = fixed_images

        output_paths = []
        for image in images:
            output_path = "/tmp/out-{}.png".format(uuid.uuid1())
            image.save(output_path)

            if GCS_BUCKET_NAME:
                import gcs

                object_name = output_path.lstrip("/")
                object_url = gcs.upload_blob(GCS_BUCKET_NAME, output_path, object_name)
                if GCS_ENABLE_SIGNED_URL:
                    signed_object_url = gcs.generate_download_signed_url_v4(
                        GCS_BUCKET_NAME,
                        object_name,
                        service_account_email=self.gcs_signing_service_account,
                    )
                    output_paths.append(signed_object_url)
                else:
                    output_paths.append(object_url)
            else:
                output_paths.append(Path(output_path))

        return output_paths

    def restore_faces(
        self, inpaint_pipe, image, strength=0.6, max_num_faces=4, **kwargs
    ):
        result = yolo.detect_faces(image, confidence=0.6)
        if not result:
            return image
        preview, masks = result

        # override output dimensions with those of the source image
        width, height = image.size
        kwargs = {
            **kwargs,
            "width": width,
            "height": height,
        }

        num_faces = min(max_num_faces, len(masks))
        final_image = image
        for i in range(num_faces):
            mask = masks[i]
            final_image = inpaint_masked(
                pipe=inpaint_pipe,
                image=final_image,
                mask_image=mask,
                strength=strength,
                **kwargs,
            )
        return final_image
