# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import shutil
import torch
import numpy
import uuid
from typing import List

from cog import BasePredictor, Input, Path

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)

from diffusers.utils import load_image

from controlnet_aux.processor import Processor

import yolo
from utils import (
    get_face_embedding,
    inference,
    downloads,
    inpaint_masked,
    resize_and_crop,
)

MODEL_ID = "Lykon/dreamshaper-xl-v2-turbo"
DEPTH_MODEL_ID = "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        downloads()
        yolo.downloads()

        self.txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")

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

    def load_image(self, path):
        print(f"load_image from {path}")
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        ref_image: Path = Input(
            description="Image for reference",
            default=None,
        ),
        faceid_image: Path = Input(
            description="Image with a face",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        args = {
            "prompt": prompt,
            "num_inference_steps": 7,
            "guidance_scale": 2,
        }

        if ref_image:
            ref_image = self.load_image(ref_image)
            depth_image = self.depth_processor(ref_image, to_pil=True)
            depth_image = resize_and_crop(depth_image, width, height)
            images = self.adapter_pipe(
                image=depth_image,
                **args,
                width=width,
                height=height,
                adapter_conditioning_scale=1,
            )

        elif faceid_image:
            face_image = self.load_image(faceid_image)
            face_image = numpy.asarray(face_image)
            face_embeddings = get_face_embedding(face_image)
            images = inference(prompt, face_embeddings)
        else:
            images = self.txt2img_pipe(
                **args,
                width=width,
                height=height,
            )

        output_paths = []
        for image in images:
            output_path = "/tmp/out-{}.png".format(uuid.uuid1())
            image.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths

    def restore_faces(self, inpaint_pipe, image, strength=0.6, **kwargs):
        result = yolo.detect_faces(image)
        if not result:
            return image
        preview, masks = result

        width, height = image.size

        final_image = image
        for mask in masks:
            final_image = inpaint_masked(
                pipe=inpaint_pipe,
                image=final_image,
                mask_image=mask,
                width=width,
                height=height,
                strength=strength,
                **kwargs,
            )
        return final_image
