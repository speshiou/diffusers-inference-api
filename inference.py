import sys

sys.path.extend(["IP-Adapter"])

import torch


def get_face_embedding(filename):
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(filename)
    faces = app.get(image)

    return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)


def inference(prompt, faceid_embeds):
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    from PIL import Image

    from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

    base_model_path = "SG161222/RealVisXL_V3.0"
    ip_ckpt = "ip-adapter-faceid_sdxl.bin"
    device = "cuda"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )

    # load ip-adapter
    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

    # generate image
    negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    )

    images = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=faceid_embeds,
        num_samples=2,
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=2024,
    )

    return images
