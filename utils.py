import sys
from urllib.request import urlretrieve
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

sys.path.extend(["/IP-Adapter"])

FACE_ID_MODEL_CACHE = "./faceid-cache"
FACE_ID_MODEL_URL = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin?download=true"


def download_weights(url, dest):
    urlretrieve(url, dest)


def downloads():
    download_weights(FACE_ID_MODEL_URL, FACE_ID_MODEL_CACHE)


def inpaint_masked(pipe, **kwargs):
    image = kwargs["image"]
    mask_image = kwargs["mask_image"]
    width, height = image.size

    mask_padding = 0
    converted_mask = mask_image.convert("L")
    crop_region = get_crop_region(np.array(converted_mask), mask_padding)
    crop_region = expand_crop_region(
        crop_region, width, height, converted_mask.width, converted_mask.height
    )
    # print(crop_region)
    # crop_region = expand_rect_to_multiple_of_8(crop_region)
    # print(crop_region)
    x1, y1, x2, y2 = crop_region
    paste_to = (x1, y1, x2 - x1, y2 - y1)

    # crop image using masked area
    converted_mask = converted_mask.crop(crop_region)
    cropped_image = image.crop(crop_region)

    # scale up the masked area to get a better prediction
    converted_mask = converted_mask.resize((width, height))
    cropped_image = cropped_image.resize((width, height))

    inputs = {
        **kwargs,
        "image": cropped_image,
        "mask_image": converted_mask,
    }

    # inpaint
    inpainted = pipe(**inputs).images[0]

    # overlay the inpainted result onto the original image
    x, y, w, h = paste_to
    inpainted = inpainted.resize((w, h))

    final_image = Image.new("RGBA", image.size)
    final_image.paste(image, (0, 0))
    final_image.paste(inpainted, (x, y))

    # convert back to RGB in order to make further predictions
    return final_image.convert("RGB")


def get_face_embedding(image):
    import torch
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(image)

    return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)


def inference(prompt, faceid_embeds):
    import torch
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    from PIL import Image

    from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

    base_model_path = "SG161222/RealVisXL_V3.0"
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
    ip_model = IPAdapterFaceIDXL(pipe, FACE_ID_MODEL_CACHE, device)

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


def resize_and_crop(img, width, height):
    output_ratio = width / height
    img_ratio = img.width / img.height
    if output_ratio > img_ratio:
        img = img.resize((width, int(width / img_ratio)))
    else:
        img = img.resize((int(height * img_ratio), height))
    img_width, img_height = img.size

    output = Image.new("RGB", (width, height), (255, 255, 255))
    offset = ((width - img_width) // 2, (height - img_height) // 2)
    output.paste(img, offset)
    return output


def mask_to_pil(masks, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)
    """

    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left - pad, 0)),
        int(max(crop_top - pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h)),
    )


def expand_crop_region(
    crop_region, processing_width, processing_height, image_width, image_height
):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128.
    """

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2 - y1))
        y1 -= desired_height_diff // 2
        y2 += desired_height_diff - desired_height_diff // 2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2 - x1))
        x1 -= desired_width_diff // 2
        x2 += desired_width_diff - desired_width_diff // 2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2
