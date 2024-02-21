import cv2
from PIL import Image
from ultralytics import YOLO
from utils import create_mask_from_bbox, mask_to_pil, download_weights

YOLO_FACE_MODEL_CACHE = "./yolo-face-cache"
YOLO_FACE_MODEL_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt?download=true"


def downloads():
    download_weights(YOLO_FACE_MODEL_URL, YOLO_FACE_MODEL_CACHE)


def detect_faces(image, confidence=0.3, device="cuda"):
    model = YOLO(YOLO_FACE_MODEL_CACHE)
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return None
    else:
        bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)
    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)
    return preview, masks
