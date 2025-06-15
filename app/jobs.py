import redis
# from paddleocr import TextRecognition
import torch
from fast_plate_ocr import ONNXPlateRecognizer
import numpy as np
import os
import cv2
from app.logger import setup_logger
from celery import Celery
from celery.signals import worker_process_init
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log = setup_logger("worker", os.path.join(BASE_DIR, "logs", "worker.log"))
r = redis.Redis()
celery_app = Celery("image_processor", broker="redis://localhost:6379")


yolo = None
plate_onnx = None
vin_ocr = None

def setup_models_and_dependencies():
    log.info("[i] Starting models")

    global yolo, plate_onnx, vin_ocr

    try:
        log.info("[i] YOLOv5s setup begin")
        yolo = torch.hub.load(
            "yolov5",
            "custom",
            path=os.path.join(BASE_DIR, "models", "plate.pt"),
            device="cuda:0",
            source="local",
            force_reload=True
        )
        if yolo: log.info("[i] YOLO setup finished")
        else: log.info("[i] YOLO setup failed!")

        log.info("[i] plate_onnx setup begin")
        plate_onnx = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model', device='cpu')
        if plate_onnx: log.info("[i] plate_onnx setup finished")
        else: log.info("[i] plate_onnx setup failed!")

        # log.info("[i] vin_ocr setup begin")
        # vin_ocr = TextRecognition(model_name="PP-OCRv5_mobile_rec", device="gpu")   
        # if plate_onnx: log.info("[i] vin_ocr setup finished")
        # else: log.info("[i] vin_ocr setup failed!")

    except Exception as e:
        log.error(f"[!] Exception during models setup occured. Exiting the function with status ERROR: {e}")

@worker_process_init.connect
def init_models(**kwargs):
    setup_models_and_dependencies()

@celery_app.task()
def process_car(car_id: str):
    global yolo, vin_ocr, plate_onnx, r
    log.info(f"[i] Processing car: {car_id}")

    try:
        r.set(f"car:{car_id}:status", "processing", ex=600)

        img_bytes = r.lrange(f"car:{car_id}:images", 0, -1)
        images = []
        for b in img_bytes:
            i = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if i is not None:
                images.append(i)

        if not images:
            print("no images detected", len(images))
            r.set(f"car:{car_id}:status", "done", ex=600)
            r.set(f"car:{car_id}:result", "", ex=600)
            return "No images to process"

        crops = yolo(images).crop(save=False)

        res = []
        for rs in crops:
            if rs['conf'] > 0.7:
                res.append((rs['conf'], rs['im']))
        res = sorted(res, key=lambda x: x[0], reverse=True)[:3]
        crops = [rs[1] for rs in res]
        batch = []
        for c in crops:
           im = np.frombuffer(c, np.uint8)
           cim = cv2.imdecode(im, cv2.IMREAD_COLOR)
           if cim:
               batch.append(cim)

        recs = plate_onnx.run(batch)
        for rc in recs:
            print(rc)

    except Exception as e:
        print(traceback.format_exc())
        r.set(f"car:{car_id}:status", "error", ex=600)
        print(e)
