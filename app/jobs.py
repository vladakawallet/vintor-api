import redis
# from paddleocr import TextRecognition
import torch
from fast_plate_ocr import ONNXPlateRecognizer
import numpy as np
import os
import cv2
from logger import setup_logger
from celery import Celery

log = setup_logger("worker", os.path.join("logs", "worker.log"))
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
            path=os.path.join("models", "plate.pt"),
            device="cpu", 
            source="local"
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
        

@celery_app.task()
def process_car(car_id: str):
    global yolo, vin_ocr, plate_onnx
    log.info(f"[i] Processing car: {car_id}")

    try:
        r.set(f"car:{car_id}:status", "processing", ex=600)

        images = r.get(f"car:{car_id}:images")
        images = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in images]

        if not images:
            r.set(f"car:{car_id}:status", "done", ex=600)
            r.set(f"car:{car_id}:result", "", ex=600)
            return "No images to process"
        
        crops = yolo(images).crop(save=False)

        res = []
        for r in crops:
            if r['conf'] > 0.7:
                res.append((r['conf'], r['im']))
        res = sorted(res, key=lambda x: x[0], reverse=True)[:3]

        recs = plate_onnx.run([r[1] for r in res])
        for r in recs:
            print(r)
            
    except Exception as e:
        r.set(f"car:{car_id}:status", "error", ex=600)
        print(e)
