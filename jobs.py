import redis
from paddleocr import TextRecognition
import torch
from fast_plate_ocr import ONNXPlateRecognizer
import numpy as np
import os
import cv2

r = redis.Redis()
# ocr = TextRecognition(model_name="PP-OCRv5_mobile_rec", device="gpu")
yolo = torch.hub.load(
    "yolov5", 
    "custom", 
    path=os.path.join("models", "plate.pt"),
    device="gpu", 
    source="local"
)
onnx = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model', device='gpu')

def process_car(car_id):
    print(f"[RQ] Processing car: {car_id}")
    try:
        r.set(f"car:{car_id}:status", "processing", ex=300)

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

        recs = onnx.run([r[1] for r in res])
        for r in recs:
            print(r)
            
    except Exception as e:
        r.set(f"car:{car_id}:status", "error", ex=600)
        print(e)
