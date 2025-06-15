from fastapi import FastAPI, UploadFile, FileAdd commentMore actions
import redis
import numpy as np
import cv2
import os
from contextlib import asynccontextmanager
from jobs import setup_models_and_dependencies, process_car

from logger import setup_logger
from celery import Celery

from typing import List

log = setup_logger("app", os.path.join("logs", "app.log"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[i] Starting FastAPI")
    setup_models_and_dependencies()
    yield

app = FastAPI(lifespan=lifespan)
celery_app = Celery("image_processor", broker="redis://localhost:6379")
r = redis.Redis()
r = None

@app.post("/plate/{car_id}")
async def receive_images(car_id: str, files: List[UploadFile] = File(...)):
    if files and car_id:
        for f in files:

            npbytes = await f.read()
            npimg = np.frombuffer(npbytes, np.uint8)
            npimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            r.rpush(f"car:{car_id}:images", npimg.tobytes())
        
        task = process_car.delay(car_id)
        return {"status": "processing", "car_id": car_id, "task": task.id}
    
    else: return {"status": "failed", "car_id": car_id if car_id else "None"}

@app.get("/result/{car_id}")
def get_car_results(car_id: str):
    status = r.get(f"car:{car_id}:status")
    result = r.get(f"car:{car_id}:result")

    return {
        "status": status.decode() if status else "pending",
        "result": result.decode() if result else None
    }
