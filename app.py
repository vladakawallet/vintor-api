from fastapi import FastAPI, UploadFile, File
from redis import Redis
from rq import Queue
import numpy as np
import cv2
from jobs import process_car

app = FastAPI()
r = Redis()
q = Queue(connection=r)

@app.post("/upload/{car_id}")
async def upload_car_images(car_id: str, files: list[UploadFile] = File(...)):
    for f in files:
        npbytes = await f.read()
        npimg = np.frombuffer(npbytes, np.uint8)
        npimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        r.rpush(f"car:{car_id}:images", npimg.tobytes())
    
    q.enqueue(process_car, car_id)

@app.get("/result/{car_id}")
def get_car_results(car_id: str):
    status = r.get(f"car:{car_id}:status")
    result = r.get(f"car:{car_id}:result")

    return {
        "status": status.decode() if status else "pending",
        "result": result.decode() if result else None
    }
