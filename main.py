
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse
import base64
app = FastAPI()

model = YOLO("best.pt")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message" : "YOLO API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("preprocessing received")

    

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    results = model(img, conf=0.15, iou=0.5, imgsz=416)

    count = len(results[0].boxes) if results[0].boxes is not None else 0

    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    print("🔥 REQUEST RECEIVED")

    # Save raw image
    with open("debug_upload.jpg", "wb") as f:
        f.write(contents)

    print("🔥 FILE SIZE:", len(contents))

    return {
        "count": count,
        "image": img_base64
    }

    

from fastapi.responses import Response

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    results = model(img, conf=0.15, iou=0.5, imgsz=416)
    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/health")
def health():
    return {"status": "ready"}

@app.get("/warmup")
def warmup():
    """Run a dummy inference to pre-load the YOLO model into memory.
    Called by the frontend after /health confirms the server is up."""
    try:
        dummy = np.zeros((416, 416, 3), dtype=np.uint8)
        model(dummy, imgsz=416, verbose=False)
        return {"status": "warmed_up"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

#for multiple images 


from typing import List
from fastapi import UploadFile, File

@app.post("/predict-multiple")
async def predict_multiple(files: list[UploadFile] = File(...)):

    counts = []
    total = 0

    for file in files:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        

        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)

        results = model(img, conf=0.15, iou=0.5, imgsz=416)

        if results[0].boxes is not None:
            count = len(results[0].boxes)
        else:
            count = 0

        counts.append(count)
        total += count


    # 🔥 BASIC STATS
    num_images = len(counts)
    avg = total / num_images if num_images > 0 else 0
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0

    return {
        "counts_per_image": counts,
        "total": total,
        "num_images": num_images,
        "average": avg,
        "max": max_count,
        "min": min_count
    }

    


