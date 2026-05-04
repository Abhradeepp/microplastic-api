from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse, Response
import base64
from typing import List

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
    return {"message": "YOLO API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("preprocessing received")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=1.8, beta=40)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    results = model(img, conf=0.25, iou=0.5, imgsz=416)

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


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=1.8, beta=40)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    results = model(img, conf=0.25, iou=0.5, imgsz=416)
    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/health")
def health():
    return {"status": "ready"}

@app.get("/warmup")
def warmup():
    """Run a dummy inference to pre-load the YOLO model into memory."""
    try:
        dummy = np.zeros((416, 416, 3), dtype=np.uint8)
        model(dummy, imgsz=416, verbose=False)
        return {"status": "warmed_up"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── BATCH ENDPOINT (now returns annotated images) ──────────────────────────────
@app.post("/predict-multiple")
async def predict_multiple(files: list[UploadFile] = File(...)):

    counts = []
    images = []          # base64 annotated images — NEW
    filenames = []       # original filenames  — NEW
    total = 0

    for file in files:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.convertScaleAbs(img, alpha=1.8, beta=40)

        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)

        results = model(img, conf=0.25, iou=0.5, imgsz=416)

        count = len(results[0].boxes) if results[0].boxes is not None else 0
        counts.append(count)
        total += count

        # Encode annotated image to base64 — NEW
        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        images.append(img_b64)
        filenames.append(file.filename)

    num_images = len(counts)
    avg = round(total / num_images, 2) if num_images > 0 else 0
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0

    return {
        "counts_per_image": counts,
        "images": images,            # ← NEW
        "filenames": filenames,      # ← NEW
        "total": total,
        "num_images": num_images,
        "average": avg,
        "max": max_count,
        "min": min_count
    }
