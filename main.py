from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse, Response
import base64
from typing import List

app = FastAPI()

model = YOLO("best.pt")

# ── Run a real warmup at startup so the model is hot before first request ──────
import threading

def _warmup_on_start():
    try:
        # Use a realistic non-blank image (random noise) for a more effective warmup
        dummy = np.random.randint(50, 200, (416, 416, 3), dtype=np.uint8)
        model(dummy, imgsz=416, verbose=False)
        print("✅ Model warmed up at startup")
    except Exception as e:
        print(f"⚠️  Warmup error: {e}")

threading.Thread(target=_warmup_on_start, daemon=True).start()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared preprocessing ───────────────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=1.8, beta=40)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def decode_image(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

# ── Routes ────────────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "YOLO API is running"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ready"}

@app.get("/warmup")
def warmup():
    """
    Explicit warmup endpoint — uses random-noise image (not blank zeros)
    so the model processes a realistic input and fully warms up.
    """
    try:
        dummy = np.random.randint(50, 200, (416, 416, 3), dtype=np.uint8)
        model(dummy, imgsz=416, verbose=False)
        return {"status": "warmed_up"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"🔥 /predict received — file size: {len(contents)} bytes")

    img = decode_image(contents)
    img = preprocess(img)

    results = model(img, conf=0.25, iou=0.5, imgsz=416)
    count = len(results[0].boxes) if results[0].boxes is not None else 0

    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {"count": count, "image": img_base64}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    img = preprocess(img)

    results = model(img, conf=0.25, iou=0.5, imgsz=416)
    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/predict-multiple")
async def predict_multiple(files: list[UploadFile] = File(...)):
    counts = []
    images = []
    filenames = []
    total = 0

    for file in files:
        contents = await file.read()
        img = decode_image(contents)
        img = preprocess(img)

        results = model(img, conf=0.25, iou=0.5, imgsz=416)

        count = len(results[0].boxes) if results[0].boxes is not None else 0
        counts.append(count)
        total += count

        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        images.append(img_b64)
        filenames.append(file.filename)

    num_images = len(counts)
    avg = round(total / num_images, 2) if num_images > 0 else 0

    return {
        "counts_per_image": counts,
        "images": images,
        "filenames": filenames,
        "total": total,
        "num_images": num_images,
        "average": avg,
        "max": max(counts) if counts else 0,
        "min": min(counts) if counts else 0,
    }