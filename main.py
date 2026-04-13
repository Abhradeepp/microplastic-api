
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

    results = model(img, conf=0.25)

    count = len(results[0].boxes) if results[0].boxes is not None else 0

    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

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

    results = model(img, conf=0.25)

    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

    


