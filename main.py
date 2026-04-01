
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import Response
app = FastAPI()

model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message" : "YOLO API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    #draw boxes on the given image..
    annotated = results[0].plot()
    #convert to jpeg
    _, buffer = cv2.imencode(".jpg", annotated)

    return Response(content = buffer.tobytes(), media_type = "image/jpeg")

    boxes = results[0].boxes.xyxy.tolist()
    count = len(boxes)


    return {
        "detection" : boxes,
        "count" : count
    }


