
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

    results = model(img)

    #counting of microplastics
    if results[0].boxes is not None:
        count = len(results[0].boxes)
    else:
        count = 0

    #draw boxes on the given image..
    
    annotated = results[0].plot()
    
    #convert to jpeg
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    #return both

    return JSONResponse({
        "count" : count,
        "image" : img_base64
    })

    boxes = results[0].boxes.xyxy.tolist()
    count = len(boxes)


    return {
        "detection" : boxes,
        "count" : count
    }


