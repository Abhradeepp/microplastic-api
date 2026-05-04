from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import Response
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

def extract_particle_stats(img, boxes):
    """Helper function to calculate size and density for each detected particle."""
    stats = []
    if boxes is not None:
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            width = x2 - x1
            height = y2 - y1
            size = width * height
            
            if size > 0:
                crop = img[y1:y2, x1:x2]
                # Convert to grayscale to get a single average intensity value
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                density = float(np.mean(gray_crop))
            else:
                density = 0.0
                
            stats.append({"size": int(size), "density": round(density, 2)})
    return stats

@app.get("/")
def home():
    return {"message": "YOLO API is running"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    contents = file.file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=1.8, beta=40)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    results = model(img, conf=0.25, iou=0.5, imgsz=416)

    count = len(results[0].boxes) if results[0].boxes is not None else 0
    
    # Extract advanced stats
    particle_stats = extract_particle_stats(img, results[0].boxes)

    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "count": count,
        "image": img_base64,
        "particle_stats": particle_stats # <-- New data being sent to frontend
    }

@app.get("/health")
def health():
    return {"status": "ready"}

@app.get("/warmup")
def warmup():
    try:
        dummy = np.zeros((416, 416, 3), dtype=np.uint8)
        model(dummy, imgsz=416, verbose=False)
        return {"status": "warmed_up"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/predict-multiple")
def predict_multiple(files: list[UploadFile] = File(...)):
    counts = []
    images = []         
    filenames = []
    all_particle_stats = [] # <-- Array to hold stats for all images
    total = 0

    for file in files:
        contents = file.file.read()

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
        
        # Extract and store stats for this specific image
        stats = extract_particle_stats(img, results[0].boxes)
        all_particle_stats.extend(stats)

        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        images.append(img_base64)
        filenames.append(file.filename)

    num_images = len(counts)
    avg = round(total / num_images, 2) if num_images > 0 else 0
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0

    return {
        "counts_per_image": counts,
        "images": images,            
        "filenames": filenames,      
        "total": total,
        "num_images": num_images,
        "average": avg,
        "max": max_count,
        "min": min_count,
        "particle_stats": all_particle_stats # <-- Passing merged stats to frontend
    }