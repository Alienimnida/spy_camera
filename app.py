from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import os
import shutil
from pathlib import Path
import base64

app = FastAPI(title="YOLOv5 Object Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASEPATH = "uploads"
os.makedirs(BASEPATH, exist_ok=True)

# Handle torch.load PosixPath issue
import torch.serialization
original_load = torch.load

def patched_torch_load(*args, **kwargs):
    try:
        return original_load(*args, **kwargs)
    except NotImplementedError as e:
        if "PosixPath" in str(e):
            import pathlib
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            result = original_load(*args, **kwargs)
            pathlib.PosixPath = temp
            return result
        else:
            raise

torch.load = patched_torch_load

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', trust_repo=True)
print("Model loaded successfully!")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.post("/predict")
async def predict_img(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        return JSONResponse({"success": False, "error": "Invalid file format. Supported: jpg, jpeg, png"})

    file_path = os.path.join(BASEPATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)
    df = results.pandas().xyxy[0]

    detections = [{
        "class": row["name"],
        "confidence": float(row["confidence"]),
        "bbox": {
            "x1": float(row["xmin"]),
            "y1": float(row["ymin"]),
            "x2": float(row["xmax"]),
            "y2": float(row["ymax"]),
        }
    } for _, row in df.iterrows()]

    save_dir = Path(BASEPATH)
    results.save(save_dir=save_dir)

    result_files = list(save_dir.glob(f"*{os.path.splitext(file.filename)[0]}*"))
    if not result_files:
        return JSONResponse({"success": False, "error": "No result image found"})

    image_base64 = encode_image_to_base64(str(result_files[0]))
    return JSONResponse({
        "success": True,
        "detections": detections,
        "image": image_base64
    })

@app.get("/")
async def root():
    return {"message": "YOLOv5 Object Detection API. Upload an image at /predict"}

app.mount("/results", StaticFiles(directory=BASEPATH), name="results")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)