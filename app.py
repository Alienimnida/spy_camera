from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import torch
import cv2
import os
import shutil
import time
import numpy as np
import sys
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="YOLOv5 Object Detection API")

# Create directories if they don't exist
BASEPATH = "uploads"
DETECT_FOLDER = "runs/detect"
os.makedirs(BASEPATH, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

# Patch torch.load to handle PosixPath on Windows
import torch.serialization
original_load = torch.load

def patched_torch_load(*args, **kwargs):
    try:
        return original_load(*args, **kwargs)
    except NotImplementedError as e:
        if "PosixPath" in str(e):
            # Monkey patch pathlib.PosixPath for this session
            import pathlib
            import io
            
            # Override the PosixPath class with WindowsPath on Windows
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            
            # Retry loading with patched pathlib
            result = original_load(*args, **kwargs)
            
            # Restore the original PosixPath
            pathlib.PosixPath = temp
            
            return result
        else:
            raise

# Replace torch.load with our patched version
torch.load = patched_torch_load

# Use a simpler approach with YOLOv5
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
print("Model loaded successfully!")

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.post("/predict")
async def predict_img(request: Request, file: UploadFile = File(...)):
    # Check if file is an image
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png']:
        # Save the uploaded file
        file_path = os.path.join(BASEPATH, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Upload folder is {file_path}")
        
        # Process image with YOLOv5
        results = model(file_path)
        
        # Extract detection data
        detection_results = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
        
        detections = []
        for _, detection in detection_results.iterrows():
            detections.append({
                "class": detection["name"],
                "confidence": float(detection["confidence"]),
                "bbox": {
                    "x1": float(detection["xmin"]),
                    "y1": float(detection["ymin"]),
                    "x2": float(detection["xmax"]),
                    "y2": float(detection["ymax"])
                }
            })
        
        # Save results
        save_dir = Path(BASEPATH)
        results.save(save_dir=save_dir)
        
        # Find the saved result image
        result_files = list(save_dir.glob(f"*{os.path.splitext(file.filename)[0]}*"))
        
        if result_files:
            saved_file = str(result_files[0])
            # Encode image to base64
            image_base64 = encode_image_to_base64(saved_file)
            
            # Return both image and detection data
            return JSONResponse({
                "success": True,
                "detections": detections,
                "image": image_base64
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "No result image found"
            })
    
    elif file_extension == '.mp4':
        # Save the uploaded video
        file_path = os.path.join(BASEPATH, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process video frames
        return StreamingResponse(process_video(file_path), media_type="multipart/x-mixed-replace; boundary=frame")
    
    else:
        return JSONResponse({
            "success": False,
            "error": "Invalid file format. Supported formats: jpg, jpeg, png, mp4"
        })

async def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Process frame with YOLOv5
        results = model(frame)
        
        # Get the rendered frame with detections
        rendered_frame = results.render()[0]
        
        # Convert to JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', rendered_frame)
        
        # Yield the frame in a format suitable for multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
        # Control frame rate
        time.sleep(0.1)
    
    cap.release()

@app.get("/")
async def root():
    return {"message": "YOLOv5 Object Detection API. Upload an image or video to /predict"}

# Mount the detection results directory
app.mount("/results", StaticFiles(directory=BASEPATH), name="results")

# For testing the video streaming feature in a browser
@app.get("/video-feed")
async def video_feed():
    # Path to the test video file
    video_path = "output.mp4"
    
    # Check if file exists
    if not os.path.exists(video_path):
        return {"error": "Test video file not found"}
    
    return StreamingResponse(process_video(video_path), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)