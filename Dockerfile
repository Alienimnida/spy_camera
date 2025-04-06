# Use an official Python base image
FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5 repo to use torch.hub locally (optional)
RUN git clone https://github.com/ultralytics/yolov5 && \
    pip install -r yolov5/requirements.txt

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
