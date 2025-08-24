# syntax=docker/dockerfile:1

# --- Stage 1: CPU Builder ---
# Use the full python image and add the required system library for headless OpenCV.
FROM python:3.10 AS builder

# Install essential system dependencies for headless OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation.
RUN pip install uv

WORKDIR /app

# First, install a CPU-only version of torch to satisfy dependencies for model downloading.
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file.
COPY backend/requirements-ai-gpu.txt .

# Install the rest of the dependencies from the requirements file.
RUN uv pip install --system -r requirements-ai-gpu.txt

# Copy assets needed for model pre-loading.
COPY assets /app/assets

# Preload model weights to prevent cold starts in production.
RUN CUDA_VISIBLE_DEVICES="" python3 - <<'PY'
from ultralytics import YOLO
import os
from simple_lama_inpainting import SimpleLama
from manga_ocr import MangaOcr

# Warm up the models by initializing them, which triggers weight downloads.
YOLO(os.path.join('/app/assets', 'models', 'model.pt'))
SimpleLama(download_only=True)
MangaOcr()
PY


# --- Stage 2: Final Production Image ---
# Hardcoded to your local CUDA 12.9 environment.
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential runtime dependencies, including git.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv.
RUN pip install uv

# Copy the requirements file from the builder stage.
COPY --from=builder /app/requirements-ai-gpu.txt .

# CORRECT FIX: Install packages using the requirements file in the final stage.
# This ensures they are correctly installed for this specific Python environment.
RUN uv pip install --system -r requirements-ai-gpu.txt

# Copy the pre-downloaded model weights from the builder stage's cache.
COPY --from=builder /root/.cache /root/.cache

# Now, install the GPU-enabled version of PyTorch for CUDA 12.x.
# This cleanly overwrites the CPU version from the builder stage.
RUN uv pip install --system torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Copy your application code.
COPY backend/app /app/backend/app
COPY assets /app/assets

ENV PYTHONPATH=/app/backend:/app
ENV ASSETS_ROOT=/app/assets
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 5001

CMD ["python3", "-m", "uvicorn", "backend.app.gpu_service.main:app", "--host", "0.0.0.0", "--port", "5001"]
