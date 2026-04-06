# ============================================================
# Dockerfile — Chest X-Ray Diagnostic Assistant (Backend)
# Runs on Hugging Face Spaces (Docker SDK)
# Port: 7860 (required by HF Spaces)
# ============================================================

FROM python:3.10-slim

# System dependencies for OpenCV and image libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

# Install PyTorch CPU-only (much smaller than CUDA version, fine for HF free tier)
# Install separately before other packages to avoid conflicts
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Install huggingface_hub for model downloading at runtime
RUN pip install --no-cache-dir huggingface_hub==0.20.3

# Install the rest of the requirements
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pillow==10.2.0 \
    numpy==1.26.3 \
    pandas==2.1.4 \
    opencv-python-headless==4.9.0.80 \
    scikit-learn==1.4.0 \
    python-multipart==0.0.7 \
    python-dotenv==1.0.0 \
    "google-genai==0.2.2" \
    "requests==2.31.0"

# Copy application source code
COPY app.py .
COPY src/ ./src/

# Copy model downloader script
COPY download_models.py .

# Create models directory structure
RUN mkdir -p models/chexnet models/binary

# Pre-download models at build time using the download script
# Set HF_TOKEN as a build arg (injected from HF Space secrets during build)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Download models at build time (they are cached in the image layer)
RUN python download_models.py

# HF Spaces requires port 7860
EXPOSE 7860

# Set environment variables for runtime
ENV HOST=0.0.0.0
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
