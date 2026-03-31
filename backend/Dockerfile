FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install base requirements (no torch/basicsr yet)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Install PyTorch (CPU) — must be before basicsr
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install basicsr + realesrgan with no-build-isolation to avoid CUDA conflicts
RUN pip install --no-cache-dir --no-build-isolation basicsr==1.4.2
RUN pip install --no-cache-dir realesrgan==0.3.0

# Copy application code
COPY . .

# Make startup scripts executable
RUN chmod +x /app/scripts/*.sh

EXPOSE 10000
