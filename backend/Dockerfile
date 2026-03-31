FROM python:3.11-slim

# System dependencies: FFmpeg, libGL (OpenCV), libmagic, wget
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmagic1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup scripts executable
RUN chmod +x /app/scripts/*.sh

# Render uses port 10000 by default
EXPOSE 10000
