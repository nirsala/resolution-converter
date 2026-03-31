#!/bin/sh
# ── Download AI model weights if not already present ──────────────────────
# Called at container startup on Render (models are cached on the persistent disk).

ESRGAN_PATH="/storage/models/realesrgan/RealESRGAN_x4plus.pth"
U2NET_PATH="/storage/models/u2net/u2net.pth"

mkdir -p /storage/models/realesrgan
mkdir -p /storage/models/u2net
mkdir -p /storage/uploads
mkdir -p /storage/outputs

# ── Real-ESRGAN (~65 MB) ───────────────────────────────────────────────────
if [ ! -f "$ESRGAN_PATH" ]; then
  echo "📥 Downloading Real-ESRGAN weights..."
  wget -q --show-progress \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
    -O "$ESRGAN_PATH"
  echo "✅ Real-ESRGAN downloaded."
else
  echo "✅ Real-ESRGAN already present, skipping download."
fi

# ── U2Net (~176 MB) ────────────────────────────────────────────────────────
if [ ! -f "$U2NET_PATH" ]; then
  echo "📥 Downloading U2Net weights..."
  pip install -q gdown
  python -c "
import gdown, os
gdown.download(
    id='1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
    output='$U2NET_PATH',
    quiet=False
)
"
  echo "✅ U2Net downloaded."
else
  echo "✅ U2Net already present, skipping download."
fi

echo "🎉 Models ready."
