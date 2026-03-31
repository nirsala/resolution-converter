#!/bin/bash
# ────────────────────────────────────────────────────────────────────────
# Download AI model weights for ResolutionAI
# Run once before `docker compose up`
# ────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "📥 Downloading Real-ESRGAN weights (x4plus, ~65MB)..."
mkdir -p "$SCRIPT_DIR/models/realesrgan"
wget -q --show-progress \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
  -O "$SCRIPT_DIR/models/realesrgan/RealESRGAN_x4plus.pth"
echo "✅ Real-ESRGAN saved."

echo ""
echo "📥 Downloading U2Net weights (~176MB)..."
mkdir -p "$SCRIPT_DIR/models/u2net"
wget -q --show-progress \
  "https://drive.google.com/uc?export=download&id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ" \
  -O "$SCRIPT_DIR/models/u2net/u2net.pth" || \
gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ \
  -O "$SCRIPT_DIR/models/u2net/u2net.pth"
echo "✅ U2Net saved."

echo ""
echo "🎉 All models downloaded. You can now run: docker compose up --build"
