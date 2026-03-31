#!/bin/sh
set -e

echo "🚀 Starting Resolution API..."

# Download models if missing
sh /app/scripts/download_models.sh

# Start FastAPI
exec uvicorn main:app --host 0.0.0.0 --port 10000
