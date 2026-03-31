#!/bin/sh
set -e

echo "⚙️  Starting Celery Worker..."

# Download models if missing
sh /app/scripts/download_models.sh

# Start Celery with image + video queues
exec celery -A worker.celery_app worker \
  --loglevel=info \
  --concurrency=2 \
  -Q image,video
