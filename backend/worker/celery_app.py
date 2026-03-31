from celery import Celery
from config import settings

celery_app = Celery(
    "converter",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["worker.image_tasks", "worker.video_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_expires=3600,
    worker_prefetch_multiplier=1,   # one task at a time per slot — important for GPU
    task_acks_late=True,            # re-queue on worker crash
    task_track_started=True,
)
