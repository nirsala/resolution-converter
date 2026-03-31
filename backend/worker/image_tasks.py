import logging
from datetime import datetime

from sqlmodel import Session, select

from database import engine
from models.job import Job, JobStatus
from processors.image_processor import ImageProcessor
from storage import storage
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2, queue="image", name="worker.image_tasks.process_image")
def process_image(self, job_id: str):
    """Celery task: load job → process image → save output → update DB."""

    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            logger.error("Job %s not found.", job_id)
            return

        # Mark as processing
        job.status = JobStatus.PROCESSING
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)

    try:
        processor = ImageProcessor(job)
        output_bytes = processor.run()

        # Build output filename
        import os
        ext = os.path.splitext(job.original_filename)[1] or ".jpg"
        output_filename = f"{job.id}_output{ext}"
        output_path = storage.save_output(output_bytes, output_filename)

        with Session(engine) as session:
            job = session.get(Job, job_id)
            job.status = JobStatus.DONE
            job.output_path = output_path
            job.progress = 100
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()

        logger.info("Image job %s completed. Output: %s", job_id, output_path)

    except Exception as exc:
        logger.exception("Image job %s failed: %s", job_id, exc)
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()
        raise self.retry(exc=exc, countdown=5)
