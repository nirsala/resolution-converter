import logging
from datetime import datetime

from sqlmodel import Session

from database import engine
from models.job import Job, JobStatus
from processors.video_processor import VideoProcessor
from storage import storage
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=1, queue="video", name="worker.video_tasks.process_video")
def process_video(self, job_id: str):
    """Celery task: load job → process video → save output → update DB."""

    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            logger.error("Job %s not found.", job_id)
            return
        job.status = JobStatus.PROCESSING
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)

    def progress_callback(pct: int):
        """Write frame progress to DB so the frontend can show a progress bar."""
        with Session(engine) as s:
            j = s.get(Job, job_id)
            if j:
                j.progress = pct
                j.updated_at = datetime.utcnow()
                s.add(j)
                s.commit()

    try:
        processor = VideoProcessor(job, progress_callback=progress_callback)
        output_bytes = processor.run()

        output_filename = f"{job_id}_output.mp4"
        output_path = storage.save_output(output_bytes, output_filename)

        with Session(engine) as session:
            job = session.get(Job, job_id)
            job.status = JobStatus.DONE
            job.output_path = output_path
            job.progress = 100
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()

        logger.info("Video job %s completed. Output: %s", job_id, output_path)

    except Exception as exc:
        logger.exception("Video job %s failed: %s", job_id, exc)
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()
        raise self.retry(exc=exc, countdown=5)
