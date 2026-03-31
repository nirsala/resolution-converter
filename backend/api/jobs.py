"""
Job API
-------
POST /api/jobs   — upload file + create conversion job
GET  /api/jobs/{job_id} — poll job status
GET  /api/jobs   — list recent jobs (last 50)
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import magic
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select

from config import settings
from database import get_session
from models.job import ConversionStrategy, Job, JobResponse, JobStatus
from storage import storage

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed MIME types → (input_type, extension)
ALLOWED_MIMES = {
    "image/jpeg": ("image", ".jpg"),
    "image/png": ("image", ".png"),
    "image/webp": ("image", ".webp"),
    "image/tiff": ("image", ".tiff"),
    "image/bmp": ("image", ".bmp"),
    "video/mp4": ("video", ".mp4"),
    "video/quicktime": ("video", ".mov"),
    "video/x-msvideo": ("video", ".avi"),
    "video/x-matroska": ("video", ".mkv"),
    "video/webm": ("video", ".webm"),
}


def _to_response(job: Job) -> JobResponse:
    download_url = None
    if job.status == JobStatus.DONE and job.output_path:
        download_url = storage.get_download_url(job.output_path)

    return JobResponse(
        id=job.id,
        status=job.status,
        input_type=job.input_type,
        original_filename=job.original_filename,
        target_width=job.target_width,
        target_height=job.target_height,
        strategy=job.strategy,
        progress=job.progress,
        download_url=download_url,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/jobs", response_model=JobResponse, status_code=202)
async def create_job(
    file: UploadFile = File(...),
    target_width: int = Form(...),
    target_height: int = Form(...),
    strategy: ConversionStrategy = Form(ConversionStrategy.SMART_CROP),
    session: Session = Depends(get_session),
):
    # ── Validate dimensions ────────────────────────────────────────────────
    if not (1 <= target_width <= 7680) or not (1 <= target_height <= 4320):
        raise HTTPException(400, "target_width/height must be between 1 and 7680/4320")

    # ── Read file bytes ────────────────────────────────────────────────────
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    # ── Detect MIME from magic bytes (not just extension) ──────────────────
    mime = magic.from_buffer(file_bytes[:2048], mime=True)
    if mime not in ALLOWED_MIMES:
        raise HTTPException(
            415,
            f"Unsupported file type: {mime}. "
            "Allowed: JPEG, PNG, WebP, TIFF, BMP, MP4, MOV, AVI, MKV, WebM.",
        )

    input_type, ext = ALLOWED_MIMES[mime]

    # ── File size limit ────────────────────────────────────────────────────
    limit_mb = settings.MAX_IMAGE_SIZE_MB if input_type == "image" else settings.MAX_VIDEO_SIZE_MB
    if file_size_mb > limit_mb:
        raise HTTPException(413, f"File too large: {file_size_mb:.1f}MB (limit {limit_mb}MB)")

    # ── Save upload ────────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    upload_filename = f"{job_id}_input{ext}"
    input_path = storage.save_upload(file_bytes, upload_filename)

    # ── Persist job ────────────────────────────────────────────────────────
    job = Job(
        id=job_id,
        input_type=input_type,
        original_filename=file.filename or f"upload{ext}",
        input_path=input_path,
        target_width=target_width,
        target_height=target_height,
        strategy=strategy,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # ── Dispatch to Celery queue ───────────────────────────────────────────
    if input_type == "image":
        from worker.image_tasks import process_image
        process_image.apply_async(args=[job_id], queue="image")
    else:
        from worker.video_tasks import process_video
        process_video.apply_async(args=[job_id], queue="video")

    logger.info("Created %s job %s → %dx%d (%s)", input_type, job_id, target_width, target_height, strategy)
    return _to_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id!r} not found.")
    return _to_response(job)


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(session: Session = Depends(get_session)):
    jobs = session.exec(
        select(Job).order_by(Job.created_at.desc()).limit(50)
    ).all()
    return [_to_response(j) for j in jobs]
