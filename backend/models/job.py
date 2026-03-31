from sqlmodel import SQLModel, Field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ConversionStrategy(str, Enum):
    FIT_BLUR = "fit_blur"           # ⭐ All content visible + blurred background fill
    FIT_PAD = "fit_pad"             # All content visible + black bars
    SMART_CROP = "smart_crop"       # AI saliency crop (may cut edges)
    UPSCALE = "upscale"             # AI upscale (Real-ESRGAN) + fit_blur
    STRETCH = "stretch"             # Direct stretch (distorts AR)


class Job(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status: JobStatus = JobStatus.PENDING
    input_type: str                         # "image" or "video"
    original_filename: str
    input_path: str
    output_path: Optional[str] = None
    target_width: int
    target_height: int
    strategy: str = ConversionStrategy.SMART_CROP
    progress: int = 0                       # 0-100 (used for video frame progress)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Request schema ────────────────────────────────────────────────────────────
class CreateJobRequest(SQLModel):
    target_width: int
    target_height: int
    strategy: ConversionStrategy = ConversionStrategy.SMART_CROP


# ── Response schema (no internal paths exposed) ───────────────────────────────
class JobResponse(SQLModel):
    id: str
    status: JobStatus
    input_type: str
    original_filename: str
    target_width: int
    target_height: int
    strategy: str
    progress: int
    download_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
