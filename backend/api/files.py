"""
File download endpoint (local storage only).
For S3, the download URL is a presigned URL returned directly by the storage backend.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings

router = APIRouter()

BASE = Path(settings.LOCAL_STORAGE_PATH)


@router.get("/files/outputs/{filename}")
def download_output(filename: str):
    # Security: no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")

    path = BASE / "outputs" / filename
    if not path.exists():
        raise HTTPException(404, "File not found.")

    # Determine media type
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=filename,
    )
