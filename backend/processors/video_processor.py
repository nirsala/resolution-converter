"""
Video Processing Pipeline
--------------------------
Two paths:

  Downscale only:
    FFmpeg direct resize (fast, no frame extraction needed).
    Uses Lanczos filter with scale={W}:{H}.

  Upscale path (frame-by-frame):
    1. Probe input with ffprobe
    2. Extract frames as PNG sequence to a temp dir
    3. Run Real-ESRGAN on each frame (with progress updates)
    4. Re-encode with FFmpeg, preserving audio

Safety cap: MAX_VIDEO_FRAMES (default 300) to limit CPU processing time.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from config import settings
from models.job import ConversionStrategy, Job
from processors.upscaler import Upscaler
from processors.saliency import SaliencyDetector
from processors.smart_crop import smart_crop, fit_pad

logger = logging.getLogger(__name__)


def _ensure_even(v: int) -> int:
    return v if v % 2 == 0 else v + 1


def _probe_video(input_path: str) -> dict:
    """Run ffprobe and return stream info."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )

    if not video_stream:
        raise ValueError("No video stream found in input file.")

    # Parse FPS (may be "30/1" or "30000/1001")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, fps_str.split("/"))
    fps = num / den

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "duration": float(info.get("format", {}).get("duration", 0)),
        "has_audio": audio_stream is not None,
        "nb_frames": int(video_stream.get("nb_frames", 0)) or int(fps * float(info.get("format", {}).get("duration", 0))),
    }


class VideoProcessor:

    def __init__(self, job: Job, progress_callback: Optional[Callable[[int], None]] = None):
        self.job = job
        self.target_w = _ensure_even(job.target_width)
        self.target_h = _ensure_even(job.target_height)
        self.strategy = ConversionStrategy(job.strategy)
        self.progress_callback = progress_callback or (lambda p: None)

    def run(self) -> bytes:
        """
        Execute the video pipeline.
        Returns the processed video as raw MP4 bytes.
        """
        logger.info(
            "Processing video job=%s target=%dx%d strategy=%s",
            self.job.id, self.target_w, self.target_h, self.strategy,
        )

        info = _probe_video(self.job.input_path)
        orig_w, orig_h = info["width"], info["height"]
        fps = info["fps"]
        has_audio = info["has_audio"]
        nb_frames = info["nb_frames"]

        logger.info("Input: %dx%d @ %.2ffps, %d frames, audio=%s",
                    orig_w, orig_h, fps, nb_frames, has_audio)

        scale_needed = max(self.target_w / orig_w, self.target_h / orig_h)
        needs_upscale = scale_needed > 1.0 or self.strategy == ConversionStrategy.UPSCALE

        if not needs_upscale:
            # Fast path: FFmpeg-only downscale/reformat
            logger.info("Fast path: FFmpeg resize only.")
            return self._ffmpeg_resize(info)
        else:
            # AI path: frame-by-frame upscaling
            if nb_frames > settings.MAX_VIDEO_FRAMES:
                raise ValueError(
                    f"Video has {nb_frames} frames which exceeds the "
                    f"limit of {settings.MAX_VIDEO_FRAMES} frames "
                    f"({settings.MAX_VIDEO_FRAMES / fps:.0f}s at {fps:.0f}fps). "
                    "Please trim the video or use a downscale strategy."
                )
            logger.info("AI path: frame-by-frame upscaling (%d frames).", nb_frames)
            return self._ai_upscale(info)

    # ── Fast path ─────────────────────────────────────────────────────────────

    def _ffmpeg_resize(self, info: dict) -> bytes:
        """Resize with FFmpeg (downscale or reformat, no AI)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        try:
            vf_filter = self._build_vf_filter(info["width"], info["height"])
            cmd = [
                "ffmpeg", "-y",
                "-i", self.job.input_path,
                "-vf", vf_filter,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
            ]
            if info["has_audio"]:
                cmd += ["-c:a", "aac", "-b:a", "128k"]
            else:
                cmd += ["-an"]
            cmd.append(output_path)

            subprocess.run(cmd, check=True, capture_output=True)
            self.progress_callback(100)
            return Path(output_path).read_bytes()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def _build_vf_filter(self, orig_w: int, orig_h: int) -> str:
        """Build FFmpeg -vf filter string for the chosen strategy."""
        tw, th = self.target_w, self.target_h

        if self.strategy == ConversionStrategy.STRETCH:
            return f"scale={tw}:{th}"

        elif self.strategy == ConversionStrategy.FIT_PAD:
            return (
                f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black"
            )

        elif self.strategy in (ConversionStrategy.SMART_CROP, ConversionStrategy.UPSCALE):
            orig_ar = orig_w / orig_h
            target_ar = tw / th
            if orig_ar > target_ar:
                # Too wide — crop sides
                scaled_w = int(orig_h * target_ar)
                crop_x = (orig_w - scaled_w) // 2
                return f"crop={scaled_w}:{orig_h}:{crop_x}:0,scale={tw}:{th}"
            else:
                # Too tall — crop top/bottom
                scaled_h = int(orig_w / target_ar)
                crop_y = (orig_h - scaled_h) // 2
                return f"crop={orig_w}:{scaled_h}:0:{crop_y},scale={tw}:{th}"

        return f"scale={tw}:{th}"

    # ── AI upscale path ───────────────────────────────────────────────────────

    def _ai_upscale(self, info: dict) -> bytes:
        tmp_dir = tempfile.mkdtemp(prefix="vc_ai_")
        try:
            frames_in_dir = os.path.join(tmp_dir, "in")
            frames_out_dir = os.path.join(tmp_dir, "out")
            os.makedirs(frames_in_dir)
            os.makedirs(frames_out_dir)

            # 1. Extract frames
            logger.info("Extracting frames to %s", frames_in_dir)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", self.job.input_path,
                os.path.join(frames_in_dir, "frame_%05d.png"),
            ], check=True, capture_output=True)

            frames = sorted(Path(frames_in_dir).glob("frame_*.png"))
            total = len(frames)
            logger.info("Extracted %d frames.", total)

            # Prepare saliency/crop on first frame to set crop params
            first_frame = cv2.imread(str(frames[0]))
            orig_h, orig_w = first_frame.shape[:2]

            needs_crop = self.strategy in (ConversionStrategy.SMART_CROP,)
            if needs_crop:
                saliency = SaliencyDetector.get_instance().generate(first_frame)
            else:
                saliency = None

            upscaler = Upscaler.get_instance()

            # 2. Process each frame
            for idx, frame_path in enumerate(frames):
                img = cv2.imread(str(frame_path))

                # Upscale
                img = upscaler.upscale(img)
                up_h, up_w = img.shape[:2]

                # Apply spatial strategy
                if self.strategy == ConversionStrategy.SMART_CROP and saliency is not None:
                    # Upscale the saliency map too
                    sal_up = cv2.resize(saliency, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
                    img = smart_crop(img, self.target_w, self.target_h, sal_up)
                elif self.strategy == ConversionStrategy.FIT_PAD:
                    img = fit_pad(img, self.target_w, self.target_h)
                else:
                    img = cv2.resize(img, (self.target_w, self.target_h),
                                     interpolation=cv2.INTER_LANCZOS4)

                out_path = os.path.join(frames_out_dir, f"frame_{idx + 1:05d}.png")
                cv2.imwrite(out_path, img)

                progress = int((idx + 1) / total * 90)  # 0-90% for frames
                self.progress_callback(progress)

            # 3. Re-encode
            logger.info("Re-encoding video.")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                output_path = tmp.name

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(info["fps"]),
                "-i", os.path.join(frames_out_dir, "frame_%05d.png"),
            ]
            if info["has_audio"]:
                cmd += ["-i", self.job.input_path, "-c:a", "aac", "-b:a", "128k"]
            cmd += [
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
            ]
            if not info["has_audio"]:
                cmd += ["-an"]
            cmd.append(output_path)

            subprocess.run(cmd, check=True, capture_output=True)
            self.progress_callback(100)

            result = Path(output_path).read_bytes()
            os.unlink(output_path)
            return result

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
