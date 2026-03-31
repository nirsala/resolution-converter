"""
Image Processing Pipeline
--------------------------
Decision tree:

  1. Is upscaling needed?
     → Yes: run Real-ESRGAN ×4, then downscale to target if needed
     → No:  go to step 2

  2. Aspect ratios match (within 5%)?
     → Yes: simple Lanczos resize, done

  3. Apply strategy:
     smart_crop → saliency-guided crop + resize
     fit_pad    → letterbox/pillarbox + resize
     upscale    → Real-ESRGAN only (resize to exact target after)
     stretch    → direct stretch to target (no AR preservation)
"""

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from models.job import ConversionStrategy, Job
from processors.upscaler import Upscaler
from processors.saliency import SaliencyDetector
from processors.smart_crop import smart_crop, fit_pad

logger = logging.getLogger(__name__)

AR_TOLERANCE = 0.05  # 5% aspect-ratio tolerance before cropping is needed


def _ensure_even(v: int) -> int:
    return v if v % 2 == 0 else v + 1


def _load_image(path: str) -> np.ndarray:
    """Load any Pillow-supported format into a BGR numpy array (cv2 convention)."""
    with Image.open(path) as pil_img:
        pil_img = pil_img.convert("RGB")
        img_rgb = np.array(pil_img)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _save_image(img_bgr: np.ndarray, ext: str) -> bytes:
    """Encode BGR image to bytes in the given format."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    fmt = ext.upper().lstrip(".")
    if fmt == "JPG":
        fmt = "JPEG"
    pil_img.save(buf, format=fmt, quality=95)
    return buf.getvalue()


class ImageProcessor:

    def __init__(self, job: Job):
        self.job = job
        self.target_w = _ensure_even(job.target_width)
        self.target_h = _ensure_even(job.target_height)
        self.strategy = ConversionStrategy(job.strategy)

    def run(self) -> bytes:
        """
        Execute the full pipeline.
        Returns the processed image as raw bytes (JPEG/PNG matching input ext).
        """
        logger.info(
            "Processing image job=%s target=%dx%d strategy=%s",
            self.job.id, self.target_w, self.target_h, self.strategy,
        )

        img = _load_image(self.job.input_path)
        orig_h, orig_w = img.shape[:2]
        logger.info("Input size: %dx%d", orig_w, orig_h)

        # ── Step 1: Upscaling decision ────────────────────────────────────────
        scale_needed = max(self.target_w / orig_w, self.target_h / orig_h)
        if scale_needed > 1.0 or self.strategy == ConversionStrategy.UPSCALE:
            logger.info("Upscaling required (scale=%.2fx). Running Real-ESRGAN.", scale_needed)
            img = Upscaler.get_instance().upscale(img)
            orig_h, orig_w = img.shape[:2]
            logger.info("After upscale: %dx%d", orig_w, orig_h)

        # ── Step 2: Exact size already? ───────────────────────────────────────
        if orig_w == self.target_w and orig_h == self.target_h:
            logger.info("Already at target size, skipping resize.")
            ext = Path(self.job.original_filename).suffix or ".jpg"
            return _save_image(img, ext)

        # ── Step 3: Aspect ratio check ────────────────────────────────────────
        orig_ar = orig_w / orig_h
        target_ar = self.target_w / self.target_h

        if abs(orig_ar - target_ar) / target_ar < AR_TOLERANCE:
            # ARs are close enough — simple resize suffices
            logger.info("ARs match (%.3f vs %.3f). Simple Lanczos resize.", orig_ar, target_ar)
            result = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            result = self._apply_strategy(img)

        ext = Path(self.job.original_filename).suffix or ".jpg"
        return _save_image(result, ext)

    def _apply_strategy(self, img: np.ndarray) -> np.ndarray:
        orig_h, orig_w = img.shape[:2]

        if self.strategy == ConversionStrategy.SMART_CROP:
            logger.info("Strategy: smart_crop — generating saliency map.")
            saliency = SaliencyDetector.get_instance().generate(img)
            return smart_crop(img, self.target_w, self.target_h, saliency)

        elif self.strategy == ConversionStrategy.FIT_PAD:
            logger.info("Strategy: fit_pad — letterboxing.")
            return fit_pad(img, self.target_w, self.target_h)

        elif self.strategy == ConversionStrategy.UPSCALE:
            # Already upscaled in Step 1; just resize to exact target
            logger.info("Strategy: upscale — resizing to exact target after ESRGAN.")
            return cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)

        elif self.strategy == ConversionStrategy.STRETCH:
            logger.info("Strategy: stretch — direct resize (no AR preservation).")
            return cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
