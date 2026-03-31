"""
Image Processing Pipeline
--------------------------
Decision tree:

  1. Is upscaling needed?
     → Yes: run Real-ESRGAN ×4, then apply strategy
     → No:  go to step 2

  2. Aspect ratios match (within 5%)?
     → Yes: simple Lanczos resize, done

  3. Apply strategy:
     recompose  → ⭐ Decompose layers (fg/bg) → rebuild at target size
     fit_blur   → All content visible + blurred background fill
     fit_pad    → All content visible + black bars
     smart_crop → saliency-guided crop (may cut edges)
     upscale    → Real-ESRGAN + fit_blur (no distortion)
     stretch    → direct stretch (distorts AR)
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
from processors.smart_crop import smart_crop, fit_pad, fit_blur
from processors.recompose import recompose

logger = logging.getLogger(__name__)

AR_TOLERANCE = 0.05  # 5% aspect-ratio tolerance before strategy is applied


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
        """Execute the full pipeline. Returns processed image as raw bytes."""
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
            logger.info("Already at target size.")
            ext = Path(self.job.original_filename).suffix or ".jpg"
            return _save_image(img, ext)

        # ── Step 3: Aspect ratio check ────────────────────────────────────────
        orig_ar = orig_w / orig_h
        target_ar = self.target_w / self.target_h
        ar_diff = abs(orig_ar - target_ar) / target_ar

        if ar_diff < AR_TOLERANCE:
            # ARs close enough — simple Lanczos resize (no distortion)
            logger.info("ARs match (%.3f vs %.3f). Simple resize.", orig_ar, target_ar)
            result = cv2.resize(img, (self.target_w, self.target_h),
                                interpolation=cv2.INTER_LANCZOS4)
        else:
            result = self._apply_strategy(img)

        ext = Path(self.job.original_filename).suffix or ".jpg"
        return _save_image(result, ext)

    def _apply_strategy(self, img: np.ndarray) -> np.ndarray:

        if self.strategy == ConversionStrategy.RECOMPOSE:
            # ⭐ Decompose foreground/background → rebuild at target size
            logger.info("Strategy: recompose — decompose layers + rebuild.")
            saliency = SaliencyDetector.get_instance().generate(img)
            return recompose(img, self.target_w, self.target_h, saliency)

        elif self.strategy == ConversionStrategy.FIT_BLUR:
            # ⭐ Best default: all content visible, blurred background fills the frame
            logger.info("Strategy: fit_blur — all content visible + blur background.")
            return fit_blur(img, self.target_w, self.target_h)

        elif self.strategy == ConversionStrategy.FIT_PAD:
            logger.info("Strategy: fit_pad — letterbox with black bars.")
            return fit_pad(img, self.target_w, self.target_h)

        elif self.strategy == ConversionStrategy.SMART_CROP:
            logger.info("Strategy: smart_crop — AI saliency crop.")
            saliency = SaliencyDetector.get_instance().generate(img)
            return smart_crop(img, self.target_w, self.target_h, saliency)

        elif self.strategy == ConversionStrategy.UPSCALE:
            # Already upscaled in Step 1 — use fit_blur so AR is preserved
            logger.info("Strategy: upscale — fit_blur after ESRGAN (no distortion).")
            return fit_blur(img, self.target_w, self.target_h)

        elif self.strategy == ConversionStrategy.STRETCH:
            logger.info("Strategy: stretch — direct resize (distorts AR).")
            return cv2.resize(img, (self.target_w, self.target_h),
                              interpolation=cv2.INTER_LANCZOS4)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
