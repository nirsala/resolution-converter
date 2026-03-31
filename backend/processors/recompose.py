"""
Decompose & Recompose
---------------------
The "true" resolution adaptation:

  1. DECOMPOSE — separate image into layers using U2Net saliency:
       - Foreground layer  (subjects, people, objects)
       - Background layer  (scenery, environment)

  2. REBUILD BACKGROUND — scale the background to fill the target canvas.
     Uses cv2.inpaint to fill the area behind the subject first,
     then scales the clean background to target size.

  3. RECOMPOSE — place the scaled foreground back on the background,
     centred, with a soft alpha blend on the edges.

Result: every element is preserved, the image adapts to the new
aspect ratio without cropping or black bars, and the subject stays sharp.
"""

import logging
import numpy as np
import cv2
from typing import Tuple

logger = logging.getLogger(__name__)


def _ensure_even(v: int) -> int:
    return v if v % 2 == 0 else v + 1


def _build_fg_mask(saliency: np.ndarray, threshold: int = 80) -> np.ndarray:
    """
    Convert saliency map → clean foreground binary mask.
    Applies threshold + morphological closing to fill holes inside objects.
    """
    _, binary = cv2.threshold(saliency, threshold, 255, cv2.THRESH_BINARY)

    # Close small holes inside the subject
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate a bit to get a slightly generous mask (helps inpainting)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    return dilated  # uint8, 0 or 255


def _inpaint_background(img_bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Remove the foreground and fill the hole using OpenCV Telea inpainting.
    Returns a clean background image (no subjects).
    """
    # Inpainting works best with a slightly dilated mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    inpaint_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    logger.info("Inpainting background (removing foreground)...")
    bg_clean = cv2.inpaint(img_bgr, inpaint_mask, inpaintRadius=20, flags=cv2.INPAINT_TELEA)
    return bg_clean


def _scale_foreground(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    target_w: int,
    target_h: int,
    padding: float = 0.92,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Scale the full original image (with the foreground embedded) to fit
    inside the target canvas while preserving aspect ratio.

    Returns:
        scaled_img   - the scaled image
        scaled_mask  - the scaled foreground mask
        x_offset     - where to place it on the canvas (x)
        y_offset     - where to place it on the canvas (y)
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # Scale to fit inside target (contain), with small padding
    scale = min(target_w / orig_w, target_h / orig_h) * padding
    new_w = _ensure_even(max(1, int(orig_w * scale)))
    new_h = _ensure_even(max(1, int(orig_h * scale)))

    scaled_img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    scaled_mask = cv2.resize(fg_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2

    return scaled_img, scaled_mask, x_off, y_off


def _composite(
    bg: np.ndarray,
    fg_img: np.ndarray,
    fg_mask: np.ndarray,
    x_off: int,
    y_off: int,
) -> np.ndarray:
    """
    Soft-composite the foreground (with its mask) onto the background.
    Uses a Gaussian-softened alpha so there are no hard edges.
    """
    canvas = bg.copy()
    fg_h, fg_w = fg_img.shape[:2]

    # Soft alpha from mask
    alpha_float = cv2.GaussianBlur(
        fg_mask.astype(np.float32), (31, 31), 0
    ) / 255.0
    alpha = alpha_float[:, :, np.newaxis]  # (H, W, 1)

    # ROI in canvas
    roi = canvas[y_off: y_off + fg_h, x_off: x_off + fg_w]

    blended = (fg_img.astype(np.float32) * alpha +
               roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    canvas[y_off: y_off + fg_h, x_off: x_off + fg_w] = blended
    return canvas


def recompose(
    img_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    saliency_map: np.ndarray,
) -> np.ndarray:
    """
    Full Decompose → Rebuild → Recompose pipeline.

    Parameters
    ----------
    img_bgr      : original image (BGR, uint8)
    target_w/h   : desired output dimensions
    saliency_map : H×W uint8 saliency (from SaliencyDetector)

    Returns
    -------
    Recomposed image at (target_w × target_h), all subjects preserved.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    logger.info("Recompose: %dx%d → %dx%d", orig_w, orig_h, target_w, target_h)

    # ── Step 1: Build foreground mask ────────────────────────────────────────
    fg_mask = _build_fg_mask(saliency_map)
    fg_coverage = np.count_nonzero(fg_mask) / fg_mask.size

    logger.info("Foreground coverage: %.1f%%", fg_coverage * 100)

    # ── Step 2: Build clean background ───────────────────────────────────────
    # If foreground covers <90% of the image, inpaint it out of the background
    if fg_coverage < 0.90:
        bg_clean = _inpaint_background(img_bgr, fg_mask)
    else:
        # Whole image is "foreground" (e.g. a portrait filling the frame)
        # Use the original as background too
        bg_clean = img_bgr.copy()

    # ── Step 3: Scale background to fill entire target canvas ─────────────────
    # Cover mode: fill every pixel, may crop a little at the edges
    scale_cover = max(target_w / orig_w, target_h / orig_h)
    bg_w = _ensure_even(int(orig_w * scale_cover))
    bg_h = _ensure_even(int(orig_h * scale_cover))
    bg_scaled = cv2.resize(bg_clean, (bg_w, bg_h), interpolation=cv2.INTER_LANCZOS4)

    # Centre-crop background to exact target
    bx = (bg_w - target_w) // 2
    by = (bg_h - target_h) // 2
    background = bg_scaled[by: by + target_h, bx: bx + target_w]

    # ── Step 4: Scale foreground (fit, no crop) ───────────────────────────────
    fg_scaled, fg_mask_scaled, x_off, y_off = _scale_foreground(
        img_bgr, fg_mask, target_w, target_h
    )

    # ── Step 5: Composite ─────────────────────────────────────────────────────
    result = _composite(background, fg_scaled, fg_mask_scaled, x_off, y_off)

    logger.info("Recompose complete: %dx%d", target_w, target_h)
    return result
