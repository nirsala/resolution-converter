"""
Smart (Content-Aware) Crop
--------------------------
Uses a saliency map to determine the best crop window
that fits the target aspect ratio while keeping the important
subject(s) fully inside the crop.
"""

import numpy as np
import cv2
from typing import Tuple


def _ensure_even(v: int) -> int:
    """FFmpeg and some codecs require even dimensions."""
    return v if v % 2 == 0 else v + 1


def compute_smart_crop_box(
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    saliency_map: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Compute the best axis-aligned crop box.

    Parameters
    ----------
    orig_w, orig_h  : original image dimensions
    target_w, target_h : desired output dimensions
    saliency_map    : H×W uint8 saliency mask (0=background, 255=salient)

    Returns
    -------
    (x1, y1, x2, y2)  crop box in original image coordinates
    """
    target_ar = target_w / target_h

    # ── Crop size in original-image pixels ──────────────────────────────────
    # We want the largest crop that fits inside the image and has the target AR
    if orig_w / orig_h > target_ar:
        # Image is wider than target → crop width
        crop_h = orig_h
        crop_w = int(round(crop_h * target_ar))
    else:
        # Image is taller than target → crop height
        crop_w = orig_w
        crop_h = int(round(crop_w / target_ar))

    crop_w = min(_ensure_even(crop_w), orig_w)
    crop_h = min(_ensure_even(crop_h), orig_h)

    # ── Find the salient region bounding box ─────────────────────────────────
    threshold = 128
    binary = (saliency_map >= threshold).astype(np.uint8)

    coords = cv2.findNonZero(binary)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        # Centroid of the salient region
        cx = bx + bw // 2
        cy = by + bh // 2
    else:
        # No salient region found → center crop
        cx, cy = orig_w // 2, orig_h // 2

    # ── Place crop window centred on the salient region ──────────────────────
    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2

    # Clamp to image bounds
    x1 = max(0, min(x1, orig_w - crop_w))
    y1 = max(0, min(y1, orig_h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    return x1, y1, x2, y2


def smart_crop(
    img_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    saliency_map: np.ndarray,
) -> np.ndarray:
    """
    Crop + resize an image to (target_w × target_h) using saliency guidance.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    x1, y1, x2, y2 = compute_smart_crop_box(orig_w, orig_h, target_w, target_h, saliency_map)
    cropped = img_bgr[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return resized


def fit_pad(
    img_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Fit image inside (target_w × target_h) with letterbox/pillarbox padding.
    Preserves the full original image, no cropping.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = _ensure_even(int(orig_w * scale))
    new_h = _ensure_even(int(orig_h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create padded canvas
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off: y_off + new_h, x_off: x_off + new_w] = resized
    return canvas


def fit_blur(
    img_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    blur_strength: int = 55,
) -> np.ndarray:
    """
    Fit + Blur Background — ALL content visible, no black bars, no cropping.

    How it works:
      1. Scale the original image to FILL the entire target (cover mode) → blurred bg
      2. Scale the original image to FIT inside the target (contain mode) → sharp foreground
      3. Place the sharp foreground centered on the blurred background

    Result: every pixel of the original is visible, aspect ratio preserved,
    and the empty areas are filled with a beautiful blurred version of the image.
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # ── Background: scale to COVER (fill entire canvas, may overflow) ─────────
    scale_cover = max(target_w / orig_w, target_h / orig_h)
    bg_w = _ensure_even(int(orig_w * scale_cover))
    bg_h = _ensure_even(int(orig_h * scale_cover))
    bg = cv2.resize(img_bgr, (bg_w, bg_h), interpolation=cv2.INTER_LANCZOS4)

    # Crop background to exact target size (centred)
    bx = (bg_w - target_w) // 2
    by = (bg_h - target_h) // 2
    bg = bg[by: by + target_h, bx: bx + target_w]

    # Apply heavy Gaussian blur
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    bg = cv2.GaussianBlur(bg, (k, k), 0)

    # Darken background slightly so the foreground pops
    bg = (bg * 0.6).astype(np.uint8)

    # ── Foreground: scale to FIT (contain, no cropping) ───────────────────────
    scale_fit = min(target_w / orig_w, target_h / orig_h)
    fg_w = _ensure_even(int(orig_w * scale_fit))
    fg_h = _ensure_even(int(orig_h * scale_fit))
    fg = cv2.resize(img_bgr, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)

    # ── Composite: place foreground centred on background ─────────────────────
    y_off = (target_h - fg_h) // 2
    x_off = (target_w - fg_w) // 2
    canvas = bg.copy()
    canvas[y_off: y_off + fg_h, x_off: x_off + fg_w] = fg

    return canvas
