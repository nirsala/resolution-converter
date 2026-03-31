"""
Seam Carving — Content-Aware Image Retargeting
-----------------------------------------------
Classic Avidan & Shamir (2007) algorithm with U2Net-based protection mask.

How it works:
  1. Compute energy map = gradient magnitude (edges = high energy = important)
  2. Optionally boost energy of detected foreground objects (U2Net mask)
     → protected pixels are never removed
  3. Find minimum-cost vertical or horizontal seam using dynamic programming
  4. Remove (or duplicate) that seam
  5. Repeat until target dimensions are reached

For large images, we process at a reduced resolution and upscale back.
For extreme aspect-ratio changes (>50% in one dimension), we warn and
fall back to fit_blur to avoid poor quality.

Performance: processed at max 500px to stay within seconds on CPU.
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Max pixels in either dimension during seam processing (for speed)
MAX_PROCESS_DIM = 500
# Protection weight for important pixels
PROTECTION_ENERGY = 1e5
# Max change ratio before falling back to fit_blur
MAX_SEAM_RATIO = 0.55


def _ensure_even(v: int) -> int:
    return v if v % 2 == 0 else v + 1


# ── Energy computation ────────────────────────────────────────────────────────

def _energy_map(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gradient-based energy map.
    High energy = visually important (edges, textures, subjects).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(dx) + np.abs(dy)


def _apply_protection(energy: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Boost energy of protected (foreground) pixels so they are never removed."""
    if mask is None:
        return energy
    protected = (mask > 80).astype(np.float32)
    return energy + protected * PROTECTION_ENERGY


# ── Seam finding (vertical — removes one column of pixels) ───────────────────

def _cumulative_cost(energy: np.ndarray) -> np.ndarray:
    """DP: compute minimum cumulative energy for each pixel (top-down)."""
    h, w = energy.shape
    cost = energy.copy()
    for i in range(1, h):
        # Shift left and right for neighbour comparison
        left  = np.pad(cost[i-1, :-1], (1, 0), constant_values=np.inf)
        right = np.pad(cost[i-1, 1:],  (0, 1), constant_values=np.inf)
        cost[i] += np.minimum(np.minimum(cost[i-1], left), right)
    return cost


def _trace_seam(cost: np.ndarray) -> np.ndarray:
    """Backtrack from bottom row to find the minimum-cost vertical seam."""
    h, w = cost.shape
    seam = np.empty(h, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for i in range(h - 2, -1, -1):
        j = seam[i + 1]
        lo = max(0, j - 1)
        hi = min(w, j + 2)
        seam[i] = lo + int(np.argmin(cost[i, lo:hi]))
    return seam


def _remove_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Remove one vertical seam from the image (or single-channel mask)."""
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3
    row_idx = np.arange(h)

    if is_color:
        mask3 = np.ones((h, w, img.shape[2]), dtype=bool)
        mask3[row_idx, seam] = False
        return img[mask3].reshape(h, w - 1, img.shape[2])
    else:
        mask1 = np.ones((h, w), dtype=bool)
        mask1[row_idx, seam] = False
        return img[mask1].reshape(h, w - 1)


def _duplicate_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Insert one vertical seam (duplicate adjacent pixel average) to widen image."""
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3
    out_shape = (h, w + 1, img.shape[2]) if is_color else (h, w + 1)
    out = np.zeros(out_shape, dtype=img.dtype)

    for i in range(h):
        j = seam[i]
        if is_color:
            out[i, :j] = img[i, :j]
            # Insert average of pixel and its right neighbour
            right = min(j + 1, w - 1)
            out[i, j] = ((img[i, j].astype(np.int32) + img[i, right].astype(np.int32)) // 2).astype(img.dtype)
            out[i, j + 1:] = img[i, j:]
        else:
            out[i, :j] = img[i, :j]
            right = min(j + 1, w - 1)
            out[i, j] = int((int(img[i, j]) + int(img[i, right])) // 2)
            out[i, j + 1:] = img[i, j:]
    return out


# ── Horizontal seam (transpose trick) ────────────────────────────────────────

def _remove_horizontal_seam(img: np.ndarray, mask: Optional[np.ndarray],
                             energy_boost: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img_t = np.transpose(img, (1, 0, 2)) if len(img.shape) == 3 else img.T
    energy = _energy_map(img_t if len(img_t.shape) == 3 else
                         cv2.cvtColor(np.stack([img_t]*3, axis=-1), cv2.COLOR_BGR2GRAY)
                         .reshape(img_t.shape))

    mask_t = mask.T if mask is not None else None
    energy = _apply_protection(energy, mask_t)
    cost = _cumulative_cost(energy)
    seam = _trace_seam(cost)
    img_out = _remove_vertical_seam(img_t, seam)
    mask_out = _remove_vertical_seam(mask_t, seam) if mask_t is not None else None

    # Transpose back
    img_out = np.transpose(img_out, (1, 0, 2)) if len(img_out.shape) == 3 else img_out.T
    mask_out = mask_out.T if mask_out is not None else None
    return img_out, mask_out


def _duplicate_horizontal_seam(img: np.ndarray) -> np.ndarray:
    img_t = np.transpose(img, (1, 0, 2)) if len(img.shape) == 3 else img.T
    energy = _energy_map(img_t)
    cost = _cumulative_cost(energy)
    seam = _trace_seam(cost)
    out_t = _duplicate_vertical_seam(img_t, seam)
    return np.transpose(out_t, (1, 0, 2)) if len(out_t.shape) == 3 else out_t.T


# ── Main seam-carving function ────────────────────────────────────────────────

def seam_carve(
    img_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    protection_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Retarget image to (target_w × target_h) using seam carving.

    Parameters
    ----------
    img_bgr          : input image (BGR, uint8)
    target_w/h       : desired output size
    protection_mask  : H×W uint8 mask — high values = protected pixels (never removed)

    Returns
    -------
    Retargeted image at (target_w × target_h).
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # ── Validate change ratios ────────────────────────────────────────────────
    w_ratio = abs(target_w - orig_w) / orig_w
    h_ratio = abs(target_h - orig_h) / orig_h

    if w_ratio > MAX_SEAM_RATIO or h_ratio > MAX_SEAM_RATIO:
        logger.warning(
            "Aspect ratio change too large for seam carving "
            "(w: %.0f%%, h: %.0f%%). Falling back to fit_blur.",
            w_ratio * 100, h_ratio * 100,
        )
        from processors.smart_crop import fit_blur
        return fit_blur(img_bgr, target_w, target_h)

    # ── Downscale for processing speed ───────────────────────────────────────
    scale = min(1.0, MAX_PROCESS_DIM / max(orig_w, orig_h))
    proc_w = _ensure_even(int(orig_w * scale))
    proc_h = _ensure_even(int(orig_h * scale))
    proc_tw = _ensure_even(int(target_w * scale))
    proc_th = _ensure_even(int(target_h * scale))

    img_small = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    mask_small = (
        cv2.resize(protection_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        if protection_mask is not None else None
    )

    logger.info(
        "Seam carving: %dx%d → %dx%d  (processing at %dx%d → %dx%d)",
        orig_w, orig_h, target_w, target_h,
        proc_w, proc_h, proc_tw, proc_th,
    )

    result = img_small
    prot   = mask_small

    # ── Remove/add vertical seams (change width) ──────────────────────────────
    dw = proc_tw - proc_w
    if dw < 0:
        logger.info("Removing %d vertical seams.", -dw)
        for step in range(-dw):
            energy = _energy_map(result)
            energy = _apply_protection(energy, prot)
            cost   = _cumulative_cost(energy)
            seam   = _trace_seam(cost)
            result = _remove_vertical_seam(result, seam)
            if prot is not None:
                prot = _remove_vertical_seam(prot, seam)
            if step % 20 == 0:
                logger.debug("  width seam %d/%d", step + 1, -dw)
    elif dw > 0:
        logger.info("Inserting %d vertical seams.", dw)
        for step in range(dw):
            energy = _energy_map(result)
            energy = _apply_protection(energy, prot)
            cost   = _cumulative_cost(energy)
            seam   = _trace_seam(cost)
            result = _duplicate_vertical_seam(result, seam)
            if prot is not None:
                prot = _duplicate_vertical_seam(prot.reshape(*prot.shape, 1), seam).squeeze(-1) \
                       if len(prot.shape) == 2 else _duplicate_vertical_seam(prot, seam)
            if step % 20 == 0:
                logger.debug("  width seam +%d/%d", step + 1, dw)

    # ── Remove/add horizontal seams (change height) ───────────────────────────
    dh = proc_th - proc_h
    if dh < 0:
        logger.info("Removing %d horizontal seams.", -dh)
        for step in range(-dh):
            result, prot = _remove_horizontal_seam(result, prot)
            if step % 20 == 0:
                logger.debug("  height seam %d/%d", step + 1, -dh)
    elif dh > 0:
        logger.info("Inserting %d horizontal seams.", dh)
        for step in range(dh):
            result = _duplicate_horizontal_seam(result)
            if step % 20 == 0:
                logger.debug("  height seam +%d/%d", step + 1, dh)

    # ── Upscale result to true target size ────────────────────────────────────
    if result.shape[1] != target_w or result.shape[0] != target_h:
        result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    logger.info("Seam carving done: %dx%d", result.shape[1], result.shape[0])
    return result
