"""
Seam Carving — Content-Aware Image Retargeting (no size limits)
---------------------------------------------------------------
Works for ANY aspect-ratio change:

  Reduction  → remove low-energy seams (classic seam carving)
  Expansion  → find k seams first (batch), then insert all at once
               prevents the "same seam repeated" artifact
  Extreme    → multi-pass: seam carve + reflection-fill for large gaps

Processing is done at ≤500px for speed, then upscaled back to target.

Reference: Avidan & Shamir (2007) "Seam Carving for Content-Aware Image Resizing"
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

MAX_PROCESS_DIM   = 500     # max pixels during seam processing (speed)
PROTECTION_ENERGY = 1e5     # added to protected pixels so they're never removed


def _ensure_even(v: int) -> int:
    return v if v % 2 == 0 else v + 1


# ── Energy map ────────────────────────────────────────────────────────────────

def _energy(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(dx) + np.abs(dy)


def _boost(energy: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return energy
    return energy + (mask > 80).astype(np.float32) * PROTECTION_ENERGY


# ── DP: cumulative cost + backtrack ──────────────────────────────────────────

def _dp(energy: np.ndarray) -> np.ndarray:
    h, w = energy.shape
    cost = energy.copy()
    for i in range(1, h):
        left  = np.pad(cost[i-1, :-1], (1, 0), constant_values=np.inf)
        right = np.pad(cost[i-1, 1:],  (0, 1), constant_values=np.inf)
        cost[i] += np.minimum(np.minimum(cost[i-1], left), right)
    return cost


def _trace(cost: np.ndarray) -> np.ndarray:
    h, w = cost.shape
    seam = np.empty(h, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for i in range(h - 2, -1, -1):
        j = seam[i + 1]
        lo, hi = max(0, j - 1), min(w, j + 2)
        seam[i] = lo + int(np.argmin(cost[i, lo:hi]))
    return seam


# ── Remove one vertical seam ─────────────────────────────────────────────────

def _remove_v(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    row  = np.arange(h)
    if img.ndim == 3:
        m = np.ones((h, w, img.shape[2]), dtype=bool)
        m[row, seam] = False
        return img[m].reshape(h, w - 1, img.shape[2])
    else:
        m = np.ones((h, w), dtype=bool)
        m[row, seam] = False
        return img[m].reshape(h, w - 1)


# ── Find k seams (batch, without repeated insertion artifact) ─────────────────

def _find_k_seams(img: np.ndarray,
                  k: int,
                  mask: Optional[np.ndarray]) -> List[np.ndarray]:
    """
    Find k lowest-energy vertical seams simultaneously.
    Uses an index map so we can record original pixel positions.
    Returns a list of k seams, each as original-column indices per row.
    """
    h, w = img.shape[:2]
    cur   = img.copy()
    prot  = mask.copy() if mask is not None else None
    # index_map[row, col] = original column this pixel came from
    idx_map = np.tile(np.arange(w, dtype=np.int32), (h, 1))

    seams_orig: List[np.ndarray] = []
    for _ in range(k):
        e    = _energy(cur)
        e    = _boost(e, prot)
        cost = _dp(e)
        seam = _trace(cost)           # seam in *current* coordinates
        orig = idx_map[np.arange(h), seam]   # map back to original coords
        seams_orig.append(orig)
        cur     = _remove_v(cur, seam)
        idx_map = _remove_v(idx_map, seam)
        if prot is not None:
            prot = _remove_v(prot, seam)
    return seams_orig


# ── Insert k seams at once (no artifact) ─────────────────────────────────────

def _insert_k_seams(img: np.ndarray,
                    seams: List[np.ndarray]) -> np.ndarray:
    """
    Insert k seams into image simultaneously (Avidan & Shamir §4).
    Seams carry original column positions; we sort insertions left→right.
    """
    h, w = img.shape[:2]
    k     = len(seams)
    nc    = img.ndim
    out   = np.zeros((h, w + k, img.shape[2] if nc == 3 else 1),
                     dtype=img.dtype).squeeze(-1) if nc == 2 else \
            np.zeros((h, w + k, img.shape[2]), dtype=img.dtype)

    # Build offset array: how many seams have been inserted left of each orig col
    # For each row, collect all seam positions and sort them
    for row in range(h):
        positions = sorted(s[row] for s in seams)
        src_col = 0
        dst_col = 0
        ins_idx = 0
        src_row = img[row]

        while src_col < w:
            # Insert seam pixel before current src_col if needed
            while ins_idx < k and positions[ins_idx] == src_col:
                # Average current and previous pixel
                prev = src_row[max(0, src_col - 1)]
                curr = src_row[src_col]
                if nc == 3:
                    out[row, dst_col] = ((prev.astype(np.int32) + curr.astype(np.int32)) // 2).astype(img.dtype)
                else:
                    out[row, dst_col] = int((int(prev) + int(curr)) // 2)
                dst_col += 1
                ins_idx += 1

            out[row, dst_col] = src_row[src_col]
            dst_col += 1
            src_col += 1

        # Insert remaining seams at end (right edge)
        while ins_idx < k:
            out[row, dst_col] = src_row[w - 1]
            dst_col += 1
            ins_idx += 1

    return out


# ── Seam ops on full image (vertical & horizontal) ───────────────────────────

def _change_width(img: np.ndarray, target_w: int,
                  mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    cur_w = img.shape[1]
    if cur_w == target_w:
        return img, mask
    if target_w < cur_w:
        # Remove seams one by one
        n = cur_w - target_w
        cur, pm = img.copy(), mask.copy() if mask is not None else None
        for i in range(n):
            e    = _energy(cur)
            e    = _boost(e, pm)
            seam = _trace(_dp(e))
            cur  = _remove_v(cur, seam)
            if pm is not None:
                pm = _remove_v(pm, seam)
            if i % 30 == 0:
                logger.debug("  remove width seam %d/%d", i+1, n)
        return cur, pm
    else:
        # Insert seams in batch
        k   = target_w - cur_w
        logger.debug("  batch insert %d vertical seams", k)
        seams = _find_k_seams(img, k, mask)
        out   = _insert_k_seams(img, seams)
        # We don't update mask for insertion (expand the mask too)
        pm_out = None
        if mask is not None:
            pm_out = _insert_k_seams(mask.reshape(*mask.shape, 1), seams).squeeze(-1) \
                     if mask.ndim == 2 else _insert_k_seams(mask, seams)
        return out, pm_out


def _change_height(img: np.ndarray, target_h: int,
                   mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Transpose → change width → transpose back."""
    def _t(x):
        return np.transpose(x, (1, 0, 2)) if x.ndim == 3 else x.T

    img_t  = _t(img)
    mask_t = _t(mask) if mask is not None else None
    out_t, pm_t = _change_width(img_t, target_h, mask_t)
    return _t(out_t), (_t(pm_t) if pm_t is not None else None)


# ── Reflection fill (for extreme expansions) ─────────────────────────────────

def _reflection_fill(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Place the (already seam-carved) image in the centre of the target canvas.
    Fill the remaining space by tiling reflected copies of the image edge strips.
    Blend the seam between original and fill with a gradient so it's invisible.
    """
    h, w = img.shape[:2]

    if w == target_w and h == target_h:
        return img

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off  = (target_w - w) // 2
    y_off  = (target_h - h) // 2

    # ── Build background by tiling+reflecting the image ──────────────────────
    # Repeat the image enough times to fill the canvas in both directions
    # using reflect-101 (BORDER_REFLECT_101)
    pad_l = x_off
    pad_r = target_w - w - x_off
    pad_t = y_off
    pad_b = target_h - h - y_off

    bg = cv2.copyMakeBorder(
        img,
        pad_t, pad_b, pad_l, pad_r,
        borderType=cv2.BORDER_REFLECT_101,
    )
    # bg is now exactly target_h × target_w
    # Apply blur so the fill looks like a natural extension, not a mirror
    k = 71
    bg_blur = cv2.GaussianBlur(bg, (k, k), 0)
    # Darken slightly so the original stands out
    bg_blur = (bg_blur.astype(np.float32) * 0.75).astype(np.uint8)

    # ── Place original in centre ──────────────────────────────────────────────
    canvas = bg_blur.copy()
    canvas[y_off: y_off + h, x_off: x_off + w] = img

    # ── Blend the seam with a gradient for smooth transition ─────────────────
    blend_px = min(40, x_off, y_off, target_w - w - x_off + 1, target_h - h - y_off + 1)
    if blend_px > 2:
        # Left blend
        if x_off > 0:
            for d in range(blend_px):
                alpha = d / blend_px
                col   = x_off + d
                if col < target_w:
                    canvas[:, col] = (
                        img[:, min(d, w-1)].astype(np.float32) * alpha +
                        bg_blur[:, col].astype(np.float32) * (1 - alpha)
                    ).astype(np.uint8)
        # Right blend
        if pad_r > 0:
            for d in range(blend_px):
                alpha = d / blend_px
                col   = x_off + w - 1 - d
                if 0 <= col < target_w:
                    canvas[:, col] = (
                        img[:, max(0, w-1-d)].astype(np.float32) * alpha +
                        bg_blur[:, col].astype(np.float32) * (1 - alpha)
                    ).astype(np.uint8)

    return canvas


# ── Public API ────────────────────────────────────────────────────────────────

def seam_carve(
    img_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    protection_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Retarget image to (target_w × target_h).
    Works for ANY scale change — no size limits.

    Small/moderate changes: pure seam carving (best quality).
    Large expansions: seam carving + reflection fill with gradient blend.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    logger.info("Seam carve: %dx%d → %dx%d", orig_w, orig_h, target_w, target_h)

    # ── Downscale to processing resolution ───────────────────────────────────
    scale  = min(1.0, MAX_PROCESS_DIM / max(orig_w, orig_h))
    proc_w = _ensure_even(max(4, int(orig_w * scale)))
    proc_h = _ensure_even(max(4, int(orig_h * scale)))
    ptw    = _ensure_even(max(4, int(target_w * scale)))
    pth    = _ensure_even(max(4, int(target_h * scale)))

    small = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    pmask = (cv2.resize(protection_mask, (proc_w, proc_h),
                        interpolation=cv2.INTER_NEAREST)
             if protection_mask is not None else None)

    logger.info("  processing at %dx%d → %dx%d", proc_w, proc_h, ptw, pth)

    # ── Width change ──────────────────────────────────────────────────────────
    logger.info("  width:  %d → %d  (%+d seams)", proc_w, ptw, ptw - proc_w)
    small, pmask = _change_width(small, ptw, pmask)

    # ── Height change ─────────────────────────────────────────────────────────
    logger.info("  height: %d → %d  (%+d seams)", proc_h, pth, pth - proc_h)
    small, pmask = _change_height(small, pth, pmask)

    # ── Reflection fill for extreme expansions ────────────────────────────────
    if small.shape[1] != ptw or small.shape[0] != pth:
        small = _reflection_fill(small, ptw, pth)

    # ── Upscale to final target ───────────────────────────────────────────────
    result = cv2.resize(small, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    logger.info("  done: %dx%d", target_w, target_h)
    return result
