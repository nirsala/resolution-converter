"""
Real-ESRGAN AI Upscaler
-----------------------
Singleton pattern: weights (~65MB) are loaded once per worker process.
Uses tiled processing (tile=512) to avoid OOM on large images.
Falls back to Lanczos interpolation if model weights are missing.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class Upscaler:
    _instance = None

    @classmethod
    def get_instance(cls) -> "Upscaler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._esrgan = None
        self._load_model()

    def _load_model(self):
        model_path = settings.REALESRGAN_MODEL_PATH
        if not Path(model_path).exists():
            logger.warning(
                "Real-ESRGAN weights not found at %s. "
                "Falling back to Lanczos interpolation.",
                model_path,
            )
            return

        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            half = device == "cuda"  # fp16 only on GPU

            self._esrgan = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=512,       # tile to handle large images without OOM
                tile_pad=10,
                pre_pad=0,
                half=half,
                device=torch.device(device),
            )
            logger.info("Real-ESRGAN loaded on %s (half=%s)", device, half)

        except Exception as exc:
            logger.error("Failed to load Real-ESRGAN: %s", exc)
            self._esrgan = None

    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Upscale image by 4× using Real-ESRGAN.
        Falls back to cv2.resize (INTER_LANCZOS4) if model not available.
        """
        if self._esrgan is not None:
            try:
                output, _ = self._esrgan.enhance(img_bgr, outscale=4)
                return output
            except Exception as exc:
                logger.warning("Real-ESRGAN inference failed: %s. Using Lanczos.", exc)

        # Fallback: Lanczos ×4
        h, w = img_bgr.shape[:2]
        return cv2.resize(img_bgr, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
