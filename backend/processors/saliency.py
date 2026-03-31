"""
Saliency Map Generator (U²-Net)
--------------------------------
Produces a per-pixel importance map (0-255, uint8) at the original image size.
High values = visually important / salient regions.

Falls back to center-weighted Gaussian if model weights are missing.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# U2Net input size (fixed by architecture)
U2NET_INPUT_SIZE = 320


class SaliencyDetector:
    _instance = None

    @classmethod
    def get_instance(cls) -> "SaliencyDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._model = None
        self._device = None
        self._load_model()

    def _load_model(self):
        model_path = settings.U2NET_MODEL_PATH
        if not Path(model_path).exists():
            logger.warning(
                "U2Net weights not found at %s. Falling back to center-weighted saliency.",
                model_path,
            )
            return

        try:
            import torch
            from .u2net_arch import U2NET

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = U2NET(in_ch=3, out_ch=1)
            net.load_state_dict(torch.load(model_path, map_location=self._device))
            net.eval()
            net.to(self._device)
            self._model = net
            logger.info("U2Net loaded on %s", self._device)

        except Exception as exc:
            logger.error("Failed to load U2Net: %s", exc)
            self._model = None

    def generate(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a saliency map as uint8 array (H x W) same size as input.
        Values: 0 = background, 255 = most salient.
        """
        orig_h, orig_w = img_bgr.shape[:2]

        if self._model is not None:
            return self._run_u2net(img_bgr, orig_w, orig_h)

        return self._center_weighted_fallback(orig_w, orig_h)

    def _run_u2net(self, img_bgr: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        import torch

        # Preprocess: resize → normalize → CHW tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (U2NET_INPUT_SIZE, U2NET_INPUT_SIZE))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(self._device)

        with torch.no_grad():
            outputs = self._model(tensor)
            # d1 is the primary (highest quality) output
            pred = outputs[1].squeeze().cpu().numpy()

        # Normalize to 0-255
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_uint8 = (pred * 255).astype(np.uint8)

        # Resize back to original dimensions
        saliency = cv2.resize(pred_uint8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return saliency

    @staticmethod
    def _center_weighted_fallback(w: int, h: int) -> np.ndarray:
        """
        Simple center-weighted Gaussian as a fallback when U2Net is unavailable.
        Assumes subjects are roughly centered (true most of the time).
        """
        cx, cy = w // 2, h // 2
        sigma_x, sigma_y = w * 0.4, h * 0.4

        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        gauss = np.exp(
            -((xx - cx) ** 2 / (2 * sigma_x ** 2) + (yy - cy) ** 2 / (2 * sigma_y ** 2))
        )
        gauss = (gauss / gauss.max() * 255).astype(np.uint8)
        return gauss
