"""
Exposure and Gain Compensation module for panorama stitching.

Adjusts the relative brightness and color balance between overlapping images
to eliminate exposure seams caused by auto-exposure / white-balance shifts.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class GainCompensator:
    """
    Computes per-image channel-wise gain coefficients to equalize exposure
    across all overlapping image pairs relative to a central reference frame.
    """

    def __init__(self, min_overlap_pixels: int = 200, min_gain: float = 0.6, max_gain: float = 1.6):
        self.min_overlap_pixels = min_overlap_pixels
        self.min_gain = min_gain
        self.max_gain = max_gain

    def compute_gains(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        ref_idx: int = 0,
    ) -> List[np.ndarray]:
        """
        Compute per-channel gain vectors (B, G, R) for each image relative to ref_idx.

        Args:
            images: List of warped BGR images on the global canvas.
            masks: List of corresponding uint8 binary masks.
            ref_idx: Index of the reference image (gain locked at [1.0, 1.0, 1.0]).

        Returns:
            List of 3-element float32 arrays representing (gain_B, gain_G, gain_R).
        """
        n = len(images)
        gains = [np.ones(3, dtype=np.float32) for _ in range(n)]
        if n <= 1:
            return gains

        # Propagate gains outward to the right of reference
        for i in range(ref_idx + 1, n):
            prev = i - 1
            overlap = (masks[prev] > 0) & (masks[i] > 0)
            if overlap.sum() >= self.min_overlap_pixels:
                mean_prev = np.array([images[prev][overlap, c].mean() for c in range(3)], dtype=np.float32)
                mean_curr = np.array([images[i][overlap, c].mean() for c in range(3)], dtype=np.float32)
                
                # Avoid division by zero
                ratio = np.where(mean_curr > 1.0, mean_prev / (mean_curr + 1e-4), 1.0)
                gain = gains[prev] * ratio
                gains[i] = np.clip(gain, self.min_gain, self.max_gain)
            else:
                gains[i] = gains[prev].copy()

        # Propagate gains outward to the left of reference
        for i in range(ref_idx - 1, -1, -1):
            succ = i + 1
            overlap = (masks[succ] > 0) & (masks[i] > 0)
            if overlap.sum() >= self.min_overlap_pixels:
                mean_succ = np.array([images[succ][overlap, c].mean() for c in range(3)], dtype=np.float32)
                mean_curr = np.array([images[i][overlap, c].mean() for c in range(3)], dtype=np.float32)

                ratio = np.where(mean_curr > 1.0, mean_succ / (mean_curr + 1e-4), 1.0)
                gain = gains[succ] * ratio
                gains[i] = np.clip(gain, self.min_gain, self.max_gain)
            else:
                gains[i] = gains[succ].copy()

        return gains

    def apply_gains(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        gains: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Apply computed gains to each image in place / copy.
        """
        compensated: List[np.ndarray] = []
        for img, mask, g in zip(images, masks, gains):
            if np.allclose(g, 1.0, atol=1e-3):
                compensated.append(img.copy())
                continue
            
            # Multiply each channel and clip to valid uint8 range
            scaled = img.astype(np.float32) * g.reshape(1, 1, 3)
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)
            # Retain pure black background outside mask
            valid = mask > 0
            out = np.zeros_like(img)
            out[valid] = scaled[valid]
            compensated.append(out)

        return compensated

