"""
preprocessor.py — Frame preprocessing: resize, ROI masking, and normalization.

Handles frame preparation before feeding into the detection pipeline.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses camera frames for the detection pipeline."""

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (640, 480),
    ):
        """
        Args:
            target_resolution: (width, height) to resize frames to.
        """
        self.target_width, self.target_height = target_resolution
        self._roi_mask: Optional[np.ndarray] = None
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

    def set_roi_mask(
        self,
        polygons: List[List[List[int]]],
        frame_shape: Tuple[int, int],
    ) -> None:
        """Create a combined ROI mask from multiple polygon zones.

        This mask can be used to blank out areas outside the ROI,
        reducing false positives from irrelevant regions.

        Args:
            polygons: List of polygons, each polygon is [[x,y], [x,y], ...]
            frame_shape: (height, width) of the processed frame.
        """
        if not polygons:
            self._roi_mask = None
            return

        mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        self._roi_mask = mask
        logger.debug("ROI mask created with %d polygons", len(polygons))

    def process(
        self,
        frame: np.ndarray,
        apply_roi: bool = False,
    ) -> np.ndarray:
        """Resize and optionally apply ROI mask to a frame.

        Args:
            frame: Input BGR frame from camera.
            apply_roi: If True, black out regions outside ROI polygons.

        Returns:
            Processed frame (resized, optionally masked).
        """
        original_h, original_w = frame.shape[:2]

        # Calculate scale factors for coordinate mapping
        self._scale_x = original_w / self.target_width
        self._scale_y = original_h / self.target_height

        # Resize
        if (original_w, original_h) != (self.target_width, self.target_height):
            frame = cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Apply ROI mask if requested
        if apply_roi and self._roi_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=self._roi_mask)

        return frame

    def map_to_original(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """Map coordinates from processed frame back to original resolution.

        Args:
            x, y: Coordinates in processed frame space.

        Returns:
            (x, y) in original frame space.
        """
        return x * self._scale_x, y * self._scale_y

    @property
    def scale_factors(self) -> Tuple[float, float]:
        """Get current scale factors (scale_x, scale_y)."""
        return self._scale_x, self._scale_y


def compute_brightness(frame: np.ndarray) -> float:
    """Compute average brightness of a frame (0.0-255.0).

    Useful for detecting camera tamper (covered/dark).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(np.mean(gray))


def compute_black_ratio(frame: np.ndarray, threshold: int = 15) -> float:
    """Compute ratio of near-black pixels in a frame (0.0-1.0).

    Args:
        frame: Input frame.
        threshold: Pixel value below which is considered "black".

    Returns:
        Ratio of black pixels (0.0 = no black, 1.0 = all black).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    black_pixels = np.sum(gray < threshold)
    total_pixels = gray.shape[0] * gray.shape[1]
    return float(black_pixels / total_pixels) if total_pixels > 0 else 0.0
