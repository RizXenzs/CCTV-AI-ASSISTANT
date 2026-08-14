"""
motion_detector.py — Motion detection using MOG2 background subtraction.

Acts as a gate for the YOLO detector: if no significant motion is detected,
expensive person detection is skipped to save CPU/GPU resources.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Motion sensitivity presets: lower threshold = more sensitive
_SENSITIVITY_MAP = {
    "low": 0.05,
    "med": 0.02,
    "high": 0.008,
}


class MotionDetector:
    """MOG2-based motion detection with configurable sensitivity.

    Returns a motion level (0.0-1.0) representing the fraction of the frame
    with detected motion. This is used as a gate to skip YOLO inference
    when the scene is static.
    """

    def __init__(
        self,
        sensitivity: str = "med",
        history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = False,
        kernel_size: int = 5,
        min_contour_area: int = 500,
    ):
        """
        Args:
            sensitivity: "low", "med", "high", or a numeric string (0.0-1.0).
            history: Number of frames for MOG2 background model.
            var_threshold: MOG2 variance threshold.
            detect_shadows: Whether MOG2 should detect shadows.
            kernel_size: Morphological kernel size for noise removal.
            min_contour_area: Minimum contour area to consider as motion.
        """
        # Parse sensitivity to threshold
        self.motion_threshold = self._parse_sensitivity(sensitivity)

        # Create MOG2 background subtractor
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        # Morphological kernel for noise cleanup
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        self._min_contour_area = min_contour_area

        # State
        self._warmup_frames = 30  # Frames to skip during initial learning
        self._frame_count = 0
        self._last_motion_level: float = 0.0
        self._last_mask: np.ndarray = np.array([])

        logger.info(
            "MotionDetector initialized: threshold=%.4f, history=%d",
            self.motion_threshold,
            history,
        )

    @staticmethod
    def _parse_sensitivity(sensitivity: str) -> float:
        """Convert sensitivity string/number to a threshold value."""
        try:
            return float(sensitivity)
        except ValueError:
            return _SENSITIVITY_MAP.get(sensitivity.lower(), 0.02)

    def detect(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run motion detection on a frame.

        Args:
            frame: Input BGR frame (already resized by preprocessor).

        Returns:
            Tuple of (motion_level, motion_mask).
            - motion_level: 0.0-1.0 fraction of frame with motion.
            - motion_mask: Binary mask of motion regions (255 = motion).
        """
        self._frame_count += 1

        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply MOG2 background subtraction
        fg_mask = self._bg_subtractor.apply(gray)

        # Threshold to binary (remove shadows marked as 127)
        _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self._kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, self._kernel)
        binary_mask = cv2.dilate(binary_mask, self._kernel, iterations=2)

        # During warmup, don't trigger motion (model is still learning)
        if self._frame_count < self._warmup_frames:
            self._last_motion_level = 0.0
            self._last_mask = binary_mask
            return 0.0, binary_mask

        # Calculate motion level as fraction of white pixels
        white_pixels = np.count_nonzero(binary_mask)
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        motion_level = white_pixels / total_pixels if total_pixels > 0 else 0.0

        self._last_motion_level = motion_level
        self._last_mask = binary_mask

        return motion_level, binary_mask

    def has_significant_motion(self, motion_level: float = -1.0) -> bool:
        """Check if motion level exceeds the configured threshold.

        Args:
            motion_level: Explicit motion level to check.
                         If -1.0, uses the last computed level.

        Returns:
            True if motion is above threshold.
        """
        level = motion_level if motion_level >= 0 else self._last_motion_level
        return level >= self.motion_threshold

    def get_motion_contours(
        self, mask: np.ndarray = None
    ) -> list:
        """Get motion contours from the binary mask.

        Returns list of contours that are larger than min_contour_area.
        Useful for localizing where motion occurred.
        """
        if mask is None:
            mask = self._last_mask

        if mask.size == 0:
            return []

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by minimum area
        significant = [
            c for c in contours if cv2.contourArea(c) >= self._min_contour_area
        ]

        return significant

    @property
    def last_motion_level(self) -> float:
        """Get the last computed motion level."""
        return self._last_motion_level

    @property
    def last_mask(self) -> np.ndarray:
        """Get the last computed motion mask."""
        return self._last_mask

    @property
    def is_warmed_up(self) -> bool:
        """Check if the background model has completed warmup."""
        return self._frame_count >= self._warmup_frames

    def reset(self) -> None:
        """Reset the background model (e.g. after scene change)."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._bg_subtractor.getHistory(),
            varThreshold=self._bg_subtractor.getVarThreshold(),
            detectShadows=self._bg_subtractor.getDetectShadows(),
        )
        self._frame_count = 0
        self._last_motion_level = 0.0
        logger.info("MotionDetector background model reset")
