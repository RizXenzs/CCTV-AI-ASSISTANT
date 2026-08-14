"""
snapshot_scheduler.py — Captures and manages snapshots during active events.

Handles:
- Initial alert snapshot (highest confidence frame)
- Periodic snapshots every 120s during active events
- File naming convention: cameraId_eventId_YYYYmmdd_HHMMSS.jpg
- Deduplication via image hash comparison
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SnapshotScheduler:
    """Manages snapshot capture during active events.

    Selects the best frame (highest detection confidence) within each
    snapshot interval and saves it with a standardized filename.
    """

    def __init__(
        self,
        snapshot_dir: str = "snapshots",
        quality: int = 85,
    ):
        """
        Args:
            snapshot_dir: Directory to save snapshot images.
            quality: JPEG quality (0-100).
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality

        # Best frame buffer: (camera_id, event_id) -> (frame, confidence, timestamp)
        self._best_frames: Dict[str, Tuple[np.ndarray, float, float]] = {}

        # Last snapshot hash for deduplication: (camera_id, event_id) -> hash
        self._last_hashes: Dict[str, str] = {}

        logger.info("SnapshotScheduler initialized: dir=%s, quality=%d", snapshot_dir, quality)

    def update_best_frame(
        self,
        camera_id: str,
        event_id: str,
        frame: np.ndarray,
        confidence: float,
        bbox: Optional[np.ndarray] = None,
    ) -> None:
        """Track the best (highest confidence) frame for the current interval.

        Call this on every frame during an active event. The scheduler keeps
        the frame with the highest detection confidence.

        Args:
            camera_id: Camera identifier.
            event_id: Current event identifier.
            frame: BGR frame from camera.
            confidence: Detection confidence (higher = better frame).
            bbox: Bounding box [x1, y1, x2, y2] to crop the person (optional).
        """
        key = f"{camera_id}:{event_id}"
        current = self._best_frames.get(key)

        if current is None or confidence > current[1]:
            save_frame = frame.copy()
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = save_frame.shape[:2]
                margin = 30
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                save_frame = save_frame[y1:y2, x1:x2]
                
            self._best_frames[key] = (save_frame, confidence, 0.0)

    def capture_snapshot(
        self,
        camera_id: str,
        event_id: str,
        snapshot_type: str = "alert",
        frame_override: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Capture and save a snapshot for an event.

        Uses the best buffered frame, or frame_override if provided.

        Args:
            camera_id: Camera identifier.
            event_id: Event identifier.
            snapshot_type: "alert" for initial, "periodic" for interval snapshots.
            frame_override: If provided, use this frame instead of the buffer.

        Returns:
            File path of saved snapshot, or None if no frame available or duplicate.
        """
        key = f"{camera_id}:{event_id}"

        # Get the frame to save
        if frame_override is not None:
            frame = frame_override
        elif key in self._best_frames:
            frame = self._best_frames[key][0]
        else:
            logger.warning("[%s] No frame available for snapshot", camera_id)
            return None

        # Deduplication: compute hash and compare with last snapshot
        frame_hash = self._compute_hash(frame)
        if key in self._last_hashes and self._last_hashes[key] == frame_hash:
            logger.debug("[%s] Duplicate snapshot skipped", camera_id)
            return None

        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_event = event_id[:8]  # Use first 8 chars of UUID
        filename = f"{camera_id}_{short_event}_{timestamp_str}.jpg"

        # Save to disk
        filepath = self.snapshot_dir / filename
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            success = cv2.imwrite(str(filepath), frame, encode_params)

            if not success:
                logger.error("[%s] Failed to write snapshot: %s", camera_id, filepath)
                return None

            # Update dedup hash
            self._last_hashes[key] = frame_hash

            # Reset best frame buffer for next interval
            self._best_frames.pop(key, None)

            file_size_kb = filepath.stat().st_size / 1024
            logger.info(
                "[%s] Snapshot saved: %s (%.1f KB, type=%s)",
                camera_id,
                filename,
                file_size_kb,
                snapshot_type,
            )
            return str(filepath)

        except Exception as e:
            logger.error("[%s] Error saving snapshot: %s", camera_id, e)
            return None

    def get_snapshot_as_bytes(self, filepath: str) -> Optional[bytes]:
        """Read a snapshot file and return as bytes (for Telegram sending).

        Args:
            filepath: Path to the snapshot file.

        Returns:
            JPEG bytes, or None if file doesn't exist.
        """
        try:
            with open(filepath, "rb") as f:
                return f.read()
        except FileNotFoundError:
            logger.error("Snapshot file not found: %s", filepath)
            return None
        except Exception as e:
            logger.error("Error reading snapshot: %s", e)
            return None

    @staticmethod
    def _compute_hash(frame: np.ndarray) -> str:
        """Compute a perceptual hash of a frame for deduplication.

        Uses a downscaled grayscale version to detect similar (not identical) frames.
        """
        # Resize to small thumbnail for hash comparison
        small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        return hashlib.md5(gray.tobytes()).hexdigest()

    def cleanup_event(self, camera_id: str, event_id: str) -> None:
        """Clean up buffers for a resolved event."""
        key = f"{camera_id}:{event_id}"
        self._best_frames.pop(key, None)
        self._last_hashes.pop(key, None)

    def get_snapshot_count(self, camera_id: str) -> int:
        """Count snapshot files for a camera."""
        pattern = f"{camera_id}_*"
        return len(list(self.snapshot_dir.glob(pattern)))

    def cleanup_old_snapshots(self, max_age_days: int = 30) -> int:
        """Delete snapshot files older than N days."""
        import time as _time

        now = _time.time()
        cutoff = now - (max_age_days * 86400)
        deleted = 0

        for filepath in self.snapshot_dir.glob("*.jpg"):
            try:
                if filepath.stat().st_mtime < cutoff:
                    filepath.unlink()
                    deleted += 1
            except Exception as e:
                logger.debug("Error deleting old snapshot %s: %s", filepath, e)

        if deleted > 0:
            logger.info("Cleaned up %d old snapshots (>%d days)", deleted, max_age_days)
        return deleted
