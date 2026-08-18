"""
alarm_gate.py — Anti-False Alarm multi-stage validation pipeline.

Filters out false detections before they reach the rule engine.
Each track must pass ALL stages to be considered valid:

1. Confidence Gate  — detection confidence ≥ threshold
2. Frame Persistence — track must exist for ≥ N consecutive frames
3. Bbox Size Filter  — bounding box area within reasonable bounds
4. Aspect Ratio Filter — bbox aspect ratio consistent with a real person
5. Tracking Consistency — same track ID maintained (handled by ByteTrack)

This significantly reduces false alarms caused by:
- Shadows
- Animal misclassification
- Vehicle headlights
- Camera noise / artifacts
- Brief passersby
- Lighting changes
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from src.tracker import PersonTracker, TrackHistory

logger = logging.getLogger(__name__)


@dataclass
class RejectionStats:
    """Tracks why detections were rejected, for debugging and dashboard display."""
    total_evaluated: int = 0
    total_passed: int = 0
    rejected_confidence: int = 0
    rejected_persistence: int = 0
    rejected_bbox_too_small: int = 0
    rejected_bbox_too_large: int = 0
    rejected_aspect_ratio: int = 0

    # Rolling window stats (reset every N seconds)
    _window_start: float = field(default_factory=time.time)
    _window_evaluated: int = 0
    _window_passed: int = 0
    _window_rejected_confidence: int = 0
    _window_rejected_persistence: int = 0
    _window_rejected_bbox: int = 0
    _window_rejected_aspect: int = 0

    def reset_window(self) -> None:
        """Reset the rolling window counters."""
        self._window_start = time.time()
        self._window_evaluated = 0
        self._window_passed = 0
        self._window_rejected_confidence = 0
        self._window_rejected_persistence = 0
        self._window_rejected_bbox = 0
        self._window_rejected_aspect = 0

    @property
    def pass_rate(self) -> float:
        """Percentage of tracks that passed validation."""
        if self.total_evaluated == 0:
            return 100.0
        return (self.total_passed / self.total_evaluated) * 100.0

    @property
    def window_pass_rate(self) -> float:
        """Pass rate for the current rolling window."""
        if self._window_evaluated == 0:
            return 100.0
        return (self._window_passed / self._window_evaluated) * 100.0

    def to_dict(self) -> dict:
        """Serialize stats for API response."""
        return {
            "total_evaluated": self.total_evaluated,
            "total_passed": self.total_passed,
            "pass_rate": round(self.pass_rate, 1),
            "rejected": {
                "confidence": self.rejected_confidence,
                "persistence": self.rejected_persistence,
                "bbox_too_small": self.rejected_bbox_too_small,
                "bbox_too_large": self.rejected_bbox_too_large,
                "aspect_ratio": self.rejected_aspect_ratio,
            },
            "window": {
                "evaluated": self._window_evaluated,
                "passed": self._window_passed,
                "pass_rate": round(self.window_pass_rate, 1),
                "duration_sec": round(time.time() - self._window_start, 0),
            },
        }


@dataclass
class TrackValidation:
    """Validation result for a single track."""
    track_id: int
    passed: bool = False
    rejection_reason: Optional[str] = None

    # Stage results
    confidence_ok: bool = False
    persistence_ok: bool = False
    bbox_size_ok: bool = False
    aspect_ratio_ok: bool = False


class AlarmGate:
    """Multi-stage validation pipeline to filter false alarms.

    Each tracked person must pass all validation stages before
    being forwarded to the rule engine for scoring.

    Stages:
    1. Confidence ≥ confidence_gate (default 0.60)
    2. Track seen in ≥ min_frames consecutive frames
    3. Bounding box area within [min_bbox_area, max_bbox_ratio * frame_area]
    4. Bbox aspect ratio within [min_aspect, max_aspect] for a real person
    """

    # Reasonable human aspect ratio bounds (height / width)
    MIN_HUMAN_ASPECT = 0.4   # Very wide / crouching person
    MAX_HUMAN_ASPECT = 5.0   # Very tall / narrow (edge case)

    def __init__(
        self,
        confidence_gate: float = 0.60,
        min_frames: int = 3,
        min_bbox_area: int = 400,
        max_bbox_ratio: float = 0.5,
        max_aspect_ratio: float = 5.0,
        stats_window_sec: float = 300.0,
    ):
        """
        Args:
            confidence_gate: Minimum detection confidence to pass (0.0-1.0).
            min_frames: Minimum number of frames a track must persist.
            min_bbox_area: Minimum bounding box area in pixels².
            max_bbox_ratio: Maximum bbox area as a fraction of total frame area.
            max_aspect_ratio: Maximum height/width ratio for a valid person bbox.
            stats_window_sec: Rolling window duration for stats reset.
        """
        self.confidence_gate = confidence_gate
        self.min_frames = min_frames
        self.min_bbox_area = min_bbox_area
        self.max_bbox_ratio = max_bbox_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self._stats_window_sec = stats_window_sec

        # Stats tracking
        self.stats = RejectionStats()

        # Per-track: tracks that have previously passed (cache for perf)
        self._validated_tracks: Set[int] = set()

        logger.info(
            "AlarmGate initialized: conf_gate=%.2f, min_frames=%d, "
            "min_bbox=%d, max_bbox_ratio=%.2f, max_aspect=%.1f",
            confidence_gate, min_frames, min_bbox_area,
            max_bbox_ratio, max_aspect_ratio,
        )

    def validate_track(
        self,
        track: TrackHistory,
        frame_shape: Tuple[int, int],
    ) -> TrackValidation:
        """Validate a single track through all pipeline stages.

        Args:
            track: TrackHistory from the PersonTracker.
            frame_shape: (height, width) of the processed frame.

        Returns:
            TrackValidation with pass/fail status and rejection reason.
        """
        result = TrackValidation(track_id=track.track_id)
        frame_h, frame_w = frame_shape

        # Stage 1: Confidence Gate
        confidence = track.latest_confidence
        result.confidence_ok = confidence >= self.confidence_gate
        if not result.confidence_ok:
            result.rejection_reason = (
                f"confidence_low ({confidence:.2f} < {self.confidence_gate:.2f})"
            )
            self.stats.rejected_confidence += 1
            self.stats._window_rejected_confidence += 1
            return result

        # Stage 2: Frame Persistence
        frame_count = len(track.points)
        result.persistence_ok = frame_count >= self.min_frames
        if not result.persistence_ok:
            result.rejection_reason = (
                f"insufficient_frames ({frame_count} < {self.min_frames})"
            )
            self.stats.rejected_persistence += 1
            self.stats._window_rejected_persistence += 1
            return result

        # Stage 3: Bounding Box Size Filter
        bbox = track.latest_bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_w = max(x2 - x1, 1)
            bbox_h = max(y2 - y1, 1)
            bbox_area = bbox_w * bbox_h
            frame_area = frame_h * frame_w

            if bbox_area < self.min_bbox_area:
                result.rejection_reason = (
                    f"bbox_too_small ({bbox_area:.0f} < {self.min_bbox_area})"
                )
                result.bbox_size_ok = False
                self.stats.rejected_bbox_too_small += 1
                self.stats._window_rejected_bbox += 1
                return result

            if bbox_area > frame_area * self.max_bbox_ratio:
                result.rejection_reason = (
                    f"bbox_too_large ({bbox_area:.0f} > {frame_area * self.max_bbox_ratio:.0f})"
                )
                result.bbox_size_ok = False
                self.stats.rejected_bbox_too_large += 1
                self.stats._window_rejected_bbox += 1
                return result

            result.bbox_size_ok = True

            # Stage 4: Aspect Ratio Filter
            aspect_ratio = bbox_h / bbox_w
            if aspect_ratio < self.MIN_HUMAN_ASPECT or aspect_ratio > self.max_aspect_ratio:
                result.rejection_reason = (
                    f"aspect_ratio_invalid ({aspect_ratio:.2f} not in "
                    f"[{self.MIN_HUMAN_ASPECT:.1f}, {self.max_aspect_ratio:.1f}])"
                )
                result.aspect_ratio_ok = False
                self.stats.rejected_aspect_ratio += 1
                self.stats._window_rejected_aspect += 1
                return result

            result.aspect_ratio_ok = True
        else:
            # No bbox available — shouldn't happen for active tracks, but handle gracefully
            result.bbox_size_ok = True
            result.aspect_ratio_ok = True

        # All stages passed
        result.passed = True
        return result

    def get_validated_tracks(
        self,
        tracker: PersonTracker,
        frame_shape: Tuple[int, int],
    ) -> Dict[int, TrackHistory]:
        """Filter active tracks through the validation pipeline.

        Only returns tracks that pass ALL validation stages.
        Only human tracks are evaluated (animals are excluded upstream).

        Args:
            tracker: PersonTracker instance.
            frame_shape: (height, width) of the processed frame.

        Returns:
            Dict of track_id -> TrackHistory for validated tracks only.
        """
        now = time.time()

        # Reset rolling window stats if expired
        if now - self.stats._window_start >= self._stats_window_sec:
            self.stats.reset_window()

        active_tracks = tracker.get_active_tracks()
        validated: Dict[int, TrackHistory] = {}

        for track_id, track in active_tracks.items():
            # Only validate human tracks (animals don't trigger rules)
            if track.class_type != "human":
                continue

            self.stats.total_evaluated += 1
            self.stats._window_evaluated += 1

            # Quick path: if track was previously validated and is still
            # active with sufficient confidence, skip re-validation of
            # persistence and bbox (they won't regress).
            if track_id in self._validated_tracks:
                # Still check confidence (can fluctuate frame to frame)
                if track.latest_confidence >= self.confidence_gate:
                    validated[track_id] = track
                    self.stats.total_passed += 1
                    self.stats._window_passed += 1
                    continue
                else:
                    # Confidence dropped — remove from cache and re-validate
                    self._validated_tracks.discard(track_id)

            # Full validation
            result = self.validate_track(track, frame_shape)

            if result.passed:
                validated[track_id] = track
                self._validated_tracks.add(track_id)
                self.stats.total_passed += 1
                self.stats._window_passed += 1
                logger.debug(
                    "Track #%d passed alarm gate (conf=%.2f, frames=%d)",
                    track_id, track.latest_confidence, len(track.points),
                )
            else:
                logger.debug(
                    "Track #%d rejected: %s",
                    track_id, result.rejection_reason,
                )

        # Cleanup: remove stale track IDs from validated cache
        active_ids = set(active_tracks.keys())
        stale = self._validated_tracks - active_ids
        self._validated_tracks -= stale

        return validated

    def get_rejection_stats(self) -> dict:
        """Get comprehensive rejection statistics for dashboard display."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = RejectionStats()
        logger.info("AlarmGate stats reset")
