"""
tracker.py — Multi-object tracking using ByteTrack via supervision.

Assigns persistent IDs to detected persons across frames and maintains
per-track position history for feature extraction.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


@dataclass
class TrackPoint:
    """A single tracked position at a point in time."""
    centroid: Tuple[float, float]  # (cx, cy)
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    timestamp: float  # time.time()
    keypoints: Optional[np.ndarray] = None  # (17, 2) or (17, 3) keypoints array


@dataclass
class TrackHistory:
    """Accumulated history for a single tracked person or animal."""
    track_id: int
    first_seen: float = 0.0
    last_seen: float = 0.0
    points: Deque[TrackPoint] = field(default_factory=lambda: deque(maxlen=300))
    is_active: bool = True
    class_type: str = "human"  # "human" or "animal"

    @property
    def age_sec(self) -> float:
        """How long this track has been alive."""
        if self.first_seen == 0:
            return 0.0
        return self.last_seen - self.first_seen

    @property
    def latest_centroid(self) -> Optional[Tuple[float, float]]:
        """Get the most recent centroid position."""
        if self.points:
            return self.points[-1].centroid
        return None

    @property
    def latest_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the most recent bounding box."""
        if self.points:
            return self.points[-1].bbox
        return None

    @property
    def latest_confidence(self) -> float:
        """Get the most recent detection confidence."""
        if self.points:
            return self.points[-1].confidence
        return 0.0

    @property
    def latest_keypoints(self) -> Optional[np.ndarray]:
        """Get the most recent keypoints."""
        if self.points:
            return self.points[-1].keypoints
        return None


class PersonTracker:
    """ByteTrack-based person tracker with position history.

    Wraps supervision.ByteTrack to provide:
    - Persistent person IDs across frames
    - Per-track position history (for speed/trajectory analysis)
    - Track lifecycle management (active/lost)
    """

    def __init__(
        self,
        camera_id: str,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 10,
        history_maxlen: int = 300,
    ):
        """
        Args:
            camera_id: Camera this tracker belongs to.
            track_activation_threshold: Min confidence to create a new track.
            lost_track_buffer: Frames to keep a lost track before removal.
            minimum_matching_threshold: IoU threshold for matching.
            frame_rate: Expected frame rate (for ByteTrack internals).
            history_maxlen: Max history points per track.
        """
        self.camera_id = camera_id
        self._history_maxlen = history_maxlen

        # Initialize ByteTrack
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

        # Track history storage: track_id -> TrackHistory
        self._tracks: Dict[int, TrackHistory] = {}

        # Track IDs seen in the current frame
        self._active_ids: set = set()

        # Stats
        self._total_tracks_created: int = 0

        logger.info("[%s] PersonTracker initialized (ByteTrack)", camera_id)

    def update(
        self, 
        detections: sv.Detections, 
        keypoints: Optional[sv.KeyPoints] = None
    ) -> sv.Detections:
        """Update tracker with new detections.

        Args:
            detections: Person detections from the current frame.
            keypoints: Associated keypoints (if available).

        Returns:
            Tracked detections with tracker_id assigned.
        """
        now = time.time()

        # Run ByteTrack update
        tracked = self._tracker.update_with_detections(detections)

        # Update internal history
        current_ids = set()

        if tracked.tracker_id is not None:
            # We need to map tracked back to original detections to get keypoints
            # ByteTrack reorders/filters detections, but tracked.class_id etc might be preserved.
            # supervision ByteTrack preserves the original index in some versions, 
            # but usually we can match by bbox IoU or just rely on the fact that 
            # tracked returns the same xyxy order or we can just use the xyxy.
            # Let's match tracked to original keypoints using bbox centers or IoU.
            # Actually, `tracked.tracker_id` is aligned with `tracked.xyxy`.
            
            for i, track_id in enumerate(tracked.tracker_id):
                track_id = int(track_id)
                current_ids.add(track_id)

                # Get detection data
                bbox = tuple(tracked.xyxy[i].tolist())
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                class_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                class_type = "human" if class_id == 0 else "animal"
                
                # Match keypoints (find closest bbox in original detections)
                kp = None
                if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > 0:
                    # Find closest bbox by center
                    orig_centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
                    orig_centers_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
                    dists = (orig_centers_x - cx)**2 + (orig_centers_y - cy)**2
                    closest_idx = np.argmin(dists)
                    if dists[closest_idx] < 100: # threshold for match
                        kp = keypoints.xy[closest_idx]

                point = TrackPoint(
                    centroid=(cx, cy),
                    bbox=bbox,
                    confidence=conf,
                    timestamp=now,
                    keypoints=kp,
                )

                if track_id not in self._tracks:
                    # New track
                    self._tracks[track_id] = TrackHistory(
                        track_id=track_id,
                        first_seen=now,
                        last_seen=now,
                        points=deque(maxlen=self._history_maxlen),
                        is_active=True,
                        class_type=class_type,
                    )
                    self._tracks[track_id].points.append(point)
                    self._total_tracks_created += 1
                    logger.debug("[%s] New track: #%d (%s)", self.camera_id, track_id, class_type)
                else:
                    # Update existing track
                    self._tracks[track_id].last_seen = now
                    self._tracks[track_id].is_active = True
                    self._tracks[track_id].points.append(point)
                    # Update class type if it somehow changed (e.g., misclassification in earlier frame)
                    if self._tracks[track_id].class_type != class_type:
                         self._tracks[track_id].class_type = class_type

        # Mark tracks not in current frame as inactive
        for tid, th in self._tracks.items():
            if tid not in current_ids:
                th.is_active = False

        self._active_ids = current_ids

        # Cleanup very old tracks (not seen for >60 seconds)
        stale_ids = [
            tid for tid, th in self._tracks.items()
            if not th.is_active and (now - th.last_seen) > 60
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        return tracked

    def get_track_history(self, track_id: int) -> Optional[TrackHistory]:
        """Get the full history for a specific track."""
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> Dict[int, TrackHistory]:
        """Get all currently active tracks."""
        return {
            tid: th for tid, th in self._tracks.items() if th.is_active
        }

    def get_all_tracks(self) -> Dict[int, TrackHistory]:
        """Get all tracks (active and inactive)."""
        return dict(self._tracks)

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._active_ids)

    @property
    def active_ids(self) -> set:
        """Set of currently active track IDs."""
        return self._active_ids.copy()

    @property
    def total_tracks(self) -> int:
        """Total tracks created since initialization."""
        return self._total_tracks_created

    def get_recent_centroids(
        self, track_id: int, window_sec: float = 5.0
    ) -> List[Tuple[float, float]]:
        """Get centroids for a track within a time window.

        Args:
            track_id: The track to query.
            window_sec: How far back in time to look (seconds).

        Returns:
            List of (cx, cy) centroids, oldest first.
        """
        th = self._tracks.get(track_id)
        if th is None:
            return []

        now = time.time()
        cutoff = now - window_sec

        return [
            pt.centroid for pt in th.points if pt.timestamp >= cutoff
        ]

    def get_track_bboxes(
        self, track_id: int, window_sec: float = 5.0
    ) -> List[Tuple[float, float, float, float]]:
        """Get bounding boxes for a track within a time window."""
        th = self._tracks.get(track_id)
        if th is None:
            return []

        now = time.time()
        cutoff = now - window_sec

        return [
            pt.bbox for pt in th.points if pt.timestamp >= cutoff
        ]

    def reset(self) -> None:
        """Reset the tracker (e.g. on scene change)."""
        self._tracker.reset()
        self._tracks.clear()
        self._active_ids.clear()
        logger.info("[%s] Tracker reset", self.camera_id)
