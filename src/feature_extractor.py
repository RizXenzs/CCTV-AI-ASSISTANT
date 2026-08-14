"""
feature_extractor.py — Extract behavioral features from tracked persons.

Computes per-track features like speed, dwell time, trajectory patterns,
zone presence, and more. These features are consumed by the rule engine.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import supervision as sv
except ImportError:
    sv = None

from src.tracker import PersonTracker, TrackHistory
from src.config_loader import CameraConfig, ROIZone
from src.zone_manager import ZoneManager

logger = logging.getLogger(__name__)


@dataclass
class TrackFeatures:
    """Computed features for a single tracked person."""
    track_id: int

    # Spatial features
    speed_px_per_sec: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # Temporal features
    dwell_time_sec: float = 0.0
    track_age_sec: float = 0.0
    is_new_track: bool = False  # appeared within last N seconds

    # Zone features
    current_zones: Set[str] = field(default_factory=set)
    zone_entries_count: int = 0
    line_crossings: List[Tuple[str, str]] = field(default_factory=list)

    # Trajectory features
    path_entropy: float = 0.0  # direction change frequency
    direction_changes: int = 0
    is_back_and_forth: bool = False

    # Posture features
    bbox_aspect_ratio: float = 1.0  # height / width
    bbox_height_ratio_change: float = 1.0  # current / initial height ratio
    bbox_aspect_stable_sec: float = 0.0  # how long aspect ratio has been low

    # Context features
    confidence: float = 0.0
    person_count: int = 0  # total persons in frame

    # Approach features (movement toward a specific zone)
    approaching_zone: Optional[str] = None
    approach_speed: float = 0.0


@dataclass
class SceneFeatures:
    """Scene-level features (not per-track)."""
    person_count: int = 0
    motion_level: float = 0.0
    black_frame_ratio: float = 0.0
    is_night_hours: bool = False
    timestamp: float = 0.0


class FeatureExtractor:
    """Extracts behavioral features from tracked persons and scene context.

    Works in conjunction with the PersonTracker to compute features
    like speed, dwell time, trajectory patterns, and zone presence.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        app_config: "AppConfig", # Added app_config for global night mode
        frame_shape: Tuple[int, int] = (480, 640),  # (height, width)
        new_track_threshold_sec: float = 3.0,
    ):
        """
        Args:
            camera_config: Camera configuration with ROI zones.
            app_config: Application configuration for global settings.
            frame_shape: Frame dimensions (height, width) for zone setup.
            new_track_threshold_sec: Track younger than this is "new".
        """
        self.camera_config = camera_config
        self.app_config = app_config
        self._new_track_threshold = new_track_threshold_sec
        self._frame_shape = frame_shape

        # Zone tracking: track_id -> {zone_name -> entry_count}
        self._zone_entry_counts: Dict[int, Dict[str, int]] = {}
        # Previous zone state per track
        self._prev_zones: Dict[int, Set[str]] = {}

        # Aspect ratio tracking for crouch detection
        self._initial_bbox_height: Dict[int, float] = {}
        self._low_aspect_start: Dict[int, float] = {}



        # Zone & Tripwire Manager
        self.zone_manager = ZoneManager(camera_config, app_config)

    def extract(
        self,
        tracker: PersonTracker,
        detections,  # sv.Detections
        motion_level: float = 0.0,
        black_frame_ratio: float = 0.0,
    ) -> Tuple[Dict[int, TrackFeatures], SceneFeatures]:
        """Extract features for all active tracks and the scene.

        Args:
            tracker: PersonTracker instance with updated tracks.
            detections: Current frame's tracked detections (with tracker_id).
            motion_level: From motion detector (0.0-1.0).
            black_frame_ratio: From preprocessor (0.0-1.0).

        Returns:
            Tuple of (per-track features dict, scene features).
        """
        now = time.time()
        # Only process human tracks for features/rules
        active_tracks = {
            tid: th for tid, th in tracker.get_active_tracks().items()
            if th.class_type == "human"
        }
        person_count = len(active_tracks)

        # Scene-level features
        scene = SceneFeatures(
            person_count=person_count,
            motion_level=motion_level,
            black_frame_ratio=black_frame_ratio,
            is_night_hours=self._check_night_hours(),
            timestamp=now,
        )

        # Zone containment check (batch)
        track_zones = self.zone_manager.compute_zones(detections)

        # Per-track features
        per_track: Dict[int, TrackFeatures] = {}

        for track_id, history in active_tracks.items():
            features = TrackFeatures(track_id=track_id)

            # Basic info
            features.person_count = person_count
            features.track_age_sec = history.age_sec
            features.is_new_track = history.age_sec < self._new_track_threshold
            features.confidence = history.latest_confidence

            if history.latest_centroid:
                features.centroid = history.latest_centroid
            if history.latest_bbox:
                features.bbox = history.latest_bbox

            # Speed
            features.speed_px_per_sec = self._compute_speed(history)

            # Dwell time (total time in scene)
            features.dwell_time_sec = history.age_sec

            # Current zones
            features.current_zones = track_zones.get(track_id, set())

            # Zone entry counting
            features.zone_entries_count = self._update_zone_entries(
                track_id, features.current_zones
            )

            # Line crossing checks
            if len(history.points) >= 2:
                prev_centroid = history.points[-2].centroid
                curr_centroid = history.points[-1].centroid
                features.line_crossings = self.zone_manager.check_crossings(prev_centroid, curr_centroid)

            # Trajectory analysis
            direction_changes, entropy = self._compute_trajectory_features(
                tracker, track_id
            )
            features.direction_changes = direction_changes
            features.path_entropy = entropy
            features.is_back_and_forth = direction_changes >= 3  # simplified

            # Bbox aspect ratio (crouch detection)
            if history.latest_bbox:
                features.bbox_aspect_ratio = self._compute_aspect_ratio(history.latest_bbox)
                features.bbox_height_ratio_change = self._compute_height_ratio_change(
                    track_id, history.latest_bbox
                )
                features.bbox_aspect_stable_sec = self._compute_low_aspect_duration(
                    track_id, features.bbox_height_ratio_change, now
                )

            # Approach detection
            approach_zone, approach_speed = self._detect_approach(
                tracker, track_id, features.speed_px_per_sec
            )
            features.approaching_zone = approach_zone
            features.approach_speed = approach_speed

            per_track[track_id] = features

        return per_track, scene

    # ----- Feature computation helpers -----

    def _compute_speed(self, history: TrackHistory) -> float:
        """Compute speed in pixels/second from recent centroids."""
        if len(history.points) < 2:
            return 0.0

        # Use last 2 points for instantaneous speed
        p1 = history.points[-2]
        p2 = history.points[-1]

        dt = p2.timestamp - p1.timestamp
        if dt <= 0:
            return 0.0

        dx = p2.centroid[0] - p1.centroid[0]
        dy = p2.centroid[1] - p1.centroid[1]
        distance = math.sqrt(dx * dx + dy * dy)

        return distance / dt
    def _update_zone_entries(self, track_id: int, current_zones: Set[str]) -> int:
        """Track zone entry counts for a person."""
        if track_id not in self._zone_entry_counts:
            self._zone_entry_counts[track_id] = {}
            self._prev_zones[track_id] = set()

        prev = self._prev_zones[track_id]
        entries = self._zone_entry_counts[track_id]

        # Count new zone entries (zones in current but not in previous)
        new_entries = current_zones - prev
        for zone in new_entries:
            entries[zone] = entries.get(zone, 0) + 1

        self._prev_zones[track_id] = current_zones.copy()

        return sum(entries.values())

    def _compute_trajectory_features(
        self, tracker: PersonTracker, track_id: int, window_sec: float = 20.0
    ) -> Tuple[int, float]:
        """Compute direction changes and path entropy.

        Returns:
            (direction_changes, path_entropy)
        """
        centroids = tracker.get_recent_centroids(track_id, window_sec)
        if len(centroids) < 3:
            return 0, 0.0

        # Compute direction vectors
        directions = []
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            if abs(dx) > 1 or abs(dy) > 1:  # filter noise
                angle = math.atan2(dy, dx)
                directions.append(angle)

        if len(directions) < 2:
            return 0, 0.0

        # Count significant direction changes (> 90 degrees)
        changes = 0
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i - 1])
            # Normalize to [0, pi]
            diff = min(diff, 2 * math.pi - diff)
            if diff > math.pi / 2:  # > 90 degrees
                changes += 1

        # Path entropy: normalized direction change frequency
        entropy = changes / len(directions) if directions else 0.0

        return changes, entropy

    @staticmethod
    def _compute_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
        """Compute height/width ratio of a bounding box."""
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        return height / width

    def _compute_height_ratio_change(
        self, track_id: int, bbox: Tuple[float, float, float, float]
    ) -> float:
        """Compute current height relative to initial height (crouch detection)."""
        _, y1, _, y2 = bbox
        current_height = max(y2 - y1, 1)

        if track_id not in self._initial_bbox_height:
            self._initial_bbox_height[track_id] = current_height
            return 1.0

        initial = self._initial_bbox_height[track_id]
        return current_height / initial if initial > 0 else 1.0

    def _compute_low_aspect_duration(
        self, track_id: int, height_ratio: float, now: float, threshold: float = 0.6
    ) -> float:
        """Track how long the height ratio has been below threshold (crouching)."""
        if height_ratio < threshold:
            if track_id not in self._low_aspect_start:
                self._low_aspect_start[track_id] = now
            return now - self._low_aspect_start[track_id]
        else:
            # Reset
            self._low_aspect_start.pop(track_id, None)
            return 0.0

    def _detect_approach(
        self,
        tracker: PersonTracker,
        track_id: int,
        current_speed: float,
    ) -> Tuple[Optional[str], float]:
        """Detect if a person is moving toward a specific zone.

        Returns:
            (zone_name, approach_speed) or (None, 0.0)
        """
        centroids = tracker.get_recent_centroids(track_id, window_sec=3.0)
        if len(centroids) < 2:
            return None, 0.0

        # Movement vector
        dx = centroids[-1][0] - centroids[0][0]
        dy = centroids[-1][1] - centroids[0][1]

        for zone_name, roi_zone in self.camera_config.roi_zones.items():
            if not roi_zone.points:
                continue

            # Zone center
            points = np.array(roi_zone.points)
            zone_cx = float(np.mean(points[:, 0]))
            zone_cy = float(np.mean(points[:, 1]))

            # Vector from current position to zone center
            to_zone_x = zone_cx - centroids[-1][0]
            to_zone_y = zone_cy - centroids[-1][1]

            # Dot product: positive = moving toward zone
            dot = dx * to_zone_x + dy * to_zone_y
            if dot > 0 and current_speed > 50:  # Moving toward zone with some speed
                return zone_name, current_speed

        return None, 0.0

    def _check_night_hours(self) -> bool:
        """Check if current time falls within configured night mode hours."""
        start_str = self.app_config.night_mode_start
        end_str = self.app_config.night_mode_end

        try:
            start_h, start_m = map(int, start_str.strip().split(":"))
            end_h, end_m = map(int, end_str.strip().split(":"))

            now = datetime.now()
            current_minutes = now.hour * 60 + now.minute
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m

            if start_minutes <= end_minutes:
                # Same day range (e.g., 08:00-17:00)
                return start_minutes <= current_minutes < end_minutes
            else:
                # Overnight range (e.g., 22:00-05:00)
                return current_minutes >= start_minutes or current_minutes < end_minutes

        except (ValueError, IndexError):
            logger.debug("Invalid night_mode format. Using start: %s, end: %s", start_str, end_str)
            return False

    def cleanup_track(self, track_id: int) -> None:
        """Clean up resources for a removed track."""
        self._zone_entry_counts.pop(track_id, None)
        self._prev_zones.pop(track_id, None)
        self._initial_bbox_height.pop(track_id, None)
        self._low_aspect_start.pop(track_id, None)
