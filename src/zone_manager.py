"""
zone_manager.py — Manage ROI zones and virtual tripwires (Line Crossing).

Handles parsing coordinates, checking if points are inside zones,
and detecting line crossing events with directional support.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
try:
    import supervision as sv
except ImportError:
    sv = None

from src.config_loader import CameraConfig

logger = logging.getLogger(__name__)


@dataclass
class LineCrossing:
    """A virtual tripwire with direction detection."""
    name: str
    point_a: Tuple[int, int]
    point_b: Tuple[int, int]
    direction_mode: str = "any"  # "in", "out", "any"
    
    # Internal vectors for math
    _line_vec: np.ndarray = None
    _line_mag_sq: float = 0.0

    def __post_init__(self):
        self._line_vec = np.array([self.point_b[0] - self.point_a[0], self.point_b[1] - self.point_a[1]])
        self._line_mag_sq = float(np.sum(self._line_vec ** 2))
        
    def check_crossing(self, prev_point: Tuple[float, float], curr_point: Tuple[float, float]) -> Optional[str]:
        """Check if a line segment from prev_point to curr_point crosses this tripwire.
        
        Returns:
            "in" or "out" depending on direction, or None if no crossing.
        """
        # A simple 2D cross product approach to check intersection
        p = np.array(self.point_a)
        r = self._line_vec
        
        q = np.array(prev_point)
        s = np.array([curr_point[0] - prev_point[0], curr_point[1] - prev_point[1]])
        
        r_cross_s = np.cross(r, s)
        
        # If r x s is 0, they are parallel (no intersection)
        if abs(r_cross_s) < 1e-8:
            return None
            
        q_minus_p = q - p
        t = np.cross(q_minus_p, s) / r_cross_s
        u = np.cross(q_minus_p, r) / r_cross_s
        
        # Intersection occurs if 0 <= t <= 1 and 0 <= u <= 1
        if 0 <= t <= 1 and 0 <= u <= 1:
            # We have a crossing! Determine direction.
            # Using the sign of r_cross_s to determine which way the crossing happened.
            # If r_cross_s > 0, the trajectory crossed from the "right" of the line to the "left".
            direction = "in" if r_cross_s > 0 else "out"
            
            if self.direction_mode == "any" or self.direction_mode == direction:
                return direction
                
        return None


class ZoneManager:
    """Manages all zones and crossing lines for a camera."""

    def __init__(self, camera_config: CameraConfig, app_config: "AppConfig"):
        self.camera_config = camera_config
        self.app_config = app_config
        self.camera_id = camera_config.camera_id
        
        # Supervision PolygonZones for 'contains' checks
        self.polygon_zones: Dict[str, object] = {}
        
        # Dictionary of LineCrossing objects
        self.lines: Dict[str, LineCrossing] = {}
        
        # Stats
        self.line_counts: Dict[str, Dict[str, int]] = {}
        
        self._setup_zones()
        self._setup_lines()

    def _setup_zones(self) -> None:
        """Create supervision PolygonZone objects from config."""
        if sv is None:
            logger.warning("[%s] supervision not available — zone detection disabled", self.camera_id)
            return

        for zone_name, roi_zone in self.camera_config.roi_zones.items():
            if not roi_zone.points:
                continue
                
            try:
                polygon = np.array(roi_zone.points, dtype=np.int32)
                pz = sv.PolygonZone(
                    polygon=polygon,
                    triggering_anchors=[sv.Position.BOTTOM_CENTER],
                )
                self.polygon_zones[zone_name] = pz
                logger.debug("[%s] Zone '%s' initialized", self.camera_id, zone_name)
            except Exception as e:
                logger.error("[%s] Failed to create zone '%s': %s", self.camera_id, zone_name, e)

    def _setup_lines(self) -> None:
        """Create LineCrossing objects from config."""
        # Check if crossing_lines exists in camera_config
        lines_config = getattr(self.camera_config, "crossing_lines", {})
        
        for line_name, data in lines_config.items():
            try:
                pt_a = tuple(data.get("point_a", [0, 0]))
                pt_b = tuple(data.get("point_b", [0, 0]))
                direction = data.get("direction", "any")
                
                self.lines[line_name] = LineCrossing(
                    name=line_name,
                    point_a=pt_a,
                    point_b=pt_b,
                    direction_mode=direction
                )
                self.line_counts[line_name] = {"in": 0, "out": 0}
                logger.info("[%s] Line '%s' initialized (%s)", self.camera_id, line_name, direction)
            except Exception as e:
                logger.error("[%s] Failed to create line '%s': %s", self.camera_id, line_name, e)

    def compute_zones(self, detections: "sv.Detections") -> Dict[int, Set[str]]:
        """Find which zones each detected object is currently in.
        
        Returns:
            Dict mapping track_id to a set of zone names.
        """
        if not self.polygon_zones or detections is None or len(detections) == 0:
            return {}

        track_zones = {int(tid): set() for tid in detections.tracker_id} if detections.tracker_id is not None else {}

        if not track_zones:
            return {}

        for zone_name, pz in self.polygon_zones.items():
            # Trigger returns a boolean mask matching the detections array
            mask = pz.trigger(detections=detections)
            for i, is_in_zone in enumerate(mask):
                if is_in_zone:
                    tid = int(detections.tracker_id[i])
                    track_zones[tid].add(zone_name)

        return track_zones

    def check_crossings(self, prev_centroid: Tuple[float, float], curr_centroid: Tuple[float, float]) -> List[Tuple[str, str]]:
        """Check if movement from prev to curr crosses any lines.
        
        Returns:
            List of tuples (line_name, direction) that were crossed.
        """
        crossings = []
        for line_name, line in self.lines.items():
            direction = line.check_crossing(prev_centroid, curr_centroid)
            if direction:
                crossings.append((line_name, direction))
                self.line_counts[line_name][direction] += 1
                logger.debug("[%s] Line '%s' crossed (%s)", self.camera_id, line_name, direction)
                
        return crossings
