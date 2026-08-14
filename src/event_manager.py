"""
event_manager.py — State machine for event lifecycle management.

Manages the event state transitions per camera:
NORMAL → MOTION → PERSON_DETECTED → SUSPICIOUS → ALERTING → RESOLVED

Handles alert cooldowns, stability windows, and event creation/resolution.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from src.rule_engine import TrackScore

logger = logging.getLogger(__name__)


class EventState(Enum):
    """Event lifecycle states."""
    NORMAL = "NORMAL"
    MOTION = "MOTION"
    PERSON_DETECTED = "PERSON_DETECTED"
    SUSPICIOUS = "SUSPICIOUS"
    ALERTING = "ALERTING"
    RESOLVED = "RESOLVED"


@dataclass
class Event:
    """Represents a single suspicious event."""
    event_id: str
    camera_id: str
    state: EventState
    score: float = 0.0
    triggered_rules: List[str] = field(default_factory=list)
    track_ids: Set[int] = field(default_factory=set)
    started_at: float = 0.0
    resolved_at: Optional[float] = None
    is_critical: bool = False

    # Internal tracking
    suspicious_since: float = 0.0      # When score first exceeded threshold
    last_alert_time: float = 0.0       # When last alert was sent
    last_snapshot_time: float = 0.0    # When last periodic snapshot was taken
    alert_sent: bool = False           # Whether initial alert was sent
    snapshot_count: int = 0

    @property
    def is_active(self) -> bool:
        return self.state in (
            EventState.SUSPICIOUS,
            EventState.ALERTING,
        )

    @property
    def duration_sec(self) -> float:
        end = self.resolved_at or time.time()
        return end - self.started_at if self.started_at > 0 else 0.0


@dataclass
class CameraState:
    """Tracks the current state for a single camera."""
    camera_id: str
    state: EventState = EventState.NORMAL
    current_event: Optional[Event] = None

    # Timing
    motion_start_time: float = 0.0
    person_start_time: float = 0.0
    no_motion_since: float = 0.0
    no_person_since: float = 0.0
    score_below_since: float = 0.0

    # Cooldown
    last_alert_time: float = 0.0

    # Stats
    total_events: int = 0
    total_alerts: int = 0


class EventManager:
    """Manages event lifecycle with state machine transitions per camera.

    State transitions:
    - NORMAL → MOTION: motion detected
    - MOTION → NORMAL: no motion for 5 seconds
    - MOTION → PERSON_DETECTED: person detected
    - PERSON_DETECTED → MOTION: person lost for 10 seconds
    - PERSON_DETECTED → SUSPICIOUS: score >= threshold sustained for stability window
    - SUSPICIOUS → ALERTING: trigger alert (immediate)
    - ALERTING → ALERTING: periodic snapshots every snapshot_interval
    - ALERTING → RESOLVED: score below threshold for 30 seconds
    - RESOLVED → NORMAL: cooldown expired
    """

    def __init__(
        self,
        suspicious_threshold: float = 70,
        critical_threshold: float = 80,
        stability_window_sec: float = 3.0,
        alert_cooldown_sec: float = 60,
        snapshot_interval_sec: float = 120,
        no_motion_timeout: float = 5.0,
        no_person_timeout: float = 10.0,
        resolve_timeout: float = 30.0,
    ):
        self.suspicious_threshold = suspicious_threshold
        self.critical_threshold = critical_threshold
        self.stability_window_sec = stability_window_sec
        self.alert_cooldown_sec = alert_cooldown_sec
        self.snapshot_interval_sec = snapshot_interval_sec
        self.no_motion_timeout = no_motion_timeout
        self.no_person_timeout = no_person_timeout
        self.resolve_timeout = resolve_timeout

        # Per-camera states
        self._cameras: Dict[str, CameraState] = {}

        # Pending actions for the main loop to process
        self._pending_alerts: List[Event] = []
        self._pending_snapshots: List[Event] = []

        logger.info(
            "EventManager initialized: threshold=%.0f, stability=%.1fs, cooldown=%.0fs",
            suspicious_threshold,
            stability_window_sec,
            alert_cooldown_sec,
        )

    def get_camera_state(self, camera_id: str) -> CameraState:
        """Get or create camera state."""
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraState(camera_id=camera_id)
        return self._cameras[camera_id]

    def update(
        self,
        camera_id: str,
        has_motion: bool,
        person_count: int,
        track_scores: Dict[int, TrackScore],
    ) -> Optional[Event]:
        """Process a frame's results and update event state.

        Args:
            camera_id: Camera identifier.
            has_motion: Whether motion was detected.
            person_count: Number of persons detected.
            track_scores: Per-track suspicion scores from rule engine.

        Returns:
            Active Event if one exists, None otherwise.
        """
        now = time.time()
        cam = self.get_camera_state(camera_id)

        # Find the highest score among all tracks
        max_score = 0.0
        max_track_score: Optional[TrackScore] = None
        all_triggered_rules: List[str] = []
        suspicious_track_ids: Set[int] = set()
        has_critical = False

        for tid, ts in track_scores.items():
            if ts.total_score > max_score:
                max_score = ts.total_score
                max_track_score = ts
            if ts.is_suspicious:
                suspicious_track_ids.add(tid)
            if ts.has_critical:
                has_critical = True
            all_triggered_rules.extend(ts.triggered_rules)

        # Deduplicate rules
        all_triggered_rules = list(dict.fromkeys(all_triggered_rules))

        # --- State Machine ---

        if cam.state == EventState.NORMAL:
            if has_motion:
                cam.state = EventState.MOTION
                cam.motion_start_time = now
                cam.no_motion_since = 0.0
                logger.debug("[%s] NORMAL → MOTION", camera_id)

        elif cam.state == EventState.MOTION:
            if not has_motion:
                if cam.no_motion_since == 0:
                    cam.no_motion_since = now
                elif now - cam.no_motion_since >= self.no_motion_timeout:
                    cam.state = EventState.NORMAL
                    cam.no_motion_since = 0.0
                    logger.debug("[%s] MOTION → NORMAL (timeout)", camera_id)
            else:
                cam.no_motion_since = 0.0

            if person_count > 0:
                cam.state = EventState.PERSON_DETECTED
                cam.person_start_time = now
                cam.no_person_since = 0.0
                logger.debug("[%s] MOTION → PERSON_DETECTED", camera_id)

        elif cam.state == EventState.PERSON_DETECTED:
            if person_count == 0:
                if cam.no_person_since == 0:
                    cam.no_person_since = now
                elif now - cam.no_person_since >= self.no_person_timeout:
                    cam.state = EventState.MOTION if has_motion else EventState.NORMAL
                    cam.no_person_since = 0.0
                    logger.debug("[%s] PERSON_DETECTED → %s (person lost)", camera_id, cam.state.value)
            else:
                cam.no_person_since = 0.0

            # Check for suspicious behavior
            if max_score >= self.suspicious_threshold or has_critical:
                if has_critical:
                    # Critical rules bypass stability window
                    cam.state = EventState.SUSPICIOUS
                    logger.info(
                        "[%s] PERSON → SUSPICIOUS (critical rule, score=%.0f)",
                        camera_id,
                        max_score,
                    )
                else:
                    # Track stability window
                    if cam.current_event is None or cam.current_event.suspicious_since == 0:
                        # Start stability timer
                        if cam.current_event is None:
                            event = self._create_event(camera_id, max_score, all_triggered_rules, suspicious_track_ids)
                            event.suspicious_since = now
                            cam.current_event = event
                        else:
                            cam.current_event.suspicious_since = now

                    elif now - cam.current_event.suspicious_since >= self.stability_window_sec:
                        cam.state = EventState.SUSPICIOUS
                        logger.info(
                            "[%s] PERSON → SUSPICIOUS (sustained %.1fs, score=%.0f)",
                            camera_id,
                            self.stability_window_sec,
                            max_score,
                        )
            else:
                # Score dropped — reset stability timer
                if cam.current_event and cam.current_event.suspicious_since > 0:
                    cam.current_event.suspicious_since = 0.0

        elif cam.state == EventState.SUSPICIOUS:
            # Immediately transition to ALERTING
            cam.state = EventState.ALERTING

            # Create or update event
            if cam.current_event is None:
                cam.current_event = self._create_event(
                    camera_id, max_score, all_triggered_rules, suspicious_track_ids
                )

            cam.current_event.state = EventState.ALERTING
            cam.current_event.score = max_score
            cam.current_event.triggered_rules = all_triggered_rules
            cam.current_event.track_ids = suspicious_track_ids
            
            # If the score is above critical threshold, mark it critical even if no critical rule fired
            if has_critical or max_score >= self.critical_threshold:
                cam.current_event.is_critical = True

            # Check alert cooldown
            if now - cam.last_alert_time >= self.alert_cooldown_sec:
                cam.current_event.alert_sent = False  # Allow new alert
                self._pending_alerts.append(cam.current_event)
                cam.last_alert_time = now
                cam.total_alerts += 1
                logger.info(
                    "[%s] SUSPICIOUS → ALERTING (event=%s, score=%.0f, rules=%s)",
                    camera_id,
                    cam.current_event.event_id[:8],
                    max_score,
                    all_triggered_rules,
                )
            else:
                remaining = self.alert_cooldown_sec - (now - cam.last_alert_time)
                logger.debug(
                    "[%s] Alert suppressed (cooldown %.0fs remaining)",
                    camera_id,
                    remaining,
                )

        elif cam.state == EventState.ALERTING:
            if cam.current_event:
                # Update event with latest data
                cam.current_event.score = max_score
                cam.current_event.triggered_rules = all_triggered_rules
                cam.current_event.track_ids.update(suspicious_track_ids)

                # Check for periodic snapshot
                if now - cam.current_event.last_snapshot_time >= self.snapshot_interval_sec:
                    self._pending_snapshots.append(cam.current_event)
                    cam.current_event.last_snapshot_time = now
                    cam.current_event.snapshot_count += 1

                # Check if we should resolve
                if max_score < self.suspicious_threshold and not has_critical:
                    if cam.score_below_since == 0:
                        cam.score_below_since = now
                    elif now - cam.score_below_since >= self.resolve_timeout:
                        self._resolve_event(cam, now)
                else:
                    cam.score_below_since = 0.0

        elif cam.state == EventState.RESOLVED:
            # Wait for cooldown, then return to NORMAL
            if cam.current_event:
                resolve_time = cam.current_event.resolved_at or now
                if now - resolve_time >= self.alert_cooldown_sec:
                    cam.state = EventState.NORMAL
                    cam.current_event = None
                    cam.score_below_since = 0.0
                    logger.debug("[%s] RESOLVED → NORMAL", camera_id)

        return cam.current_event

    def _create_event(
        self,
        camera_id: str,
        score: float,
        triggered_rules: List[str],
        track_ids: Set[int],
    ) -> Event:
        """Create a new event."""
        event = Event(
            event_id=str(uuid.uuid4()),
            camera_id=camera_id,
            state=EventState.PERSON_DETECTED,
            score=score,
            triggered_rules=triggered_rules,
            track_ids=track_ids,
            started_at=time.time(),
        )
        cam = self.get_camera_state(camera_id)
        cam.total_events += 1
        logger.info(
            "[%s] Event created: %s (score=%.0f)",
            camera_id,
            event.event_id[:8],
            score,
        )
        return event

    def _resolve_event(self, cam: CameraState, now: float) -> None:
        """Resolve the current event."""
        if cam.current_event:
            cam.current_event.state = EventState.RESOLVED
            cam.current_event.resolved_at = now
            logger.info(
                "[%s] ALERTING → RESOLVED (event=%s, duration=%.0fs)",
                cam.camera_id,
                cam.current_event.event_id[:8],
                cam.current_event.duration_sec,
            )
        cam.state = EventState.RESOLVED
        cam.score_below_since = 0.0

    def pop_pending_alerts(self) -> List[Event]:
        """Get and clear pending alert events."""
        alerts = self._pending_alerts.copy()
        self._pending_alerts.clear()
        return alerts

    def pop_pending_snapshots(self) -> List[Event]:
        """Get and clear pending snapshot events."""
        snapshots = self._pending_snapshots.copy()
        self._pending_snapshots.clear()
        return snapshots

    def get_all_camera_states(self) -> Dict[str, Dict]:
        """Get summary of all camera states."""
        return {
            cid: {
                "state": cs.state.value,
                "has_event": cs.current_event is not None,
                "event_id": cs.current_event.event_id[:8] if cs.current_event else None,
                "total_events": cs.total_events,
                "total_alerts": cs.total_alerts,
            }
            for cid, cs in self._cameras.items()
        }
