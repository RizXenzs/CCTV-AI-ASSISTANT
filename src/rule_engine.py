"""
rule_engine.py — Configurable rule engine for suspicious behavior scoring.

Evaluates all enabled rules against per-track features and accumulates
a suspicion score (0-100). Supports critical rules that bypass the
stability window for immediate triggering.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.config_loader import Rule
from src.feature_extractor import TrackFeatures, SceneFeatures

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of evaluating a single rule against a track."""
    rule_id: str
    triggered: bool
    score_delta: float
    reason: str
    critical: bool = False


@dataclass
class TrackScore:
    """Accumulated score for a single tracked person."""
    track_id: int
    total_score: float = 0.0
    triggered_rules: List[str] = field(default_factory=list)
    rule_reasons: List[str] = field(default_factory=list)
    has_critical: bool = False
    is_suspicious: bool = False


class RuleEngine:
    """Evaluates configurable rules against tracked person features.

    Each rule contributes a score_delta when triggered. The total score
    per person determines if behavior is suspicious.
    """

    def __init__(
        self,
        rules: List[Rule],
        suspicious_threshold: float = 70,
        critical_threshold: float = 80,
    ):
        """
        Args:
            rules: List of Rule definitions from config.
            suspicious_threshold: Score at which behavior is suspicious.
            critical_threshold: Score delta at which a rule is "critical"
                               (bypasses stability window).
        """
        self.rules = [r for r in rules if r.enabled]
        self.suspicious_threshold = suspicious_threshold
        self.critical_threshold = critical_threshold

        # Per-rule cooldown tracking: (rule_id, track_id) -> last_trigger_time
        self._rule_cooldowns: Dict[str, float] = {}

        logger.info(
            "RuleEngine initialized: %d rules enabled, threshold=%d",
            len(self.rules),
            suspicious_threshold,
        )
        for r in self.rules:
            logger.debug("  Rule: %s (score_delta=%.0f, critical=%s)", r.id, r.score_delta, r.critical)

    def evaluate(
        self,
        track_features: Dict[int, TrackFeatures],
        scene_features: SceneFeatures,
        night_mode_multiplier: float = 1.0,
    ) -> Dict[int, TrackScore]:
        """Evaluate all rules against all tracked persons.

        Args:
            track_features: Per-track computed features.
            scene_features: Scene-level features.
            night_mode_multiplier: Multiplier for score_delta if it's night.

        Returns:
            Dict of track_id -> TrackScore with accumulated scores.
        """
        now = time.time()
        results: Dict[int, TrackScore] = {}

        for track_id, features in track_features.items():
            score = TrackScore(track_id=track_id)

            for rule in self.rules:
                # Check rule-level cooldown
                cooldown_key = f"{rule.id}:{track_id}"
                if rule.cooldown_sec > 0:
                    last_trigger = self._rule_cooldowns.get(cooldown_key, 0)
                    if now - last_trigger < rule.cooldown_sec:
                        continue

                # Evaluate rule condition
                result = self._evaluate_rule(rule, features, scene_features)

                if result.triggered:
                    delta = result.score_delta
                    if scene_features.is_night_hours:
                        delta *= night_mode_multiplier

                    score.total_score += delta
                    score.triggered_rules.append(result.rule_id)
                    score.rule_reasons.append(result.reason)

                    if result.critical:
                        score.has_critical = True

                    # Update cooldown
                    self._rule_cooldowns[cooldown_key] = now

            # Cap score at 100
            score.total_score = min(score.total_score, 100.0)

            # Determine if suspicious
            score.is_suspicious = (
                score.total_score >= self.suspicious_threshold or score.has_critical
            )

            results[track_id] = score

        return results

    def _evaluate_rule(
        self,
        rule: Rule,
        features: TrackFeatures,
        scene: SceneFeatures,
    ) -> RuleResult:
        """Evaluate a single rule against track features.

        Returns:
            RuleResult with triggered status and score_delta.
        """
        cond = rule.condition
        params = cond.params

        triggered = False

        try:
            if cond.type == "zone_entry":
                # Check if person is in the specified zone
                zone = params.get("zone", "restricted_zone")
                triggered = zone in features.current_zones

            elif cond.type == "dwell_time":
                # Check if person has been in a zone for too long
                zone = params.get("zone", "door_zone")
                min_sec = float(params.get("min_sec", 25))
                if zone in features.current_zones:
                    triggered = features.dwell_time_sec >= min_sec
                else:
                    triggered = False

            elif cond.type == "speed":
                # Check for fast movement (optionally toward a zone)
                min_speed = float(params.get("min_px_per_sec", 180))
                toward_zone = params.get("toward_zone")
                if toward_zone:
                    triggered = (
                        features.approach_speed >= min_speed
                        and features.approaching_zone == toward_zone
                    )
                else:
                    triggered = features.speed_px_per_sec >= min_speed

            elif cond.type == "active_hours":
                # Check if current time is within risky hours
                triggered = scene.is_night_hours

            elif cond.type == "person_count":
                # Check for unusual number of persons
                min_count = int(params.get("min_count", 3))
                triggered = features.person_count >= min_count

            elif cond.type == "trajectory_pattern":
                # Check for back-and-forth movement
                pattern = params.get("pattern", "back_and_forth")
                min_turns = int(params.get("min_turns", 3))
                if pattern == "back_and_forth":
                    triggered = features.direction_changes >= min_turns

            elif cond.type == "bbox_aspect_change":
                # Check for crouch-like posture change
                max_ratio = float(params.get("max_height_ratio", 0.6))
                duration = float(params.get("duration_sec", 2))
                triggered = (
                    features.bbox_height_ratio_change <= max_ratio
                    and features.bbox_aspect_stable_sec >= duration
                )

            elif cond.type == "motion_only":
                # High motion without person detection (shadow/object)
                motion_str = params.get("motion_level", "high")
                min_motion = {"low": 0.01, "med": 0.03, "high": 0.06}.get(
                    str(motion_str), 0.06
                )
                person_detected = params.get("person_detected", False)
                duration = float(params.get("duration_sec", 3))

                triggered = (
                    scene.motion_level >= min_motion
                    and features.person_count == 0
                    and not person_detected
                )
                # Note: duration check would need frame accumulation — simplified here
                # The stability window in event_manager handles sustained detection

            elif cond.type == "new_track_appearance":
                # New track appearing (optionally near a restricted zone)
                near_zone = params.get("near_zone")
                within_sec = float(params.get("within_sec", 2))
                
                is_new = features.is_new_track and features.track_age_sec <= within_sec
                
                if near_zone and near_zone != "any":
                    triggered = is_new and near_zone in features.current_zones
                else:
                    triggered = is_new

            elif cond.type == "scene_change":
                # Camera tamper detection (black/covered frame)
                min_ratio = float(params.get("black_frame_ratio_min", 0.8))
                triggered = scene.black_frame_ratio >= min_ratio
                
            elif cond.type == "line_crossing":
                # Line crossing tripwire
                line_name = params.get("line_name")
                direction = params.get("direction", "any")
                
                for crossed_line, crossed_dir in features.line_crossings:
                    if not line_name or crossed_line == line_name:
                        if direction == "any" or crossed_dir == direction:
                            triggered = True
                            break

            else:
                logger.debug("Unknown rule condition type: %s", cond.type)

        except Exception as e:
            logger.error("Error evaluating rule '%s': %s", rule.id, e)

        return RuleResult(
            rule_id=rule.id,
            triggered=triggered,
            score_delta=rule.score_delta if triggered else 0.0,
            reason=rule.reason if triggered else "",
            critical=rule.critical if triggered else False,
        )

    def get_enabled_rules(self) -> List[Dict[str, Any]]:
        """Get summary of enabled rules for logging."""
        return [
            {
                "id": r.id,
                "score_delta": r.score_delta,
                "critical": r.critical,
                "reason": r.reason,
            }
            for r in self.rules
        ]

    def cleanup_cooldowns(self, max_age_sec: float = 300) -> None:
        """Remove stale cooldown entries."""
        now = time.time()
        stale = [k for k, v in self._rule_cooldowns.items() if now - v > max_age_sec]
        for k in stale:
            del self._rule_cooldowns[k]
