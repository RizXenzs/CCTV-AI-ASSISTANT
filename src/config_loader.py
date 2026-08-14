"""
config_loader.py — Load and validate configuration from YAML + .env files.

Supports environment variable interpolation (${VAR_NAME} syntax in YAML),
validation of all required fields, and sensible defaults.
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for typed config access
# ---------------------------------------------------------------------------

@dataclass
class ROIZone:
    """A polygon zone definition."""
    name: str
    points: List[List[int]]
    zone_type: str = "general" # 'general', 'door', 'garage', 'yard', etc.


@dataclass
class CameraConfig:
    """Per-camera configuration."""
    camera_id: str
    name: str
    rtsp_url: str
    enabled: bool = True
    active_hours: Optional[str] = None  # e.g. "22:00-05:00"
    roi_zones: Dict[str, ROIZone] = field(default_factory=dict)
    crossing_lines: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelegramConfig:
    """Telegram notification settings."""
    bot_token: str = ""
    chat_id: str = ""
    retry_count: int = 3
    retry_backoff_base: float = 1.0
    rate_limit_interval: float = 3.0
    send_timeout: int = 30


@dataclass
class DatabaseConfig:
    """Database storage settings."""
    path: str = "data/cctv_events.db"
    cleanup_days: int = 30
    snapshot_dir: str = "snapshots"


@dataclass
class RuleCondition:
    """A rule condition from rules.yaml."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """A single suspicious-behavior rule."""
    id: str
    enabled: bool
    condition: RuleCondition
    score_delta: float
    cooldown_sec: float = 0
    critical: bool = False
    reason: str = ""


@dataclass
class AppConfig:
    """Top-level application configuration."""
    # Cameras
    cameras: List[CameraConfig] = field(default_factory=list)

    # Device & model
    device_mode: str = "auto"
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    input_resolution: Tuple[int, int] = (640, 480)

    # Performance
    detect_fps_target: int = 10
    frame_skip: int = 0
    motion_gate_enabled: bool = True

    # Motion detection
    motion_sensitivity: str = "med"
    motion_history: int = 500
    motion_var_threshold: int = 16

    # Suspicious scoring
    suspicious_score_threshold: float = 70
    stability_window_sec: float = 3.0
    critical_score_threshold: float = 80

    # Alerting
    alert_cooldown_sec: float = 60
    snapshot_interval_sec: float = 120
    snapshot_quality: int = 85

    # Night Mode
    night_mode_start: str = "22:00"
    night_mode_end: str = "05:00"
    night_mode_multiplier: float = 2.0

    # Sub-configs
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Rules
    rules: List[Rule] = field(default_factory=list)

    # Logging
    log_level: str = "INFO"
    log_file: str = "data/cctv_ai.log"


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name, "")
        if not env_val:
            logger.warning("Environment variable %s is not set", var_name)
        return env_val
    return _ENV_PATTERN.sub(_replacer, value)


def _deep_interpolate(obj: Any) -> Any:
    """Recursively interpolate env vars in a nested dict/list structure."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    elif isinstance(obj, dict):
        return {k: _deep_interpolate(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_interpolate(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_camera(raw: Dict[str, Any]) -> CameraConfig:
    """Parse a single camera definition from YAML."""
    zones = {}
    raw_zones = raw.get("roi_zones") or {}
    for zone_name, zone_data in raw_zones.items():
        points = zone_data.get("points", [])
        zone_type = zone_data.get("type", "general")
        zones[zone_name] = ROIZone(name=zone_name, points=points, zone_type=zone_type)

    crossing_lines = raw.get("crossing_lines", {})

    return CameraConfig(
        camera_id=raw["camera_id"],
        name=raw["name"],
        rtsp_url=raw["rtsp_url"],
        enabled=raw.get("enabled", True),
        active_hours=raw.get("active_hours"),
        roi_zones=zones,
        crossing_lines=crossing_lines,
    )


def _parse_telegram(raw: Dict[str, Any]) -> TelegramConfig:
    """Parse Telegram configuration."""
    return TelegramConfig(
        bot_token=str(raw.get("bot_token", "")),
        chat_id=str(raw.get("chat_id", "")),
        retry_count=int(raw.get("retry_count", 3)),
        retry_backoff_base=float(raw.get("retry_backoff_base", 1.0)),
        rate_limit_interval=float(raw.get("rate_limit_interval", 3.0)),
        send_timeout=int(raw.get("send_timeout", 30)),
    )


def _parse_database(raw: Dict[str, Any]) -> DatabaseConfig:
    """Parse database configuration."""
    return DatabaseConfig(
        path=str(raw.get("path", "data/cctv_events.db")),
        cleanup_days=int(raw.get("cleanup_days", 30)),
        snapshot_dir=str(raw.get("snapshot_dir", "snapshots")),
    )


def _parse_rule_condition(raw: Dict[str, Any]) -> RuleCondition:
    """Parse a rule condition, separating 'type' from the rest as params."""
    cond_type = raw.pop("type", "unknown")
    return RuleCondition(type=cond_type, params=raw)


def _parse_rule(raw: Dict[str, Any]) -> Rule:
    """Parse a single rule definition."""
    condition_raw = dict(raw.get("condition", {}))  # copy to avoid mutation
    return Rule(
        id=raw["id"],
        enabled=raw.get("enabled", True),
        condition=_parse_rule_condition(condition_raw),
        score_delta=float(raw.get("score_delta", 0)),
        cooldown_sec=float(raw.get("cooldown_sec", 0)),
        critical=raw.get("critical", False),
        reason=raw.get("reason", ""),
    )


# ---------------------------------------------------------------------------
# Main config loader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Loads and provides typed access to the application configuration."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        rules_path: str = "config/rules.yaml",
        env_path: str = ".env",
    ):
        self.config_path = Path(config_path)
        self.rules_path = Path(rules_path)
        self.env_path = Path(env_path)
        self._config: Optional[AppConfig] = None

    def load(self) -> AppConfig:
        """Load all configuration sources and return a validated AppConfig."""
        # 1. Load .env file
        if self.env_path.exists():
            load_dotenv(self.env_path)
            logger.info("Loaded environment from %s", self.env_path)
        else:
            logger.warning(".env file not found at %s", self.env_path)

        # 2. Load main config.yaml
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

        # 3. Interpolate environment variables
        raw_config = _deep_interpolate(raw_config)

        # 4. Load rules.yaml
        rules: List[Rule] = []
        if self.rules_path.exists():
            with open(self.rules_path, "r", encoding="utf-8") as f:
                raw_rules = yaml.safe_load(f) or {}
            for rule_raw in raw_rules.get("rules", []):
                rules.append(_parse_rule(rule_raw))
            logger.info("Loaded %d rules from %s", len(rules), self.rules_path)
        else:
            logger.warning("Rules file not found: %s", self.rules_path)

        # 5. Parse cameras
        cameras = []
        for cam_raw in raw_config.get("cameras", []):
            cameras.append(_parse_camera(cam_raw))

        # 6. Parse input_resolution
        res_raw = raw_config.get("input_resolution", [640, 480])
        input_resolution = (int(res_raw[0]), int(res_raw[1]))

        # 7. Parse motion_sensitivity — normalize to numeric if string
        motion_sens = raw_config.get("motion_sensitivity", "med")

        # 8. Build AppConfig
        telegram_raw = raw_config.get("telegram", {})
        database_raw = raw_config.get("database", {})

        self._config = AppConfig(
            cameras=cameras,
            device_mode=str(raw_config.get("device_mode", "auto")),
            model_name=str(raw_config.get("model_name", "yolov8n.pt")),
            confidence_threshold=float(raw_config.get("confidence_threshold", 0.4)),
            input_resolution=input_resolution,
            detect_fps_target=int(raw_config.get("detect_fps_target", 10)),
            frame_skip=int(raw_config.get("frame_skip", 0)),
            motion_gate_enabled=bool(raw_config.get("motion_gate_enabled", True)),
            motion_sensitivity=str(motion_sens),
            motion_history=int(raw_config.get("motion_history", 500)),
            motion_var_threshold=int(raw_config.get("motion_var_threshold", 16)),
            suspicious_score_threshold=float(raw_config.get("suspicious_score_threshold", 70)),
            stability_window_sec=float(raw_config.get("stability_window_sec", 3.0)),
            critical_score_threshold=float(raw_config.get("critical_score_threshold", 80)),
            alert_cooldown_sec=float(raw_config.get("alert_cooldown_sec", 60)),
            snapshot_interval_sec=float(raw_config.get("snapshot_interval_sec", 120)),
            snapshot_quality=int(raw_config.get("snapshot_quality", 85)),
            night_mode_start=str(raw_config.get("night_mode_start", "22:00")),
            night_mode_end=str(raw_config.get("night_mode_end", "05:00")),
            night_mode_multiplier=float(raw_config.get("night_mode_multiplier", 2.0)),
            telegram=_parse_telegram(telegram_raw),
            database=_parse_database(database_raw),
            rules=rules,
            log_level=str(raw_config.get("log_level", "INFO")),
            log_file=str(raw_config.get("log_file", "data/cctv_ai.log")),
        )

        self._validate()
        logger.info(
            "Configuration loaded: %d cameras, %d rules, device=%s",
            len(self._config.cameras),
            len(self._config.rules),
            self._config.device_mode,
        )
        return self._config

    @property
    def config(self) -> AppConfig:
        """Get the loaded configuration. Raises if not yet loaded."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config

    def _validate(self) -> None:
        """Validate critical configuration values."""
        cfg = self._config
        assert cfg is not None

        if not cfg.cameras:
            logger.warning("No cameras configured — system will idle.")

        if not cfg.telegram.bot_token or cfg.telegram.bot_token == "your_bot_token_here":
            logger.warning("Telegram bot_token not configured — alerts will be disabled.")

        if not cfg.telegram.chat_id or cfg.telegram.chat_id == "your_chat_id_here":
            logger.warning("Telegram chat_id not configured — alerts will be disabled.")

        if cfg.suspicious_score_threshold < 0 or cfg.suspicious_score_threshold > 100:
            raise ValueError(
                f"suspicious_score_threshold must be 0-100, got {cfg.suspicious_score_threshold}"
            )

        if cfg.snapshot_interval_sec < 10:
            raise ValueError(
                f"snapshot_interval_sec too low ({cfg.snapshot_interval_sec}), minimum 10"
            )

        enabled_count = sum(1 for c in cfg.cameras if c.enabled)
        logger.info("Enabled cameras: %d / %d", enabled_count, len(cfg.cameras))


def get_motion_threshold(sensitivity: str) -> float:
    """Convert motion sensitivity string to a numeric threshold (0.0-1.0).

    Lower threshold = more sensitive (more motion detected).
    """
    mapping = {
        "low": 0.05,
        "med": 0.02,
        "high": 0.008,
    }
    try:
        return float(sensitivity)
    except ValueError:
        return mapping.get(sensitivity.lower(), 0.02)
