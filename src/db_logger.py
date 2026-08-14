"""
db_logger.py — Async SQLite database logger for events, snapshots, tracks, and rule triggers.

Uses aiosqlite for non-blocking database operations with WAL mode for performance.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Cameras registry
CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    rtsp_url    TEXT NOT NULL,
    enabled     INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);

-- Events (one per suspicious detection session)
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    camera_id       TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'NORMAL',
    score           REAL DEFAULT 0,
    triggered_rules TEXT DEFAULT '[]',
    track_ids       TEXT DEFAULT '[]',
    started_at      TEXT NOT NULL,
    resolved_at     TEXT,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

-- Snapshots
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    snapshot_type   TEXT NOT NULL DEFAULT 'alert',
    score           REAL DEFAULT 0,
    captured_at     TEXT NOT NULL,
    sent_telegram   INTEGER DEFAULT 0,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

-- Track history
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    event_id    TEXT,
    bbox        TEXT,
    centroid    TEXT,
    speed       REAL DEFAULT 0,
    zone        TEXT,
    timestamp   TEXT NOT NULL
);

-- Rule trigger log
CREATE TABLE IF NOT EXISTS rule_triggers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL,
    rule_id         TEXT NOT NULL,
    track_id        INTEGER,
    score_delta     REAL DEFAULT 0,
    reason          TEXT,
    triggered_at    TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_events_camera     ON events(camera_id, started_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_event   ON snapshots(event_id);
CREATE INDEX IF NOT EXISTS idx_tracks_camera     ON tracks(camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_triggers_event    ON rule_triggers(event_id);
"""


class DBLogger:
    """Async database logger for the CCTV detection system."""

    def __init__(self, db_path: str = "data/cctv_events.db"):
        self.db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create database, apply schema, and enable WAL mode."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row

        # Enable WAL mode for concurrent reads + better write performance
        await self._db.execute("PRAGMA journal_mode = WAL;")
        await self._db.execute("PRAGMA synchronous = NORMAL;")
        await self._db.execute("PRAGMA foreign_keys = ON;")

        # Apply schema
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()

        logger.info("Database initialized at %s (WAL mode)", self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Database connection closed")

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db

    # ----- Camera operations -----

    async def upsert_camera(
        self, camera_id: str, name: str, rtsp_url: str, enabled: bool = True
    ) -> None:
        """Insert or update a camera record."""
        await self.db.execute(
            """
            INSERT INTO cameras (camera_id, name, rtsp_url, enabled)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                name = excluded.name,
                rtsp_url = excluded.rtsp_url,
                enabled = excluded.enabled
            """,
            (camera_id, name, rtsp_url, int(enabled)),
        )
        await self.db.commit()

    # ----- Event operations -----

    async def create_event(
        self,
        event_id: str,
        camera_id: str,
        state: str,
        score: float = 0,
        triggered_rules: Optional[List[str]] = None,
        track_ids: Optional[List[int]] = None,
    ) -> None:
        """Create a new event record."""
        now = datetime.now().isoformat()
        await self.db.execute(
            """
            INSERT INTO events (event_id, camera_id, state, score, triggered_rules, track_ids, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                camera_id,
                state,
                score,
                json.dumps(triggered_rules or []),
                json.dumps(track_ids or []),
                now,
            ),
        )
        await self.db.commit()

    async def update_event(
        self,
        event_id: str,
        state: Optional[str] = None,
        score: Optional[float] = None,
        triggered_rules: Optional[List[str]] = None,
        track_ids: Optional[List[int]] = None,
        resolved: bool = False,
    ) -> None:
        """Update an existing event record."""
        updates = []
        params: List[Any] = []

        if state is not None:
            updates.append("state = ?")
            params.append(state)
        if score is not None:
            updates.append("score = ?")
            params.append(score)
        if triggered_rules is not None:
            updates.append("triggered_rules = ?")
            params.append(json.dumps(triggered_rules))
        if track_ids is not None:
            updates.append("track_ids = ?")
            params.append(json.dumps(track_ids))
        if resolved:
            updates.append("resolved_at = ?")
            params.append(datetime.now().isoformat())

        if not updates:
            return

        params.append(event_id)
        sql = f"UPDATE events SET {', '.join(updates)} WHERE event_id = ?"
        await self.db.execute(sql, params)
        await self.db.commit()

    async def get_active_events(self, camera_id: str) -> List[Dict[str, Any]]:
        """Get all active (non-resolved) events for a camera."""
        cursor = await self.db.execute(
            "SELECT * FROM events WHERE camera_id = ? AND resolved_at IS NULL ORDER BY started_at DESC",
            (camera_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ----- Snapshot operations -----

    async def log_snapshot(
        self,
        event_id: str,
        camera_id: str,
        file_path: str,
        snapshot_type: str = "alert",
        score: float = 0,
        sent_telegram: bool = False,
    ) -> int:
        """Log a snapshot to the database. Returns the snapshot_id."""
        now = datetime.now().isoformat()
        cursor = await self.db.execute(
            """
            INSERT INTO snapshots (event_id, camera_id, file_path, snapshot_type, score, captured_at, sent_telegram)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, camera_id, file_path, snapshot_type, score, now, int(sent_telegram)),
        )
        await self.db.commit()
        return cursor.lastrowid or 0

    async def mark_snapshot_sent(self, snapshot_id: int) -> None:
        """Mark a snapshot as sent to Telegram."""
        await self.db.execute(
            "UPDATE snapshots SET sent_telegram = 1 WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        await self.db.commit()

    # ----- Track operations -----

    async def log_track(
        self,
        camera_id: str,
        track_id: int,
        event_id: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        centroid: Optional[List[float]] = None,
        speed: float = 0,
        zone: Optional[str] = None,
    ) -> None:
        """Log a track data point."""
        now = datetime.now().isoformat()
        await self.db.execute(
            """
            INSERT INTO tracks (camera_id, track_id, event_id, bbox, centroid, speed, zone, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                track_id,
                event_id,
                json.dumps(bbox) if bbox else None,
                json.dumps(centroid) if centroid else None,
                speed,
                zone,
                now,
            ),
        )
        # Don't commit every track — batch commits handled externally
        # for performance (tracks are high-frequency)

    async def flush_tracks(self) -> None:
        """Commit any pending track writes."""
        await self.db.commit()

    # ----- Rule trigger operations -----

    async def log_rule_trigger(
        self,
        event_id: str,
        rule_id: str,
        track_id: Optional[int] = None,
        score_delta: float = 0,
        reason: str = "",
    ) -> None:
        """Log a rule trigger event."""
        now = datetime.now().isoformat()
        await self.db.execute(
            """
            INSERT INTO rule_triggers (event_id, rule_id, track_id, score_delta, reason, triggered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, rule_id, track_id, score_delta, reason, now),
        )
        await self.db.commit()

    # ----- Cleanup operations -----

    async def cleanup_old_records(self, days: int = 30) -> int:
        """Delete records older than N days. Returns count of deleted events."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Get event IDs to delete
        cursor = await self.db.execute(
            "SELECT event_id FROM events WHERE started_at < ?", (cutoff,)
        )
        old_events = [row[0] for row in await cursor.fetchall()]

        if not old_events:
            return 0

        placeholders = ",".join("?" * len(old_events))

        # Delete related records first (foreign key order)
        await self.db.execute(
            f"DELETE FROM rule_triggers WHERE event_id IN ({placeholders})",
            old_events,
        )
        await self.db.execute(
            f"DELETE FROM snapshots WHERE event_id IN ({placeholders})",
            old_events,
        )
        await self.db.execute(
            f"DELETE FROM tracks WHERE event_id IN ({placeholders})",
            old_events,
        )
        await self.db.execute(
            f"DELETE FROM events WHERE event_id IN ({placeholders})",
            old_events,
        )

        # Also clean up old tracks without events
        await self.db.execute(
            "DELETE FROM tracks WHERE timestamp < ? AND event_id IS NULL",
            (cutoff,),
        )

        await self.db.commit()
        logger.info("Cleaned up %d old events (older than %d days)", len(old_events), days)
        return len(old_events)

    # ----- Statistics -----

    async def get_today_stats(self) -> Dict[str, Any]:
        """Get summary statistics for today."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        cursor = await self.db.execute("SELECT COUNT(*) FROM events WHERE started_at >= ?", (today_start,))
        total_events = (await cursor.fetchone())[0]
        
        cursor = await self.db.execute("SELECT COUNT(*) FROM events WHERE started_at >= ? AND score >= 70", (today_start,))
        suspicious_events = (await cursor.fetchone())[0]
        
        cursor = await self.db.execute("SELECT COUNT(*) FROM events WHERE started_at >= ? AND score >= 80", (today_start,))
        critical_events = (await cursor.fetchone())[0]
        
        return {
            "total_events": total_events,
            "suspicious_events": suspicious_events,
            "critical_events": critical_events,
            "person_detections": total_events, # Simplification for now
            "animal_detections": 0,
            "masuk": 0,
            "keluar": 0,
        }
