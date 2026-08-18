import os
import time
import shutil
import logging
import asyncio
from typing import Optional
from pathlib import Path

from src.config_loader import StorageConfig
from src.db_logger import DBLogger
from src.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

class StorageManager:
    """Monitors storage usage and automatically cleans up old events/videos."""

    def __init__(
        self,
        config: StorageConfig,
        db: DBLogger,
        telegram: Optional[TelegramNotifier] = None,
        data_dir: str = "data"
    ):
        self.config = config
        self.db = db
        self.telegram = telegram
        self.data_dir = Path(data_dir)
        
        self.recordings_dir = self.data_dir / "recordings"
        self.snapshots_dir = self.data_dir / "snapshots"
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        self._last_warning_sent = 0.0

    def start(self) -> None:
        """Start the background monitoring task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop(), name="storage_manager")
        logger.info("StorageManager started. Max usage: %.1f%%", self.config.auto_delete_threshold)

    def stop(self) -> None:
        """Stop the background monitoring task."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self) -> None:
        """Periodic loop to check storage and cleanup if needed."""
        # Wait a bit before starting the first check
        await asyncio.sleep(10)
        
        while self._running:
            try:
                await self.check_and_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in StorageManager: %s", e)
            
            # Check every 5 minutes
            await asyncio.sleep(300)

    def get_storage_stats(self) -> dict:
        """Get current storage statistics."""
        try:
            total, used, free = shutil.disk_usage(str(self.data_dir.absolute()))
            usage_percent = (used / total) * 100
            return {
                "total_gb": round(total / (1024**3), 1),
                "used_gb": round(used / (1024**3), 1),
                "free_gb": round(free / (1024**3), 1),
                "usage_percent": round(usage_percent, 1)
            }
        except Exception as e:
            logger.error("Error getting disk usage: %s", e)
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}

    async def check_and_cleanup(self) -> None:
        """Check usage and perform cleanup if thresholds are exceeded."""
        stats = self.get_storage_stats()
        usage = stats["usage_percent"]
        now = time.time()
        
        logger.debug("Storage usage: %.1f%%", usage)

        # 1. Routine cleanup (retention_days)
        deleted = await self.db.cleanup_old_records(self.config.retention_days)
        if deleted > 0:
            logger.info("Routine cleanup removed %d events older than %d days", deleted, self.config.retention_days)
            # Idealnya kita juga menghapus file-file orphaned disini
            pass

        # 2. Threshold checks
        if usage >= self.config.auto_delete_threshold:
            logger.warning("Storage critical (%.1f%% >= %.1f%%). Starting auto-delete.", usage, self.config.auto_delete_threshold)
            await self._emergency_cleanup()
            
            # Send warning if we haven't recently (cooldown 1 hour)
            if self.telegram and (now - self._last_warning_sent) > 3600:
                await self.telegram.send_message(
                    f"🚨 *CRITICAL STORAGE WARNING* 🚨\n\n"
                    f"Storage is at {usage:.1f}%.\n"
                    f"System is auto-deleting oldest non-critical events to free space."
                )
                self._last_warning_sent = now

        elif usage >= self.config.critical_percent:
            if self.telegram and (now - self._last_warning_sent) > 3600:
                await self.telegram.send_message(
                    f"⚠️ *STORAGE WARNING* ⚠️\n\n"
                    f"Storage is reaching critical levels ({usage:.1f}%)."
                )
                self._last_warning_sent = now

    async def _emergency_cleanup(self) -> None:
        """Delete oldest non-critical events until usage drops below critical_percent."""
        while self.get_storage_stats()["usage_percent"] >= self.config.critical_percent:
            # Find the 10 oldest non-critical events
            # (Assuming score < 80 is non-critical)
            cursor = await self.db.db.execute(
                "SELECT event_id FROM events WHERE score < 80 ORDER BY started_at ASC LIMIT 10"
            )
            old_events = [row[0] for row in await cursor.fetchall()]
            
            if not old_events:
                # No non-critical events left!
                logger.error("Storage still full, but no non-critical events left to delete!")
                break
                
            placeholders = ",".join("?" * len(old_events))
            
            # Delete from DB
            await self.db.db.execute(f"DELETE FROM rule_triggers WHERE event_id IN ({placeholders})", old_events)
            await self.db.db.execute(f"DELETE FROM snapshots WHERE event_id IN ({placeholders})", old_events)
            await self.db.db.execute(f"DELETE FROM tracks WHERE event_id IN ({placeholders})", old_events)
            await self.db.db.execute(f"DELETE FROM events WHERE event_id IN ({placeholders})", old_events)
            await self.db.db.commit()
            
            # Delete files
            self._delete_files_for_events(old_events)
            
            logger.info("Emergency cleanup deleted %d old events.", len(old_events))
            await asyncio.sleep(1) # Yield to event loop

    def _delete_files_for_events(self, event_ids: list) -> None:
        """Delete physical files (.mp4 and .jpg) for specific event IDs."""
        if not event_ids: return
        
        dirs_to_check = [self.recordings_dir, self.snapshots_dir]
        for d in dirs_to_check:
            if not d.exists(): continue
            for f in d.iterdir():
                if not f.is_file(): continue
                for eid in event_ids:
                    if eid in f.name:
                        try:
                            f.unlink()
                            logger.debug("Deleted file: %s", f.name)
                        except Exception as e:
                            logger.error("Failed to delete %s: %s", f.name, e)
                        break
