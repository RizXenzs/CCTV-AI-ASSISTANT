import os
import cv2
import time
import logging
from collections import deque
from threading import Thread, Lock
from typing import Optional, List

logger = logging.getLogger(__name__)

class EventRecorder:
    """Records video clips for critical events with pre-event buffering."""

    def __init__(self, camera_id: str, fps: int = 25, resolution: tuple = (1920, 1080), 
                 pre_sec: int = 5, post_sec: int = 15, output_dir: str = "data/recordings"):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.buffer = deque(maxlen=fps * pre_sec)
        
        self.is_recording = False
        self.current_writer: Optional[cv2.VideoWriter] = None
        self.recording_end_time = 0.0
        self.current_event_id: Optional[str] = None
        self.current_file_path: Optional[str] = None
        
        self._lock = Lock()

    def add_frame(self, frame) -> None:
        """Add a frame to the buffer. If recording, also write to disk."""
        with self._lock:
            # Always buffer the frame
            self.buffer.append(frame)
            
            # If we are recording
            if self.is_recording:
                now = time.time()
                if now <= self.recording_end_time:
                    if self.current_writer:
                        self.current_writer.write(frame)
                else:
                    self._stop_recording_internal()

    def start_recording(self, event_id: str) -> Optional[str]:
        """Start recording an event. Dumps the pre-buffer and prepares for post-buffering."""
        with self._lock:
            if self.is_recording:
                logger.warning("[%s] Already recording event %s. Ignoring start for %s", 
                               self.camera_id, self.current_event_id, event_id)
                return None
                
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.camera_id}_{event_id}_{timestamp}.mp4"
            filepath = os.path.join(self.output_dir, filename)
            
            # Get exact resolution from the first buffered frame
            if len(self.buffer) > 0:
                height, width = self.buffer[0].shape[:2]
                record_resolution = (width, height)
            else:
                record_resolution = self.resolution

            # Use 'mp4v' for MP4 format
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.current_writer = cv2.VideoWriter(filepath, fourcc, self.fps, record_resolution)
            
            if not self.current_writer.isOpened():
                logger.error("[%s] Failed to open VideoWriter for %s", self.camera_id, filepath)
                self.current_writer = None
                return None
            
            # Dump pre-buffer
            for frame in self.buffer:
                self.current_writer.write(frame)
                
            self.is_recording = True
            self.current_event_id = event_id
            self.current_file_path = filepath
            self.recording_end_time = time.time() + self.post_sec
            
            logger.info("[%s] Started recording event %s -> %s", self.camera_id, event_id, filepath)
            return filepath
            
    def get_current_recording_path(self) -> Optional[str]:
        return self.current_file_path

    def stop_recording(self) -> None:
        """Manually stop the current recording."""
        with self._lock:
            self._stop_recording_internal()

    def _stop_recording_internal(self) -> None:
        if self.is_recording:
            if self.current_writer:
                self.current_writer.release()
                self.current_writer = None
            logger.info("[%s] Stopped recording event %s", self.camera_id, self.current_event_id)
            self.is_recording = False
            
    def cleanup_old_recordings(self, days: int = 30) -> int:
        """Delete recordings older than specified days."""
        now = time.time()
        deleted = 0
        for filename in os.listdir(self.output_dir):
            if filename.startswith(f"{self.camera_id}_") and filename.endswith(".mp4"):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        mtime = os.path.getmtime(filepath)
                        if (now - mtime) > (days * 86400):
                            os.remove(filepath)
                            deleted += 1
                except Exception as e:
                    logger.error("Error deleting old recording %s: %s", filepath, e)
        return deleted
