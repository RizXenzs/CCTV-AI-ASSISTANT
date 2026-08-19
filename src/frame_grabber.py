"""
frame_grabber.py — Threaded RTSP/HLS frame reader with reconnection logic.

Uses a dedicated thread per camera to continuously read frames from RTSP or HLS (.m3u8) streams.
Always provides the latest frame (drops old frames to prevent latency buildup).
"""

import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# OPTIMASI HLS/RTSP: Paksa FFmpeg membuang buffer agar selalu mendapatkan frame live terbaru
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|sync;ext"

logger = logging.getLogger(__name__)


class FrameGrabber:
    """Threaded RTSP/HLS/IP camera frame grabber.

    Runs a background thread that continuously reads from the video source (.m3u8 / RTSP)
    and stores only the most recent frame. This prevents buffer buildup and ensures
    the processing pipeline always works on the latest live frame.
    """

    def __init__(
        self,
        camera_id: str = "cam_01",
        stream_url: str = "",
        rtsp_url: Optional[str] = None,
        name: str = "",
        reconnect_max_retries: int = 5,
        reconnect_base_delay: float = 2.0,
        reconnect_max_delay: float = 60.0,
    ):
        self.camera_id = camera_id
        self.stream_url = stream_url or rtsp_url or ""
        self.name = name or camera_id

        # Reconnection settings
        self._reconnect_max_retries = reconnect_max_retries
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay

        # Thread-safe frame storage
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_timestamp: float = 0.0

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._connected = threading.Event()

        # Stats
        self._fps_actual: float = 0.0
        self._frame_count: int = 0
        self._drop_count: int = 0
        self._consecutive_failures: int = 0

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self) -> None:
        """Start the frame grabber thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[%s] Frame grabber already running", self.camera_id)
            return

        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"grabber-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[%s] Frame grabber started for %s", self.camera_id, self.name)

    def stop(self) -> None:
        """Stop the frame grabber thread."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._connected.clear()
        logger.info("[%s] Frame grabber stopped", self.camera_id)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get the most recent frame and its timestamp."""
        with self._frame_lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_timestamp

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """Create and configure a VideoCapture instance (supports RTSP & HLS HTTP/HTTPS)."""
        try:
            if not self.stream_url:
                logger.error("[%s] Stream URL is empty", self.camera_id)
                return None

            cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logger.error("[%s] Failed to open stream: %s", self.camera_id, self.stream_url)
                return None

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            logger.info(
                "[%s] Stream opened — %.0fx%.0f @ %.1f fps",
                self.camera_id,
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS),
            )
            return cap

        except Exception as e:
            logger.error("[%s] Error creating capture: %s", self.camera_id, e)
            return None

    def _capture_loop(self) -> None:
        """Main capture loop running in a background thread."""
        cap: Optional[cv2.VideoCapture] = None
        fps_timer = time.monotonic()
        fps_frame_count = 0

        while self._running.is_set():
            if cap is None or not cap.isOpened():
                self._connected.clear()
                cap = self._try_reconnect()
                if cap is None:
                    if self._running.is_set():
                        logger.warning(
                            "[%s] Reconnection exhausted, cooling down 30s...",
                            self.camera_id,
                        )
                        self._running.wait(timeout=30.0)
                    continue
                self._connected.set()
                self._consecutive_failures = 0
                fps_timer = time.monotonic()
                fps_frame_count = 0

            try:
                ret, frame = cap.read()
            except Exception as e:
                logger.error("[%s] Read error: %s", self.camera_id, e)
                ret = False
                frame = None

            if not ret or frame is None:
                self._consecutive_failures += 1
                self._drop_count += 1
                time.sleep(0.05)

                if self._consecutive_failures > 30:
                    logger.warning(
                        "[%s] Too many consecutive read failures (%d), reconnecting...",
                        self.camera_id,
                        self._consecutive_failures,
                    )
                    cap.release()
                    cap = None
                continue

            self._consecutive_failures = 0
            with self._frame_lock:
                self._frame = frame
                self._frame_timestamp = time.time()

            self._frame_count += 1
            fps_frame_count += 1

            elapsed = time.monotonic() - fps_timer
            if elapsed >= 2.0:
                self._fps_actual = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.monotonic()

        if cap is not None:
            cap.release()
        logger.info("[%s] Capture loop exited", self.camera_id)

    def _try_reconnect(self) -> Optional[cv2.VideoCapture]:
        """Attempt to reconnect with exponential backoff."""
        for attempt in range(1, self._reconnect_max_retries + 1):
            if not self._running.is_set():
                return None

            logger.info(
                "[%s] Reconnection attempt %d/%d...",
                self.camera_id,
                attempt,
                self._reconnect_max_retries,
            )

            cap = self._create_capture()
            if cap is not None and cap.isOpened():
                logger.info("[%s] Reconnected successfully", self.camera_id)
                return cap

            delay = min(
                self._reconnect_base_delay * (2 ** (attempt - 1)),
                self._reconnect_max_delay,
            )
            logger.info("[%s] Waiting %.1fs before next attempt...", self.camera_id, delay)
            self._running.wait(timeout=delay)

        return None

    def get_health_info(self) -> dict:
        """Get health metrics for monitoring."""
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "running": self.is_running,
            "connected": self.is_connected,
            "fps": round(self._fps_actual, 1),
            "total_frames": self._frame_count,
            "dropped_frames": self._drop_count,
        }