"""
routes.py — FastAPI routes for the CCTV Dashboard.
Serves MJPEG live streams, events, snapshots, camera management, and Telegram config.
"""

import asyncio
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from src.main import CCTVApplication

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Dependency to get app instance ---
def get_app(request: Request) -> CCTVApplication:
    return request.app.state.cctv_app

# --- Models ---
class CameraStatus(BaseModel):
    camera_id: str
    name: str
    state: str
    fps: float
    active_tracks: int
    is_connected: bool

class CameraAddRequest(BaseModel):
    camera_id: str
    name: str
    rtsp_url: str
    enabled: bool = True

class TelegramConfigRequest(BaseModel):
    bot_token: str
    chat_id: str

class TelegramTestRequest(BaseModel):
    bot_token: str
    chat_id: str
    message: str = "🔔 Test notification from CCTV AI Dashboard"

# --- Live Streaming ---
async def mjpeg_generator(app: CCTVApplication, camera_id: str):
    """Generator for MJPEG stream from the latest camera frame."""
    pipeline = app.pipelines.get(camera_id)
    if not pipeline:
        yield b""
        return

    # Frame interval for ~15 FPS stream to the browser
    frame_time = 1.0 / 15.0

    while not app.shutdown_event.is_set():
        start_time = asyncio.get_event_loop().time()

        # Get latest frame from the grabber
        frame, _ = pipeline.grabber.get_latest_frame()

        try:
            if frame is not None:
                # Process copy to avoid mutating the original
                display_frame = frame.copy()
                
                # Get scale factors (processed coords to original frame)
                orig_h, orig_w = display_frame.shape[:2]
                proc_w, proc_h = app.config.input_resolution
                scale_x = orig_w / proc_w
                scale_y = orig_h / proc_h

                # Draw AI Detections (Bounding Boxes)
                import time as _time
                active_tracks = pipeline.tracker.get_active_tracks()
                all_tracks = pipeline.tracker.get_all_tracks()
                now = _time.time()
                
                person_count = sum(1 for t in active_tracks.values() if t.class_type == "human")
                animal_count = sum(1 for t in active_tracks.values() if t.class_type == "animal")
                
                # Draw recently-lost tracks (faded yellow, last 3 seconds)
                for tid, track in all_tracks.items():
                    if track.is_active or tid in active_tracks:
                        continue
                    if (now - track.last_seen) > 3.0:
                        continue
                    bbox = track.latest_bbox
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = bbox
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                    cv2.putText(display_frame, f"Lost #{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                
                # Draw active tracks
                for track in active_tracks.values():
                    bbox = track.latest_bbox
                    if not bbox:
                        continue
                        
                    x1, y1, x2, y2 = bbox
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Get confidence from latest point
                    conf = track.points[-1].confidence if track.points else 0.0
                    
                    if track.class_type == "animal":
                        color = (255, 165, 0) # Orange/Blueish in BGR for animal
                        label = f"Animal {track.track_id} ({conf:.0%})"
                    else:
                        # Human tracking
                        color = (0, 255, 0) # Default green
                        track_score_obj = pipeline.latest_track_scores.get(track.track_id)
                        track_score = track_score_obj.total_score if track_score_obj else 0.0
                        
                        if track_score > pipeline.event_manager.suspicious_threshold:
                            color = (0, 0, 255) # Red for suspicious
                        label = f"Human {track.track_id} ({conf:.0%})"
                    
                    # Draw thick box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with confidence
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x1, y1 - 28), (x1 + w + 6, y1), color, -1)
                    cv2.putText(display_frame, label, (x1 + 3, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Draw person count HUD overlay (top-left)
                hud_text = f"Persons: {person_count} | Animals: {animal_count}"
                cv2.rectangle(display_frame, (5, 5), (280, 35), (0, 0, 0), -1)
                cv2.putText(display_frame, hud_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if person_count == 0 else (0, 0, 255), 2)

                # Encode frame to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', display_frame, encode_param)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("Error in mjpeg_generator: %s", e)
            await asyncio.sleep(0.1)

        # Control streaming FPS
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed < frame_time:
            await asyncio.sleep(frame_time - elapsed)


@router.get("/stream/{camera_id}")
async def video_stream(camera_id: str, app: CCTVApplication = Depends(get_app)):
    """Live MJPEG video stream for a specific camera."""
    if camera_id not in app.pipelines:
        raise HTTPException(status_code=404, detail="Camera not found")

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        mjpeg_generator(app, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers
    )

# --- Status & Config ---
@router.get("/status", response_model=List[CameraStatus])
async def get_status(app: CCTVApplication = Depends(get_app)):
    """Get the current status of all cameras."""
    status_list = []

    if not app.event_manager:
        return []

    cam_states = app.event_manager.get_all_camera_states()

    for cam_id, pipeline in app.pipelines.items():
        stats = pipeline.get_stats()
        state_info = cam_states.get(cam_id, {"state": "UNKNOWN"})

        status_list.append(CameraStatus(
            camera_id=cam_id,
            name=pipeline.camera_config.name,
            state=state_info["state"],
            fps=stats["actual_fps"],
            active_tracks=stats["active_tracks"],
            is_connected=stats["grabber"]["connected"]
        ))

    return status_list

@router.get("/config")
async def get_config(app: CCTVApplication = Depends(get_app)):
    """Get the current system configuration."""
    if not app.config:
        return {}

    import dataclasses
    return dataclasses.asdict(app.config)

# --- Events & Snapshots ---
@router.get("/events")
async def get_recent_events(limit: int = 50, app: CCTVApplication = Depends(get_app)):
    """Get recent suspicious events from the database."""
    if not app.db:
        return []

    cursor = await app.db.db.execute(
        """
        SELECT e.*, c.name as camera_name 
        FROM events e
        JOIN cameras c ON e.camera_id = c.camera_id
        ORDER BY e.started_at DESC LIMIT ?
        """,
        (limit,)
    )
    rows = await cursor.fetchall()

    events = []
    for row in rows:
        event = dict(row)
        event['triggered_rules'] = json.loads(event['triggered_rules']) if event['triggered_rules'] else []

        snap_cursor = await app.db.db.execute(
            "SELECT snapshot_id FROM snapshots WHERE event_id = ? AND snapshot_type = 'alert' LIMIT 1",
            (event['event_id'],)
        )
        snap_row = await snap_cursor.fetchone()
        event['has_snapshot'] = snap_row is not None
        if snap_row:
            event['snapshot_id'] = snap_row[0]

        # Check for video recording
        recording_dir = Path("data/recordings")
        event['has_video'] = False
        if recording_dir.exists():
            for _ in recording_dir.glob(f"*{event['event_id']}*.mp4"):
                event['has_video'] = True
                break

        events.append(event)

    return events

@router.get("/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: int, app: CCTVApplication = Depends(get_app)):
    """Get a snapshot image by ID."""
    if not app.db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    cursor = await app.db.db.execute(
        "SELECT file_path FROM snapshots WHERE snapshot_id = ?",
        (snapshot_id,)
    )
    row = await cursor.fetchone()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    file_path = row[0]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Snapshot file missing on disk")

    return FileResponse(file_path, media_type="image/jpeg")

# =============================================
# Camera Management API
# =============================================

@router.get("/cameras")
async def get_cameras(app: CCTVApplication = Depends(get_app)):
    """Get the list of all configured cameras from config.yaml."""
    config_path = Path(app.config_dir) / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        cameras = config_data.get("cameras", [])
        return {"cameras": cameras}
    except Exception as e:
        logger.error("Failed to read config: %s", e)
        return {"cameras": [], "error": str(e)}


@router.post("/cameras")
async def add_camera(req: CameraAddRequest, app: CCTVApplication = Depends(get_app)):
    """Add a new camera to config.yaml (requires restart to take effect)."""
    config_path = Path(app.config_dir) / "config" / "config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        cameras = config_data.get("cameras", [])

        # Check for duplicate camera_id
        for cam in cameras:
            if cam["camera_id"] == req.camera_id:
                raise HTTPException(status_code=400, detail=f"Camera ID '{req.camera_id}' already exists")

        # Add new camera
        new_cam = {
            "camera_id": req.camera_id,
            "name": req.name,
            "rtsp_url": req.rtsp_url,
            "enabled": req.enabled,
            "active_hours": None,
            "roi_zones": {
                "restricted_zone": {
                    "points": [[100, 200], [400, 200], [400, 500], [100, 500]]
                }
            }
        }
        cameras.append(new_cam)
        config_data["cameras"] = cameras

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        return {"status": "ok", "message": f"Camera '{req.name}' added. Restart the application to activate."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add camera: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/cameras/{camera_id}")
async def update_camera(camera_id: str, req: CameraAddRequest, app: CCTVApplication = Depends(get_app)):
    """Update an existing camera's RTSP URL or name in config.yaml."""
    config_path = Path(app.config_dir) / "config" / "config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        cameras = config_data.get("cameras", [])
        found = False
        for cam in cameras:
            if cam["camera_id"] == camera_id:
                cam["name"] = req.name
                cam["rtsp_url"] = req.rtsp_url
                cam["enabled"] = req.enabled
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

        config_data["cameras"] = cameras
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        return {"status": "ok", "message": f"Camera '{camera_id}' updated. Restart the application to apply changes."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str, app: CCTVApplication = Depends(get_app)):
    """Remove a camera from config.yaml."""
    config_path = Path(app.config_dir) / "config" / "config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        cameras = config_data.get("cameras", [])
        new_cameras = [c for c in cameras if c["camera_id"] != camera_id]

        if len(new_cameras) == len(cameras):
            raise HTTPException(status_code=404, detail="Camera not found")

        config_data["cameras"] = new_cameras
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        return {"status": "ok", "message": f"Camera '{camera_id}' deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# Telegram Configuration API
# =============================================

@router.get("/telegram")
async def get_telegram_config(app: CCTVApplication = Depends(get_app)):
    """Get the current Telegram configuration (token masked)."""
    env_path = Path(app.config_dir) / ".env"
    bot_token = ""
    chat_id = ""

    try:
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN="):
                        bot_token = line.split("=", 1)[1]
                    elif line.startswith("TELEGRAM_CHAT_ID="):
                        chat_id = line.split("=", 1)[1]
    except Exception as e:
        logger.error("Failed to read .env: %s", e)

    # Mask token for display
    masked_token = ""
    if bot_token and len(bot_token) > 10:
        masked_token = bot_token[:6] + "..." + bot_token[-4:]

    is_active = bool(app.telegram and app.telegram._enabled)
    stats = app.telegram.get_stats() if app.telegram else {}

    return {
        "bot_token_masked": masked_token,
        "bot_token_full": bot_token,
        "chat_id": chat_id,
        "is_active": is_active,
        "stats": stats,
    }


@router.post("/telegram")
async def save_telegram_config(req: TelegramConfigRequest, app: CCTVApplication = Depends(get_app)):
    """Save Telegram configuration to .env file."""
    env_path = Path(app.config_dir) / ".env"

    try:
        lines = []
        if env_path.exists():
            with open(env_path, "r") as f:
                lines = f.readlines()

        # Update or add token and chat_id
        token_found = False
        chat_found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("TELEGRAM_BOT_TOKEN="):
                new_lines.append(f"TELEGRAM_BOT_TOKEN={req.bot_token}\n")
                token_found = True
            elif line.strip().startswith("TELEGRAM_CHAT_ID="):
                new_lines.append(f"TELEGRAM_CHAT_ID={req.chat_id}\n")
                chat_found = True
            else:
                new_lines.append(line)

        if not token_found:
            new_lines.append(f"TELEGRAM_BOT_TOKEN={req.bot_token}\n")
        if not chat_found:
            new_lines.append(f"TELEGRAM_CHAT_ID={req.chat_id}\n")

        with open(env_path, "w") as f:
            f.writelines(new_lines)

        return {"status": "ok", "message": "Telegram configuration saved. Restart the application to apply."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telegram/test")
async def test_telegram(req: TelegramTestRequest, app: CCTVApplication = Depends(get_app)):
    """Send a test message to verify Telegram config works."""
    try:
        from telegram import Bot

        bot = Bot(token=req.bot_token)
        me = await bot.get_me()

        await bot.send_message(
            chat_id=req.chat_id,
            text=f"✅ *CCTV AI \\- Test Notification*\n\n{req.message}\n\nBot: @{me.username}",
            parse_mode="MarkdownV2",
        )

        return {"status": "ok", "bot_username": me.username, "message": "Test message sent successfully!"}
    except ImportError:
        raise HTTPException(status_code=500, detail="python-telegram-bot not installed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Telegram error: {str(e)}")

# =============================================
# Dashboard Stats & Recordings API
# =============================================

@router.get("/stats/today")
async def get_stats_today(app: CCTVApplication = Depends(get_app)):
    """Get today's statistics."""
    try:
        stats = await app.db.get_today_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get today stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recordings/{event_id}")
async def get_recording(event_id: str, app: CCTVApplication = Depends(get_app)):
    """Serve the MP4 video recording for an event."""
    recording_dir = Path("data/recordings")
    
    if not recording_dir.exists():
        raise HTTPException(status_code=404, detail="Recording directory not found")
        
    # Search for a file with the event_id
    for file in recording_dir.glob(f"*{event_id}*.mp4"):
        return FileResponse(
            path=file,
            media_type="video/mp4",
            filename=file.name
        )
        
    raise HTTPException(status_code=404, detail=f"Recording for event {event_id} not found")
