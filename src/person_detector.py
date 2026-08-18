"""
person_detector.py — YOLO-based person & animal detection with device-adaptive inference.

Wraps the Ultralytics YOLO model. Detects persons AND animals (cats, dogs, birds)
but only persons trigger security alerts. Animals are tracked separately.
Supports CPU, CUDA, OpenVINO, and auto device selection.
"""

import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs
PERSON_CLASS_ID = 0
BIRD_CLASS_ID = 14
CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

# All classes we want to detect
DETECTABLE_CLASSES = [PERSON_CLASS_ID, BIRD_CLASS_ID, CAT_CLASS_ID, DOG_CLASS_ID]
ANIMAL_CLASS_IDS = {BIRD_CLASS_ID, CAT_CLASS_ID, DOG_CLASS_ID}

# Human-readable class names
CLASS_NAMES = {
    PERSON_CLASS_ID: "Human",
    BIRD_CLASS_ID: "Bird",
    CAT_CLASS_ID: "Cat",
    DOG_CLASS_ID: "Dog",
}


@dataclass
class DetectionResult:
    """Result of detection containing both person and animal detections."""
    persons: sv.Detections        # Only person detections (for tracking/alerts)
    animals: sv.Detections        # Only animal detections (for display, no alert)
    all_detections: sv.Detections # All detections combined
    person_keypoints: Optional[sv.KeyPoints] = None  # Keypoints for persons (if pose model used)
    person_count: int = 0
    animal_count: int = 0


class PersonDetector:
    """YOLOv8-based person detector.

    Loads a YOLO model, selects the optimal device (CPU/GPU),
    and returns only 'person' detections as supervision.Detections.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        device_mode: str = "auto",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        img_size: int = 640,
    ):
        """
        Args:
            model_name: YOLO model filename (auto-downloaded if not present).
            device_mode: "cpu", "cuda", or "auto" (auto-detect GPU).
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: NMS IoU threshold.
            img_size: Input image size for the model.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # Resolve device
        self.device = self._resolve_device(device_mode)

        # Check for OpenVINO optimization
        try:
            import openvino
            has_openvino = True
        except ImportError:
            has_openvino = False

        if has_openvino and model_name.endswith(".pt") and (self.device == "cpu" or self.device.startswith("intel:")):
            ov_model_name = model_name.replace(".pt", "_openvino_model")
            import os
            if not os.path.exists(ov_model_name):
                logger.info("Exporting YOLO model to OpenVINO for faster inference on Intel hardware...")
                temp_model = YOLO(model_name)
                temp_model.export(format="openvino")
            model_name = ov_model_name
            logger.info("Using OpenVINO optimized model: %s", model_name)

        # Load model
        logger.info("Loading YOLO model: %s on device: %s", model_name, self.device)
        self.model = YOLO(model_name)
        self._lock = asyncio.Lock()

        # Warmup inference (helps with first-frame latency)
        self._warmup()

        logger.info(
            "PersonDetector ready: model=%s, device=%s, conf=%.2f",
            model_name,
            self.device,
            confidence_threshold,
        )

    @staticmethod
    def _resolve_device(device_mode: str) -> str:
        """Resolve the compute device to use."""
        if device_mode == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info("CUDA available: %s", gpu_name)
                    return "cuda"
                else:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    return "cpu"
            except ImportError:
                logger.warning("PyTorch CUDA not available, falling back to CPU")
                return "cpu"

        elif device_mode.startswith("intel:"):
            return device_mode

        elif device_mode == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info("Auto-detected CUDA device: %s", gpu_name)
                    return "cuda"
            except ImportError:
                pass
            
            try:
                import openvino as ov
                core = ov.Core()
                if "GPU" in core.available_devices:
                    logger.info("Auto-detected OpenVINO GPU")
                    return "intel:gpu"
            except Exception:
                pass

            logger.info("Auto-detected device: CPU")
            return "cpu"

        return "cpu"

    def _warmup(self) -> None:
        """Run a dummy inference to warm up the model."""
        try:
            dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                device=self.device,
                conf=self.confidence_threshold,
                verbose=False,
            )
            logger.debug("Model warmup complete")
        except Exception as e:
            logger.warning("Model warmup failed (non-critical): %s", e)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run person-only detection on a frame (backward compatible).

        Args:
            frame: Input BGR frame.

        Returns:
            supervision.Detections containing only 'person' class detections.
        """
        result = self.detect_all(frame)
        return result.persons

    def detect_all(self, frame: np.ndarray) -> DetectionResult:
        """Run detection for persons AND animals on a frame.

        Args:
            frame: Input BGR frame.

        Returns:
            DetectionResult with separate person and animal detections.
        """
        # Run YOLO inference for all detectable classes
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            classes=DETECTABLE_CLASSES,
            verbose=False,
        )

        if not results or len(results) == 0:
            return DetectionResult(
                persons=sv.Detections.empty(),
                animals=sv.Detections.empty(),
                all_detections=sv.Detections.empty(),
            )

        # Convert to supervision Detections
        all_detections = sv.Detections.from_ultralytics(results[0])

        if len(all_detections) == 0 or all_detections.class_id is None:
            return DetectionResult(
                persons=sv.Detections.empty(),
                animals=sv.Detections.empty(),
                all_detections=sv.Detections.empty(),
            )

        # Split into persons and animals
        person_mask = all_detections.class_id == PERSON_CLASS_ID
        animal_mask = np.isin(all_detections.class_id, list(ANIMAL_CLASS_IDS))

        persons = all_detections[person_mask]
        animals = all_detections[animal_mask]

        # Extract keypoints if available (YOLO-Pose model)
        person_keypoints = None
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            # ultralytics keypoints
            try:
                # Get all keypoints, filter by person mask
                all_keypoints = sv.KeyPoints.from_ultralytics(results[0])
                if all_keypoints.xy is not None and len(all_keypoints.xy) == len(all_detections):
                    person_keypoints = all_keypoints[person_mask]
            except Exception as e:
                logger.warning(f"Failed to parse keypoints: {e}")

        return DetectionResult(
            persons=persons,
            animals=animals,
            all_detections=all_detections,
            person_keypoints=person_keypoints,
            person_count=len(persons),
            animal_count=len(animals),
        )

    async def detect_all_async(self, frame: np.ndarray) -> DetectionResult:
        """Run detection asynchronously without blocking the event loop.
        
        Uses an asyncio.Lock to ensure only one frame is passed to the YOLO model 
        at a time, preventing CPU/GPU thrashing and OOM errors in multi-camera setups.
        """
        async with self._lock:
            # Run the synchronous inference in a separate thread
            result = await asyncio.to_thread(self.detect_all, frame)
            return result

    @staticmethod
    def get_class_name(class_id: int) -> str:
        """Get human-readable name for a COCO class ID."""
        return CLASS_NAMES.get(class_id, f"class_{class_id}")

    def get_model_info(self) -> dict:
        """Get model information for logging/debugging."""
        return {
            "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "unknown",
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "img_size": self.img_size,
        }
