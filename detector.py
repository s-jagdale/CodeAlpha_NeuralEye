"""
detector.py
-----------
Handles all YOLOv8-based object detection logic.
Wraps the Ultralytics YOLO model and provides a clean interface
for detecting objects in individual video frames.
"""

import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """
    Wraps YOLOv8 for frame-level object detection.

    Attributes:
        model       : Loaded YOLO model instance.
        conf_thresh : Minimum confidence score to keep a detection.
        iou_thresh  : IoU threshold used during Non-Maximum Suppression.
        device      : Compute device ('cpu', 'cuda', 'mps', …).
        class_names : Dict mapping class-index → class-name string.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.45,
        device: str = "cpu",
    ):
        """
        Load the YOLOv8 model from *model_path*.

        Args:
            model_path  : Path to a .pt weights file or a model name
                          recognised by Ultralytics (auto-downloaded).
            conf_thresh : Detections below this score are discarded.
            iou_thresh  : NMS IoU threshold (lower → fewer overlapping boxes).
            device      : Torch device string.

        Raises:
            RuntimeError: If the model file cannot be loaded.
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            # Cache class names for fast look-up later
            self.class_names: dict[int, str] = self.model.names
            print(f"[Detector] Model '{model_path}' loaded on device '{device}'.")
            print(f"[Detector] Classes available: {len(self.class_names)}")
        except Exception as exc:
            raise RuntimeError(
                f"[Detector] Failed to load model '{model_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame and return detection records.

        Args:
            frame: A HxWx3 NumPy array in BGR format (as returned by OpenCV).

        Returns:
            A list of dicts, one per detection, each containing:
                - "bbox"       : [x1, y1, x2, y2]  (pixel coordinates, floats)
                - "confidence" : float in [0, 1]
                - "class_id"   : int
                - "class_name" : str
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLOv8 inference (verbose=False suppresses per-frame console spam)
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []

        # `results` is always a list; we only submitted one frame
        for result in results:
            boxes = result.boxes  # Boxes object from Ultralytics

            if boxes is None:
                continue

            for box in boxes:
                # xyxy returns a 2-D tensor: shape (1, 4) → flatten to 1-D
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, "unknown")

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )

        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def warmup(self, img_size: tuple[int, int] = (640, 480)) -> None:
        """
        Send one dummy frame through the model so the first real frame
        does not include JIT / CUDA initialisation overhead.

        Args:
            img_size: (width, height) of the dummy frame.
        """
        dummy = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        self.detect(dummy)
        print("[Detector] Warm-up complete.")
