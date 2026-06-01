"""
tracker.py
----------
Wraps the Deep SORT algorithm (via deep-sort-realtime) to provide
persistent, unique tracking IDs across video frames.

Deep SORT extends the classic SORT tracker with a deep appearance
descriptor, which dramatically reduces ID-switches when objects
temporarily overlap or leave/re-enter the frame.
"""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class MultiObjectTracker:
    """
    Maintains object identities across frames using Deep SORT.

    Deep SORT input expects detections in the format:
        ([left, top, width, height], confidence, class_name)

    The tracker returns Track objects that carry a stable `track_id`.

    Attributes:
        tracker        : Underlying DeepSort instance.
        max_age        : Frames a track survives without a matching detection.
        min_hits       : Minimum detections before a track is confirmed.
        iou_threshold  : Minimum IoU for associating a detection to a track.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        embedder: str = "mobilenet",
        half: bool = False,
        bgr: bool = True,
    ):
        """
        Initialise the Deep SORT tracker.

        Args:
            max_age       : How many consecutive missed frames before a track
                            is deleted (higher → tracks survive occlusions longer).
            min_hits      : Frames needed to confirm a new track (higher →
                            fewer false-positive tracks).
            iou_threshold : IoU overlap required to link a detection to an
                            existing track.
            embedder      : CNN used for appearance features.
                            Options: 'mobilenet', 'torchreid', 'clip_RN50', …
                            'mobilenet' is fast and works well for most scenes.
            half          : Use FP16 for the embedder (GPU only).
            bgr           : True if frames are BGR (OpenCV default).
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracker = DeepSort(
            max_age=max_age,
            nms_max_overlap=1.0,          # rely on YOLO's NMS instead
            max_cosine_distance=0.4,      # appearance similarity threshold
            nn_budget=None,               # no limit on appearance gallery
            embedder=embedder,
            half=half,
            bgr=bgr,
            embedder_gpu=False,           # keep on CPU by default
        )
        print(
            f"[Tracker] Deep SORT initialised "
            f"(max_age={max_age}, min_hits={min_hits}, embedder='{embedder}')."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: list[dict], frame: np.ndarray
    ) -> list[dict]:
        """
        Feed new detections into the tracker and get updated track list.

        Args:
            detections : List of detection dicts from ObjectDetector.detect().
                         Each dict must have keys: 'bbox', 'confidence',
                         'class_name'.
            frame      : The current BGR frame (used by the embedder to
                         extract appearance features).

        Returns:
            A list of track dicts, one per *confirmed* active track:
                - "track_id"   : int  – stable unique ID
                - "bbox"       : [x1, y1, x2, y2]  (pixel coordinates)
                - "class_name" : str
                - "confidence" : float  (from the matched detection)
        """
        if frame is None or frame.size == 0:
            return []

        # ── Convert detections to Deep SORT input format ──────────────
        # DeepSort expects: [([l, t, w, h], confidence, class_name), …]
        ds_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            ltwh = [x1, y1, w, h]
            ds_inputs.append((ltwh, det["confidence"], det["class_name"]))

        # ── Run tracker update ─────────────────────────────────────────
        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)

        # ── Convert track output back to our dict format ───────────────
        active_tracks: list[dict] = []
        for track in tracks:
            # Skip tentative / deleted tracks
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()            # [x1, y1, x2, y2]
            class_name = track.get_det_class() or "unknown"
            confidence = track.get_det_conf() or 0.0

            # Occasionally get_det_conf returns None on pure-prediction steps
            if confidence is None:
                confidence = 0.0

            active_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": [float(v) for v in ltrb],
                    "class_name": class_name,
                    "confidence": float(confidence),
                }
            )

        return active_tracks
