"""
main.py
-------
Entry point for the YOLOv8 + Deep SORT real-time object tracking pipeline.

Usage examples
--------------
# Webcam (default, index 0)
python main.py

# Specific webcam index
python main.py --source 1

# Video file
python main.py --source path/to/video.mp4

# Custom model & confidence threshold
python main.py --model yolov8s.pt --conf 0.5

# Disable live preview window (headless / server mode)
python main.py --source video.mp4 --no-display
"""

import argparse
import os
import sys
import cv2

from detector import ObjectDetector
from tracker import MultiObjectTracker
from utils import (
    FPSCounter,
    create_video_writer,
    draw_fps,
    draw_track_count,
    draw_tracked_object,
)


# ══════════════════════════════════════════════════════════════════════
#  CLI Argument Parsing
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 + Deep SORT – Real-time Object Detection & Tracking"
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Video source. Use an integer for webcam index (e.g. 0) "
            "or a file path for a video file. (default: 0)"
        ),
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model weights. Ultralytics will auto-download if absent. "
             "(default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold [0-1]. (default: 0.40)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold [0-1]. (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: 'cpu', 'cuda', 'cuda:0', 'mps' … (default: cpu)",
    )
    parser.add_argument(
        "--output",
        default="output/result.mp4",
        help="Path for the saved output video. (default: output/result.mp4)",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        dest="max_age",
        help="Deep SORT max frames a track survives without a match. "
             "(default: 30)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live preview window (headless/server mode).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write an output video file.",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  Video Source Helper
# ══════════════════════════════════════════════════════════════════════

def open_video_source(source: str) -> cv2.VideoCapture:
    """
    Open a webcam or video file as an OpenCV VideoCapture.

    Accepts either an integer string (webcam index) or a file path.

    Args:
        source: CLI --source value.

    Returns:
        An opened cv2.VideoCapture.

    Raises:
        IOError: If the source cannot be opened.
    """
    # Decide whether source is an integer (webcam) or a path (file)
    try:
        cam_index = int(source)
        cap = cv2.VideoCapture(cam_index)
        label = f"webcam (index {cam_index})"
    except ValueError:
        if not os.path.isfile(source):
            raise IOError(f"Video file not found: '{source}'")
        cap = cv2.VideoCapture(source)
        label = f"file '{source}'"

    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {label}")

    print(f"[Main] Opened {label}.")
    return cap


# ══════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    """
    Execute the full detection + tracking loop.

    Steps per frame
    ---------------
    1. Grab frame from VideoCapture.
    2. Run YOLOv8 detection  → list of raw detections.
    3. Update Deep SORT      → list of confirmed tracks (with stable IDs).
    4. Draw bounding boxes, labels, FPS, and track count onto the frame.
    5. Show live preview (optional).
    6. Write frame to output video (optional).
    7. Break on 'q' key-press or stream end.

    Args:
        args: Parsed CLI arguments.
    """

    # ── 1. Initialise detector ────────────────────────────────────────
    try:
        detector = ObjectDetector(
            model_path=args.model,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            device=args.device,
        )
    except RuntimeError as exc:
        print(f"[Main] ERROR – {exc}")
        sys.exit(1)

    # ── 2. Initialise tracker ─────────────────────────────────────────
    tracker = MultiObjectTracker(max_age=args.max_age)

    # ── 3. Open video source ──────────────────────────────────────────
    try:
        cap = open_video_source(args.source)
    except IOError as exc:
        print(f"[Main] ERROR – {exc}")
        sys.exit(1)

    # Read frame dimensions from the capture
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[Main] Source resolution : {frame_w}x{frame_h}  @ {src_fps:.1f} fps")

    # ── 4. Initialise video writer ────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if not args.no_save:
        # Ensure output directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            writer = create_video_writer(args.output, frame_w, frame_h, src_fps)
        except IOError as exc:
            print(f"[Main] WARNING – {exc}  (output will not be saved)")

    # ── 5. Model warm-up (eliminates first-frame latency spike) ───────
    detector.warmup(img_size=(frame_w, frame_h))

    # ── 6. FPS counter ────────────────────────────────────────────────
    fps_counter = FPSCounter(window_size=30)
    current_fps = 0.0

    print("\n[Main] Pipeline running.  Press 'q' to quit.\n")

    # ══════════════════════════════════════════════════════════════════
    #  Main Loop
    # ══════════════════════════════════════════════════════════════════
    while True:
        # ── Read frame ────────────────────────────────────────────────
        ret, frame = cap.read()

        if not ret:
            # End-of-file for a video file – graceful exit
            print("[Main] End of video stream or webcam disconnected.")
            break

        # ── Detect ────────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Track ─────────────────────────────────────────────────────
        tracks = tracker.update(detections, frame)

        # ── Draw each confirmed track ─────────────────────────────────
        for track in tracks:
            draw_tracked_object(frame, track, show_conf=True)

        current_fps = fps_counter.tick()
        draw_fps(frame, current_fps)
        draw_track_count(frame, len(tracks))

        if writer is not None:
            writer.write(frame)


        if not args.no_display:
            cv2.imshow("YOLOv8 + Deep SORT", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[Main] User requested quit.")
                break


    cap.release()
    if writer is not None:
        writer.release()
        print(f"[Main] Output video saved to: {args.output}")
    cv2.destroyAllWindows()
    print("[Main] Done.")


# ══════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run(parse_args())

