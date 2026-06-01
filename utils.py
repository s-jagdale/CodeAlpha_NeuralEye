"""
utils.py
--------
Utility functions shared across the project:
  - Colour palette generation (consistent per class / track ID)
  - Bounding-box and label drawing helpers
  - FPS counter
  - Video writer factory
"""

import time
import colorsys
import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  Colour Utilities
# ══════════════════════════════════════════════════════════════════════

def generate_color_palette(n: int = 80) -> list[tuple[int, int, int]]:
    """
    Generate *n* visually distinct BGR colours using the HSV colour wheel.

    Spreads hues evenly around the wheel so that adjacent IDs have
    clearly different colours even on a busy scene.

    Args:
        n: Number of colours to generate.

    Returns:
        List of (B, G, R) tuples with values in [0, 255].
    """
    palette: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = i / n                      # 0.0 – 1.0
        saturation = 0.85
        value = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert 0-1 floats → 0-255 ints, and swap to BGR for OpenCV
        palette.append((int(b * 255), int(g * 255), int(r * 255)))
    return palette


# Pre-built palette with 100 colours (covers COCO's 80 classes + extras)
_PALETTE = generate_color_palette(100)


def get_color(index: int) -> tuple[int, int, int]:
    """
    Return a deterministic BGR colour for *index* (class ID or track ID).

    The modulo wraps around so any integer is valid.
    """
    return _PALETTE[int(index) % len(_PALETTE)]


# ══════════════════════════════════════════════════════════════════════
#  Drawing Helpers
# ══════════════════════════════════════════════════════════════════════

def draw_tracked_object(
    frame: np.ndarray,
    track: dict,
    show_conf: bool = True,
) -> None:
    """
    Draw a bounding box and an info label for one tracked object.

    The colour is chosen by *track_id* so the same object always gets
    the same colour, regardless of class.

    Label format:  ID:<id>  <class_name>  [<confidence>%]

    Args:
        frame     : BGR frame to draw on (mutated in-place).
        track     : Dict with keys 'track_id', 'bbox', 'class_name',
                    'confidence'.
        show_conf : Whether to append the confidence percentage.
    """
    x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
    track_id   = track["track_id"]
    class_name = track["class_name"]
    conf       = track["confidence"]

    color = get_color(track_id)

    # ── Bounding box ──────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    # ── Label text ───────────────────────────────────────────────────
    label = f"ID:{track_id} {class_name}"
    if show_conf:
        label += f" {conf * 100:.1f}%"

    # Choose a font scale that is readable but not oversized
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    # Position the label just above the bounding box
    label_y = max(y1 - 4, text_h + 4)

    # Filled rectangle as label background (semi-readable against any scene)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline),
        (x1 + text_w, label_y + baseline),
        color,
        thickness=cv2.FILLED,
    )

    # White text on coloured background
    cv2.putText(
        frame,
        label,
        (x1, label_y - baseline // 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Render the current FPS counter in the top-left corner.

    Args:
        frame : BGR frame to draw on (mutated in-place).
        fps   : Current frames-per-second value.
    """
    label  = f"FPS: {fps:.1f}"
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.70
    thick  = 2
    color  = (0, 255, 0)   # bright green

    # Thin dark outline for readability on any background
    cv2.putText(
        frame, label, (10, 30),
        font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, label, (10, 30),
        font, scale, color, thick, cv2.LINE_AA,
    )


def draw_track_count(frame: np.ndarray, count: int) -> None:
    """
    Render the number of active tracks in the top-right corner.

    Args:
        frame : BGR frame (mutated in-place).
        count : Number of confirmed tracks in the current frame.
    """
    label = f"Objects: {count}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 2

    (text_w, _), _ = cv2.getTextSize(label, font, scale, thick)
    x = frame.shape[1] - text_w - 10

    cv2.putText(frame, label, (x, 30), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x, 30), font, scale, (0, 200, 255), thick, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════
#  FPS Counter
# ══════════════════════════════════════════════════════════════════════

class FPSCounter:
    """
    Computes a rolling-average FPS using a sliding window of timestamps.

    A rolling average (rather than an instantaneous frame-time) avoids
    jittery readings caused by occasional slow frames.

    Attributes:
        window_size : Number of recent frames used in the average.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._timestamps: list[float] = []

    def tick(self) -> float:
        """
        Record the current timestamp and return the updated FPS.

        Returns:
            Current rolling-average FPS (0.0 until the window is full).
        """
        now = time.perf_counter()
        self._timestamps.append(now)

        # Keep only the last `window_size` entries
        if len(self._timestamps) > self.window_size:
            self._timestamps.pop(0)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed


# ══════════════════════════════════════════════════════════════════════
#  Video Writer
# ══════════════════════════════════════════════════════════════════════

def create_video_writer(
    output_path: str,
    frame_width: int,
    frame_height: int,
    fps: float = 30.0,
) -> cv2.VideoWriter:
    """
    Create a VideoWriter that saves frames to *output_path*.

    Uses the MP4V codec (widely supported); the output file should have
    a .mp4 extension.

    Args:
        output_path  : Destination file path (e.g. "output/result.mp4").
        frame_width  : Width of each frame in pixels.
        frame_height : Height of each frame in pixels.
        fps          : Target frames-per-second for the saved video.

    Returns:
        An initialised cv2.VideoWriter ready to receive frames.

    Raises:
        IOError: If the writer cannot be opened (e.g. bad path / codec).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_width, frame_height)
    )

    if not writer.isOpened():
        raise IOError(
            f"[VideoWriter] Could not open '{output_path}'. "
            "Check that the directory exists and the codec is available."
        )

    print(
        f"[VideoWriter] Saving to '{output_path}' "
        f"({frame_width}x{frame_height} @ {fps:.1f} fps)."
    )
    return writer
