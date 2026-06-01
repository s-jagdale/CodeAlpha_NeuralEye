# 🎯 YOLOv8 + Deep SORT — Real-Time Object Detection & Multi-Object Tracking

A production-ready Python pipeline that detects and tracks multiple objects in real time from a webcam or video file, using **YOLOv8** for detection and **Deep SORT** for persistent identity assignment.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Deep SORT](https://img.shields.io/badge/Tracker-Deep%20SORT-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Object Detection** | YOLOv8n/s/m/l/x — plug any Ultralytics model |
| **Multi-Object Tracking** | Deep SORT with MobileNet appearance embedder |
| **Unique Track IDs** | Stable IDs that survive brief occlusions |
| **Visual Overlays** | Colour-coded boxes, class names, confidence %, FPS, object count |
| **Video Output** | Saves annotated MP4 with original frame rate |
| **Flexible Source** | Webcam index or any video file |
| **Headless Mode** | Run on servers without a display |
| **Error Handling** | Graceful handling of missing models, bad sources, stream drops |

---

## 📁 Project Structure

```
yolo_deepsort/
├── main.py           # Entry point & main processing loop
├── detector.py       # YOLOv8 detection wrapper
├── tracker.py        # Deep SORT tracking wrapper
├── utils.py          # Drawing helpers, FPS counter, video writer
├── requirements.txt  # Python dependencies
├── output/           # Saved output videos (auto-created)
└── README.md
```

---

## 🚀 Installation

### 1 · Clone the repository

```bash
git clone https://github.com/your-username/yolo-deepsort.git
cd yolo-deepsort
```

### 2 · Create a virtual environment (recommended)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is listed in `requirements.txt` as a CPU build. For GPU acceleration (CUDA), install the appropriate PyTorch version first from https://pytorch.org before running `pip install -r requirements.txt`.

### 4 · (Optional) Download a YOLO model manually

```bash
# Ultralytics will auto-download on first run, but you can pre-fetch:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## 🎬 Usage

### Webcam (default)

```bash
python main.py
```

### Specific webcam index

```bash
python main.py --source 1
```

### Video file

```bash
python main.py --source path/to/video.mp4
```

### Full options

```
python main.py --help

optional arguments:
  --source      Video source: webcam index or file path   (default: 0)
  --model       YOLOv8 weights file                       (default: yolov8n.pt)
  --conf        Detection confidence threshold 0-1        (default: 0.40)
  --iou         NMS IoU threshold 0-1                     (default: 0.45)
  --device      Torch device: cpu | cuda | mps            (default: cpu)
  --output      Output video path                         (default: output/result.mp4)
  --max-age     Deep SORT track survival frames           (default: 30)
  --no-display  Headless mode – no preview window
  --no-save     Do not write an output video
```

### Examples

```bash
# Higher-accuracy model, GPU, custom output path
python main.py --model yolov8s.pt --device cuda --output output/street.mp4

# Process a CCTV clip headlessly on a server
python main.py --source cctv_footage.mp4 --no-display --conf 0.5

# Lightweight run with no video saved
python main.py --no-save
```

Press **`q`** or **`ESC`** in the preview window to quit at any time.

---

## 🧩 Module Reference

### `detector.py` — `ObjectDetector`

| Method | Description |
|---|---|
| `__init__(model_path, conf_thresh, iou_thresh, device)` | Load YOLOv8 model |
| `detect(frame)` | Run inference; return list of `{bbox, confidence, class_id, class_name}` |
| `warmup(img_size)` | Send a dummy frame to pre-compile CUDA/JIT kernels |

### `tracker.py` — `MultiObjectTracker`

| Method | Description |
|---|---|
| `__init__(max_age, min_hits, iou_threshold, embedder, …)` | Initialise Deep SORT |
| `update(detections, frame)` | Associate detections to existing tracks; return `{track_id, bbox, class_name, confidence}` per confirmed track |

### `utils.py`

| Symbol | Description |
|---|---|
| `generate_color_palette(n)` | Create *n* visually distinct BGR colours |
| `get_color(index)` | Deterministic colour for any integer ID |
| `draw_tracked_object(frame, track, show_conf)` | Draw box + label for one track |
| `draw_fps(frame, fps)` | Overlay FPS counter top-left |
| `draw_track_count(frame, count)` | Overlay object count top-right |
| `FPSCounter(window_size)` | Rolling-average FPS tracker |
| `create_video_writer(path, w, h, fps)` | Factory for `cv2.VideoWriter` |

### `main.py`

| Function | Description |
|---|---|
| `parse_args()` | CLI argument parser |
| `open_video_source(source)` | Open webcam or video file safely |
| `run(args)` | Full pipeline loop: read → detect → track → draw → save |

---

## ⚙️ Configuration Tips

| Goal | Flag |
|---|---|
| Better accuracy | `--model yolov8s.pt` or `yolov8m.pt` |
| Fewer false positives | Increase `--conf` (e.g. `0.55`) |
| Track through occlusions | Increase `--max-age` (e.g. `60`) |
| Faster inference | `--device cuda` or `--model yolov8n.pt` |
| Detect only people | Edit `detector.py` to filter `class_id == 0` |

---

## 🖥️ Sample Output

```
[Detector] Model 'yolov8n.pt' loaded on device 'cpu'.
[Detector] Classes available: 80
[Tracker]  Deep SORT initialised (max_age=30, min_hits=3, embedder='mobilenet').
[Main]     Opened webcam (index 0).
[Main]     Source resolution : 1280x720  @ 30.0 fps
[Detector] Warm-up complete.
[Main]     Pipeline running.  Press 'q' to quit.
[Main]     Output video saved to: output/result.mp4
[Main]     Done.
```

The annotated video shows each object with:
- A **colour-coded bounding box** (same colour = same identity)
- Label: `ID:3 person 87.4%`
- Live **FPS** (top-left) and **object count** (top-right)

---

## 🛠️ Troubleshooting

**`No module named 'deep_sort_realtime'`**
```bash
pip install deep-sort-realtime
```

**Webcam not opening**
- Check `--source 0` is the correct index; try `1` or `2`.
- On Linux, ensure your user is in the `video` group.

**CUDA out of memory**
- Use a smaller model: `--model yolov8n.pt`
- Reduce input resolution in `detector.py` (`imgsz` parameter).

**Very low FPS on CPU**
- YOLOv8n is the fastest model; try `--conf 0.5` to reduce NMS work.
- Consider running on a machine with a GPU.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥ 8.0 | YOLOv8 model + inference |
| `opencv-python` | ≥ 4.8 | Frame capture, drawing, I/O |
| `deep-sort-realtime` | ≥ 1.3 | Deep SORT tracker |
| `torch` / `torchvision` | ≥ 2.0 | Neural network backend |
| `numpy` | ≥ 1.24 | Array operations |

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Deep SORT Realtime](https://github.com/levan92/deep_sort_realtime)
- Original Deep SORT paper: *Wojke et al., 2017 – "Simple Online and Realtime Tracking with a Deep Association Metric"*
