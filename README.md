# 🖐️ Hand-Sign Detection Demo

A compact Hand-Sign detection demo using **real-time hand-landmark detection and gesture recognition** via [MediaPipe Hands](https://developers.google.com/mediapipe/).
It ships both a **Flask REST API** (for easy integration) and **desktop demo scripts** you can run with a webcam or a folder of test images.

## Features

| Module | What it does |
|--------|--------------|
| `app.py` | Flask server exposing **`/predict`** – send a base-64 image, get back detected hand(s) and gesture label(s). |
| `hand_detection_live.py` | OpenCV webcam loop that draws landmarks & recognised gestures live. |
| `hand_detection_static_benchmark.py` | Batch-processes a folder of images, logs FPS & accuracy, writes annotated copies to `images/out/`. |
| `hand_sign_utils.py` | Helper functions – finger-count logic & heuristic gesture mapping. |
| `gesture_recognizer/` | Pre-trained **`.task`** model file + label map for the GestureRecognizer Task. |
| `images/` | Sample frames for quick testing. |
| `post.py` | Tiny client script that base-64 encodes an image and calls the API. |

---

## Quick start

```bash
# 📦 Install deps
pip install -r requirements.txt

# 🚀 1) Run the API
python app.py                      # listening on http://localhost:5000

# 🖼️ 2) Test with sample image
python post.py images/good_luck_sign/1524139491459.jpg