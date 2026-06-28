# 😴 Driver Drowsiness Detection (EAR + MAR)

### *Your eyes and your yawns tell the story before an accident does.*

A real-time computer vision system that detects driver drowsiness and fatigue using two well-established facial metrics: **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** — calculated from 68-point facial landmarks via `dlib`.

🎥 *Demo output video(s) included in this repo*

---

## 🧠 The Core Idea

```
  Eyes closed for 20+ frames straight?  ──►  🔴 DROWSY!
  Mouth open wide for 15+ frames straight? ──►  🟠 FATIGUE (YAWNING)
```

A single closed-eye frame means nothing — it could be a blink. The "consecutive frames" requirement is what separates a real drowsiness signal from normal blinking and talking.

---

## ✨ Key Features

- 👁️ **Eye Aspect Ratio (EAR)** — tracks how open/closed the eyes are, frame by frame
- 👄 **Mouth Aspect Ratio (MAR)** — detects sustained yawning as a fatigue signal
- 🎯 **68-point facial landmarks** via dlib's pretrained shape predictor
- ⏱️ **Frame-persistence logic** — avoids false alarms from a single blink or quick mouth movement
- 📊 **Live on-screen readout** — real-time EAR/MAR values overlaid on video

---

## 🛠️ Built With

`OpenCV` · `dlib` (face detector + 68-point shape predictor) · Standard EAR/MAR formulas ([Soukupová & Čech, 2016](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf))

---

## 🚀 Setup & Run

### 1. Clone and install
```bash
git clone https://github.com/afrosejamal/driver-drowsiness-ear-mar-detection.git
cd driver-drowsiness-ear-mar-detection
pip install -r requirements.txt
```

> **Note on `dlib`:** installing dlib on Windows can fail without CMake and a C++ build tool installed first. If `pip install dlib` errors out, install [CMake](https://cmake.org/download/) and Visual Studio Build Tools (C++ workload) first, then retry.

### 2. Download the facial landmark model
This repo doesn't include the landmark model (it's a public file, not something to duplicate in every repo). Download `shape_predictor_68_face_landmarks.dat` from the [dlib-models repository](https://github.com/davisking/dlib-models), unzip it, and place it in the project root.

### 3. Run it
```bash
python drowsiness_detector.py
```
By default this uses your webcam (`VIDEO_SOURCE = 0`). To test on a video file instead, change that line to a file path, e.g. `VIDEO_SOURCE = "input_videos/test1.mp4"`.

Press `q` to quit.

---

## ⚙️ How It Works

1. `dlib`'s frontal face detector locates a face in each frame
2. The 68-point shape predictor maps facial landmarks onto that face
3. **EAR** is computed from 6 eye landmarks per eye — the ratio of vertical eye-opening distance to horizontal eye-width. It drops sharply when eyes close.
4. **MAR** is computed similarly from inner-mouth landmarks — rises sharply during a yawn
5. Both values are tracked frame-to-frame; an alert only fires once the threshold is crossed for a sustained number of consecutive frames (20 for EAR, 15 for MAR), filtering out normal blinks and speech

---

## ⚠️ Limitations

- Requires a clear, front-facing view of the face — performance drops with poor lighting, glasses glare, or extreme head angles
- Threshold values (`EAR_THRESHOLD`, `MAR_THRESHOLD`) are fixed constants — not calibrated per-individual, so sensitivity may vary slightly person to person
- dlib's HOG-based face detector is CPU-friendly but slower than deep-learning detectors on lower-end hardware

## 🔮 Roadmap

- [ ] Per-user threshold calibration step at startup
- [ ] Add audio alert alongside the visual warning
- [ ] Compare against a YOLOv8-based detection approach for accuracy/speed tradeoffs

---

## 👤 Author

**Afrose Fathima J**
📧 afrosepvt@gmail.com · 🔗 [LinkedIn](http://www.linkedin.com/in/afrose-fathima-jamal-492b57291)
