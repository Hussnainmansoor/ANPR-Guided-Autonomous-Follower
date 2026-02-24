# 🚗 ANPR-Guided-Autonomous-Follower

> A computer vision pipeline that detects, identifies, and autonomously follows a specific target vehicle using license plate recognition and delayed steering logic.

---

## 📌 Overview

**ANPR-Guided-Autonomous-Follower** is a monocular vision-based autonomous convoy system built in Python. It combines YOLOv8 vehicle detection, DeepSort multi-object tracking, and EasyOCR-powered Automatic Number Plate Recognition (ANPR) to lock onto a specific target vehicle and replicate its steering path using a time-delayed following model.

The system is split into two tightly coupled modules:

- **`SmartCarTracker.py`** — handles detection, tracking, Kalman filtering, and ANPR-based plate locking
- **`LocalVideoTracker.py`** — handles distance estimation, delayed steering logic, CSV logging, and video output

---

## 🧠 How It Works

```
Video Input
    │
    ▼
YOLOv8 Detection (cars, buses, trucks)
    │
    ▼
DeepSort Multi-Object Tracking
    │
    ▼
Kalman Filter (2D motion smoothing)
    │
    ▼
ANPR Verification (EasyOCR + plate model)
    │
    ├── Plate NOT confirmed → Steering output = 0°
    │
    └── Plate confirmed → Delayed Steering Queue → Steering Output
```

### 🔐 Plate-Lock Logic

Steering is **only activated** once the target plate (`USER_TARGET_PLATE`) is:
1. Detected by the plate model with sufficient confidence
2. Read by OCR and matched with ≥ 0.7 similarity
3. Confirmed stable across `PLATE_STABLE_FRAMES` consecutive frames

Until confirmation, the follower outputs `0°` steering — preventing accidental movement.

### ⏱️ Delayed Steering

Rather than steering toward the leader's *current* position, the follower queues steering commands and replays them after traveling a defined `FOLLOW_DISTANCE`. This causes the follower to trace the **exact same physical path** as the leader.

$$\text{Delay Time} \approx \frac{\text{FOLLOW\_DISTANCE}}{\text{FOLLOW\_SPEED}}$$

> Default: 5 m ÷ 2 m/s = **2.5 s delay**

---

## 📁 Project Structure

```
ANPR-Guided-Autonomous-Follower/
│
├── SmartCarTracker.py       # Detection, tracking, ANPR, Kalman filtering
├── LocalVideoTracker.py     # Distance estimation, delayed steering, CSV logging
│
├── yolov8n.pt               # YOLOv8 nano model (auto-downloaded)
├── best_license_plate.pt    # Custom plate detection model (optional)
│
├── TEST_LONG_VID.mp4        # Input video (required)
├── output_tracked1.mp4      # Annotated output video (generated)
└── steering_log.csv         # Steering data log (generated)
```

---

## ⚙️ Technologies Used

| Category      | Library / Tool                          |
|---------------|-----------------------------------------|
| Language      | Python 3.x                              |
| Detection     | Ultralytics YOLOv8                      |
| Tracking      | DeepSort Realtime                       |
| OCR           | EasyOCR                                 |
| Vision        | OpenCV                                  |
| Math          | NumPy, Kalman Filtering (custom 2D)     |
| Depth         | Monocular bounding box width scaling    |
| Smoothing     | EMA (distance & steering), Kalman       |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python deep-sort-realtime easyocr numpy
```

### 2. Prepare Files

| File | Required | Notes |
|------|----------|-------|
| `TEST_LONG_VID.mp4` | ✅ Yes | Input video source |
| `best_license_plate.pt` | ⚠️ Optional | Custom YOLO plate model; ANPR disabled without it |
| `yolov8n.pt` | Auto | Downloaded automatically by Ultralytics |

### 3. Set Your Target Plate

In `SmartCarTracker.py`, update:

```python
self.USER_TARGET_PLATE = "NCAI001"  # ← change to your target plate
```

### 4. Run

```bash
python LocalVideoTracker.py
```

---

## 📊 Output

| File | Description |
|------|-------------|
| `output_tracked1.mp4` | Annotated video with bounding boxes, trajectories, and steering overlays |
| `steering_log.csv` | Timestamped log of delayed steering angle and estimated distance |

### On-Screen Overlays

- **Distance** — estimated meters to leader
- **Leader steer** — current computed angle from leader position
- **Delayed steer** — angle being executed after follow delay
- **Queue** — number of pending steering commands
- **FPS / Active / Vehicles** — runtime stats

---

## 📌 Configuration Parameters

### `SmartCarTracker.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USER_TARGET_PLATE` | `"NCAI001"` | License plate string to lock onto |
| `PLATE_MATCH_THRESH` | `0.7` | Minimum OCR similarity to accept a match |
| `PLATE_STABLE_FRAMES` | `3` | Frames required to confirm plate lock |
| `OCR_EVERY_N_FRAMES` | `3` | How often OCR runs (reduces CPU load) |
| `MIN_STABILITY` | `3` | Frames before a track is considered stable |
| `MAX_TARGET_LOST` | `60` | Frames before target lock is dropped |

### `LocalVideoTracker.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FOLLOW_DISTANCE` | `5.0 m` | Distance to maintain behind leader |
| `FOLLOW_SPEED` | `2.0 m/s` | Assumed follower speed for delay calculation |
| `MAX_STEER` | `14.0°` | Maximum allowable steering angle |
| `HFOV` | `90°` | Camera horizontal field of view |
| `DIST_ALPHA` | `0.15` | EMA smoothing factor for distance |
| `STEER_ALPHA` | `0.25` | EMA smoothing factor for steering angle |
| `LEADER_STOP_EPS` | `0.05 m` | Threshold below which leader is considered stopped |

---

## 🔬 Limitations

- Monocular distance estimation is approximate — no true depth sensor
- Steering only — no throttle or braking control
- Requires an unobstructed view of the license plate during initialization
- OCR accuracy depends on plate visibility, lighting, and resolution

---

## 🗺️ Future Roadmap

- [ ] Pure Pursuit controller integration
- [ ] PID-based throttle and braking control
- [ ] Monocular Depth Network (e.g., MiDaS) for improved distance estimation
- [ ] ROS integration for real-world vehicle deployment
- [ ] Multi-follower convoy support

---

## 📄 License

This project is intended for **educational and research purposes** only.

## 🎥 Demo Output

📹 [Download output_tracked1.mp4](https://github.com/YOUR_USERNAME/ANPR-Guided-Autonomous-Follower/releases/tag/v1.0)