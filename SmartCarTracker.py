import cv2
import numpy as np
import time
import math
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
import easyocr
import re


# =========================
#  Kalman Filter (2D)
# =========================
class KalmanFilter2D:
    def __init__(self):
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4)
        self.Q[0:2, 0:2] *= 0.05
        self.Q[2:4, 2:4] *= 2.0
        self.R = np.eye(2) * 15
        self.initialized = False
        self.target_distance = None

    def update(self, measurement):
        if not self.initialized:
            self.state = np.array([measurement[0], measurement[1], 0, 0])
            self.initialized = True
            return

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        z = np.array(measurement)
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def predict(self, dt=1.0):
        if not self.initialized:
            return

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 0.95, 0],
            [0, 0, 0, 0.95]
        ])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def speed(self):
        return np.linalg.norm(self.state[2:4])


# =========================
#  Smart Vehicle Tracker
# =========================
class SmartCarTracker:
    def __init__(self, model_path='yolov8n.pt', plate_model_path=None):
        self.model = YOLO(model_path)
        self.USER_TARGET_PLATE = "NCAI001"
        self.PLATE_MATCH_THRESH = 0.7
        self.PLATE_STABLE_FRAMES = 3
        self.OCR_EVERY_N_FRAMES = 10 
        self.plate_match_counter = defaultdict(int)

        self.tracker = DeepSort(
            max_age=15,
            n_init=8,
            max_iou_distance=0.3,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
        )

        self.frame_count = 0
        self.fps = 0
        self.id_counter = 1

        self.deepsort_to_pid = {}
        self.kalman = {}
        self.positions = defaultdict(deque)
        self.trajectories = defaultdict(deque)
        self.colors = {}

        self.MIN_STABILITY = 3
        self.stability = defaultdict(int)

        # Target locking
        self.target_vehicle_id = None
        self.target_locked = False
        self.target_lost_frames = 0
        self.MAX_TARGET_LOST = 60
        self.target_bbox = None

        # ANPR (optional)
        self.anpr_enabled = False
        self.plate_model = None
        self.ocr_reader = None
        
        if plate_model_path and os.path.exists(plate_model_path):
            try:
                self.plate_model = YOLO(plate_model_path)
                self.ocr_reader = easyocr.Reader(['en'], gpu=True)
                self.anpr_enabled = True
                print("✅ ANPR enabled")
            except Exception as e:
                print(f"⚠️ ANPR disabled: {e}")
        else:
            print("⚠️ ANPR disabled (no plate model found)")

        self.plate_history = deque(maxlen=5)
        self.best_plate = None
        self.best_plate_conf = 0.0

        self.PLATE_CONF_THRESH = 0.25
        self.OCR_EVERY_N_FRAMES = 3

    def _normalize_plate(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _is_valid_plate(self, text):
        if not text or len(text) < 5:
            return False
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        return letters >= 2 and digits >= 1

    def plate_similarity(self, a, b):
        from difflib import SequenceMatcher
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _run_anpr_on_target(self, frame, bbox):
        if not self.anpr_enabled:
            return None, 0.0, None
            
        x1, y1, x2, y2 = bbox
        vehicle_roi = frame[y1:y2, x1:x2]

        if vehicle_roi.size == 0:
            return None, 0.0, None

        results = self.plate_model(vehicle_roi, conf=0.25, verbose=False)
        print("PLATE MODEL FIRED")
        
        for r in results:
            for box in r.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])

                h, w = vehicle_roi.shape[:2]
                px1 = max(0, min(px1, w - 1))
                px2 = max(0, min(px2, w - 1))
                py1 = max(0, min(py1, h - 1))
                py2 = max(0, min(py2, h - 1))

                plate_roi = vehicle_roi[py1:py2, px1:px2]
                if plate_roi.size == 0:
                    continue

                ocr_results = self.ocr_reader.readtext(
                    plate_roi, detail=1, paragraph=False
                )

                for _, text, conf in ocr_results:
                    text = self._normalize_plate(text)

                    if not self._is_valid_plate(text):
                        continue

                    similarity = self.plate_similarity(text, self.USER_TARGET_PLATE)
                    print(f"OCR RAW: {text} | CONF: {conf:.2f} | SIM: {similarity:.2f}")

                    if similarity >= self.PLATE_MATCH_THRESH:
                        return text, conf, (px1, py1, px2, py2)

        return None, 0.0, None

    def _color(self, pid):
        if pid not in self.colors:
            rng = np.random.default_rng(pid)
            self.colors[pid] = tuple(map(int, rng.integers(50, 255, 3)))
        return self.colors[pid]

    def detect(self, frame):
        results = self.model(frame, classes=[2, 5, 7], conf=0.5,iou=0.5, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                if (x2-x1)*(y2-y1) > 1200:
                    detections.append([x1, y1, x2, y2, conf, int(box.cls[0])])

        return detections

    def process_frame(self, frame):
        start = time.time()

        detections = self.detect(frame)
        ds_dets = []

        for x1, y1, x2, y2, conf, cls in detections:
            ds_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = self.tracker.update_tracks(ds_dets, frame=frame)

        annotated = frame.copy()
        active = 0
        visible_pids = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            if tid not in self.deepsort_to_pid:
                pid = self.id_counter
                self.id_counter += 1
                self.deepsort_to_pid[tid] = pid
                self.kalman[pid] = KalmanFilter2D()
            else:
                pid = self.deepsort_to_pid[tid]

            self.stability[pid] += 1
            if self.stability[pid] < self.MIN_STABILITY:
                continue

            visible_pids.append(pid)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.kalman[pid].update(center)
            self.kalman[pid].predict()

            self.positions[pid].append(center)
            self.trajectories[pid].append(center)
            if len(self.trajectories[pid]) > 200:
                self.trajectories[pid].popleft()

            active += 1
            color = self._color(pid)
            base_y = y1 - 6
            line_gap = 14

            if self.target_vehicle_id is None:
                self.target_vehicle_id = pid
                self.target_locked = True
                self.target_bbox = [x1, y1, x2, y2]

            is_target = (pid == self.target_vehicle_id)

            if self.anpr_enabled and self.frame_count % self.OCR_EVERY_N_FRAMES == 0:
                plate, conf, plate_bbox = self._run_anpr_on_target(frame, [x1, y1, x2, y2])

                if plate:
                    self.plate_match_counter[pid] += 1

                    if self.plate_match_counter[pid] >= self.PLATE_STABLE_FRAMES:
                        self.target_vehicle_id = pid
                        self.target_locked = True
                        self.target_bbox = [x1, y1, x2, y2]
                        self.best_plate = plate
                        self.best_plate_conf = conf
                        self.target_lost_frames = 0
                        is_target = True

            if is_target:
                self.target_bbox = [x1, y1, x2, y2]
                self.target_lost_frames = 0

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated,
                    f"TARGET PLATE: {self.best_plate}",
                    (x1, base_y - line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2
                )

                if self.best_plate:
                    cv2.putText(
                        annotated,
                        f"{self.best_plate} ({self.best_plate_conf:.0%})",
                        (x1, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        2
                    )

                pts = list(self.trajectories[pid])
                for i in range(len(pts) - 1):
                    cv2.line(
                        annotated,
                        (int(pts[i][0]), int(pts[i][1])),
                        (int(pts[i + 1][0]), int(pts[i + 1][1])),
                        (0, 0, 255),
                        3
                    )

        if self.target_vehicle_id is not None:
            if self.target_vehicle_id not in visible_pids:
                print("❌ Target left frame — removing immediately")

                self.target_vehicle_id = None
                self.target_locked = False
                self.target_bbox = None
                self.target_lost_frames = 0

                self.best_plate = None
                self.best_plate_conf = 0.0
                self.plate_history.clear()
                self.plate_match_counter.clear()

        self.frame_count += 1
        elapsed = time.time() - start
        self.fps = 1 / elapsed if elapsed > 0 else 0

        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Active: {active}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Vehicles: {self.id_counter - 1}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated