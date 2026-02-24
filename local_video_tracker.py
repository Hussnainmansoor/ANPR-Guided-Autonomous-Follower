#!/usr/bin/env python3

import cv2
import numpy as np
import math
import time
import csv
import os
from collections import deque

from SmartCarTracker import SmartCarTracker


class LocalVideoTracker:
    def __init__(self, video_path, output_path="output_tracked.mp4", plate_model_path=None):
        self.video_path = video_path
        self.output_path = output_path
        self.tracker = SmartCarTracker("yolov8n.pt", plate_model_path)

        self.prev_distance = None
        self.LEADER_STOP_EPS = 0.05         # slightly more tolerant to noise

        # ================= FOLLOWING PARAMETERS =================
        self.FOLLOW_DISTANCE = 5.0
        self.MAX_STEER = 14.0
        self.FOLLOW_SPEED = 2.0
        self.HFOV = math.radians(90.0)

        # ================= DELAYED STEERING =================
        self.steering_queue = deque()
        self.distance_accumulator = 0.0
        self.last_time = time.time()

        # ================= CSV =================
        self.csv_path = "steering_log.csv"
        self._init_csv()

        # ================= SIMULATED DEPTH =================
        self.baseline_bbox_width = None
        self.baseline_distance = 10.0

        # ================= SMOOTHING =================
        # EMA on distance — reduces bbox-width noise spikes
        self.smoothed_distance = None
        self.DIST_ALPHA = 0.15          # low = very smooth, high = reactive

        # EMA on raw steering angle — reduces jitter from bbox center wobble
        self.smoothed_leader_steer = 0.0
        self.STEER_ALPHA = 0.25

        # ================= STATE =================
        self.prev_gray = None
        self.last_delayed_steer = 0.0

    # ------------------------------------------------------------------
    def _init_csv(self):
        file_exists = os.path.isfile(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(["timestamp", "delayed_steering_deg", "estimated_distance_m"])

    def _log_csv(self, angle, distance):
        self.csv_writer.writerow([time.time(), angle, distance])
        self.csv_file.flush()

    # ------------------------------------------------------------------
    def estimate_distance(self, bbox):
        """Bbox-width inverse-proportion distance, then EMA smoothed."""
        x1, y1, x2, y2 = bbox
        bbox_width = max(1, x2 - x1)

        if self.baseline_bbox_width is None:
            self.baseline_bbox_width = bbox_width
            self.smoothed_distance = self.baseline_distance
            return self.baseline_distance

        raw = self.baseline_distance * (self.baseline_bbox_width / bbox_width)
        raw = max(1.0, min(50.0, raw))

        # EMA smoothing
        self.smoothed_distance = (
            self.DIST_ALPHA * raw
            + (1 - self.DIST_ALPHA) * self.smoothed_distance
        )
        return self.smoothed_distance

    # ------------------------------------------------------------------
    def _is_plate_locked(self):
        """
        Returns True only when ANPR has confirmed the target plate.
        If ANPR is disabled or no plate confirmed yet, returns False.
        """
        return (
            self.tracker.target_locked
            and self.tracker.best_plate is not None
            and len(self.tracker.best_plate) > 0
        )

    # ------------------------------------------------------------------
    def _draw_overlays(self, frame, distance_str, leader_steer, delayed_steer, queue_len, plate_locked):
        """
        Draw steering overlays.
        When plate not locked: steering shown as 0.00 (no signal being sent).
        """
        disp_leader  = leader_steer  if plate_locked else 0.0
        disp_delayed = delayed_steer if plate_locked else 0.0

        cv2.putText(frame, f"Distance: {distance_str}",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Leader steer: {disp_leader:+.2f} deg",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Delayed steer: {disp_delayed:+.2f} deg",
                    (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Queue: {queue_len} items",
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # ------------------------------------------------------------------
    def process_with_simulated_depth(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = self.tracker.process_frame(frame)

        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        plate_locked = self._is_plate_locked()

        # ============================================================
        # CASE A: No car detected in frame at all
        # → Zero steering, clear queue so no stale commands fire later
        # ============================================================
        if self.tracker.target_bbox is None:
            self.prev_gray = gray
            self.steering_queue.clear()
            self.distance_accumulator = 0.0
            self.last_delayed_steer = 0.0

            self._draw_overlays(
                annotated,
                distance_str="N/A",
                leader_steer=0.0,
                delayed_steer=0.0,
                queue_len=0,
                plate_locked=False
            )
            self._log_csv(0.0, -1.0)
            return annotated, 0.0, None

        # ============================================================
        # CASE B: Car detected but plate NOT yet confirmed
        # → Compute steering internally (for display) but output 0
        # ============================================================
        self.prev_gray = gray

        x1, y1, x2, y2 = self.tracker.target_bbox
        h, w = frame.shape[:2]
        cx = int((x1 + x2) / 2)

        distance = self.estimate_distance(self.tracker.target_bbox)

        if self.prev_distance is None:
            self.prev_distance = distance

        distance_delta = abs(distance - self.prev_distance)
        self.prev_distance = distance

        # Raw horizontal-offset steering
        dx = cx - (w / 2)
        angle_rad = (dx / (w / 2)) * (self.HFOV / 2)
        raw_steer = math.degrees(angle_rad)
        raw_steer = max(-self.MAX_STEER, min(self.MAX_STEER, raw_steer))

        # EMA smooth to kill per-frame jitter
        self.smoothed_leader_steer = (
            self.STEER_ALPHA * raw_steer
            + (1 - self.STEER_ALPHA) * self.smoothed_leader_steer
        )
        leader_steer = self.smoothed_leader_steer

        if not plate_locked:
            # Don't queue anything, output stays 0
            self.steering_queue.clear()
            self.distance_accumulator = 0.0
            self.last_delayed_steer = 0.0

            self._draw_overlays(
                annotated,
                distance_str=f"{distance:.2f}m",
                leader_steer=leader_steer,   # shown as 0 inside _draw_overlays
                delayed_steer=0.0,
                queue_len=0,
                plate_locked=False
            )
            self._log_csv(0.0, distance)
            return annotated, 0.0, distance

        # ============================================================
        # CASE C: Plate confirmed → full delayed steering active
        # ============================================================
        leader_moving = distance_delta > self.LEADER_STOP_EPS

        if leader_moving:
            d_travel = self.FOLLOW_SPEED * dt
            self.steering_queue.append((d_travel, leader_steer))
            self.distance_accumulator += d_travel

        delayed_steer = self.last_delayed_steer

        while self.distance_accumulator >= self.FOLLOW_DISTANCE and self.steering_queue:
            d_used, delayed_steer = self.steering_queue.popleft()
            self.distance_accumulator -= d_used

        self.last_delayed_steer = delayed_steer
        self._log_csv(delayed_steer, distance)

        self._draw_overlays(
            annotated,
            distance_str=f"{distance:.2f}m",
            leader_steer=leader_steer,
            delayed_steer=delayed_steer,
            queue_len=len(self.steering_queue),
            plate_locked=True
        )

        return annotated, delayed_steer, distance

    # ------------------------------------------------------------------
    def run(self):
        print("="*70)
        print("LOCAL VIDEO VEHICLE TRACKER WITH DELAYED STEERING")
        print("="*70)

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        print(f"📹 Input:        {self.video_path}")
        print(f"📹 Output:       {self.output_path}")
        print(f"📊 Resolution:   {w}x{h} @ {fps}fps")
        print(f"📊 Total frames: {total}")
        print(f"🖥️  Display: DISABLED (Headless Mode)")
        print("-"*70)

        start_time = time.time()
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, steer, dist = self.process_with_simulated_depth(frame)
                out.write(annotated)
                frame_idx += 1

                if frame_idx % 30 == 0:
                    progress = frame_idx / total * 100
                    elapsed = time.time() - start_time
                    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                    eta = (total - frame_idx) / avg_fps if avg_fps > 0 else 0
                    print(
                        f"Progress: {progress:.1f}% | FPS: {avg_fps:.1f} | "
                        f"ETA: {eta:.0f}s | Vehicles: {self.tracker.id_counter-1}",
                        end='\r'
                    )

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            cap.release()
            out.release()
            self.csv_file.close()

            total_time = time.time() - start_time
            print("\n" + "="*70)
            print("✅ PROCESSING COMPLETE")
            print("="*70)
            print(f"⏱️  Total time:      {total_time:.1f}s")
            print(f"📊 Average FPS:     {frame_idx/total_time:.1f}")
            print(f"🚗 Unique vehicles: {self.tracker.id_counter-1}")
            print(f"💾 Output video:    {self.output_path}")
            print(f"📄 Steering log:    {self.csv_path}")
            print("="*70)


def main():
    VIDEO_PATH       = "TEST_LONG_VID.mp4"
    OUTPUT_PATH      = "output_tracked1.mp4"
    PLATE_MODEL_PATH = "best_license_plate.pt"

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found: {VIDEO_PATH}")
        return

    if PLATE_MODEL_PATH and not os.path.exists(PLATE_MODEL_PATH):
        print(f"⚠️  Warning: Plate model not found: {PLATE_MODEL_PATH}")
        print("Continuing without ANPR...")
        PLATE_MODEL_PATH = None

    tracker = LocalVideoTracker(VIDEO_PATH, OUTPUT_PATH, PLATE_MODEL_PATH)
    tracker.run()


if __name__ == "__main__":
    main()