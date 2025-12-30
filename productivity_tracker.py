import time
import os
from collections import defaultdict
import numpy as np
import cv2
from datetime import datetime
import atexit
from utils.db_manager import DBManager


class ProductivityTracker:
    """
    Tracks walking, standing, sitting, hand movement, unknown pose,
    using REAL TIME (dt in seconds) instead of frame counting.
    """

    def __init__(self, output_folder, config, run_start_epoch, csv_name=None):

        self.camera_id = config.get("camera_id", "cam_1")

        # --------------------------------------------------
        # Process-level epoch (NEVER changes)
        # --------------------------------------------------
        self.run_start_epoch = run_start_epoch

        # --------------------------------------------------
        # Day-level tracking (CAN change at midnight)
        # --------------------------------------------------
        self.run_date = datetime.now().strftime("%d-%m-%Y")
        self.current_date_str = self.run_date
        self.current_day_start_epoch = self.run_start_epoch

        self.output_folder = output_folder
        self.config = config

        
        self.realtime_csv = config.get("realtime_csv", "false").lower() == "true"
        self.realtime_csv_interval = int(config.get("realtime_csv_interval_sec", 60))
        self._last_csv_write_time = 0.0

        # --------------------------------------------------
        # Time counters
        # --------------------------------------------------
        self.walking_time = defaultdict(float)
        self.hand_movement_time = defaultdict(float)
        self.sitting_time = defaultdict(float)
        self.sitting_hand_time = defaultdict(float)
        self.standing_time = defaultdict(float)
        self.unknown_time = defaultdict(float)

        self.id_to_name = {}
        self.prev_centers = {}
        self.prev_hands = {}
        self.pose_history = defaultdict(list)

        self.walk_threshold = 4
        self.hand_move_threshold = 3
        self.pose_window = 5

        self.total_video_seconds = 0.0
        self.elapsed_video_seconds = 0.0

        # -------------------------------
        # Database Manager
        # -------------------------------
        self.db = DBManager(config)
        self._finalized = False

        # --------------------------------------------------
        # Safe finalize on exit (single registration)
        # --------------------------------------------------
        import atexit
        atexit.register(self._finalize_on_exit)
   
    def _finalize_on_exit(self):
        try:
            self.finalize(fps=25)
        except Exception as e:
            print("Failed to finalize on exit:", e)
    

    def _check_daily_rollover(self, fps):
        now_date_str = datetime.now().strftime("%d-%m-%Y")

        if now_date_str != self.current_date_str:
            print(f"DB Date changed: {self.current_date_str} → {now_date_str}")
            # 1️⃣ Finalize previous day CSV
            self.finalize(fps)

            # 2️⃣ Reset date & epoch
            self.current_date_str = now_date_str
            self.current_day_start_epoch = int(time.time())
            self.run_date = now_date_str

            # 3️⃣ Reset daily counters
            self.walking_time.clear()
            self.hand_movement_time.clear()
            self.sitting_time.clear()
            self.sitting_hand_time.clear()
            self.standing_time.clear()
            self.unknown_time.clear()

            self.pose_history.clear()
            self.id_to_name.clear()
            
            print(f"Started new DB day for {self.current_date_str}")



    def save_partial(self):
        if not self.db.enabled:
            return

        now = time.time()
        if now - self._last_csv_write_time < self.realtime_csv_interval:
            return

        self._last_csv_write_time = now

        agg = defaultdict(lambda: {"active": 0.0, "idle": 0.0})

        # ACTIVE TIME
        for tid, sec in self.walking_time.items():
            name = self.id_to_name.get(tid, f"Unknown_{tid}")
            agg[name]["active"] += sec

        for tid, sec in self.hand_movement_time.items():
            name = self.id_to_name.get(tid, f"Unknown_{tid}")
            agg[name]["active"] += sec

        for tid, sec in self.sitting_hand_time.items():
            name = self.id_to_name.get(tid, f"Unknown_{tid}")
            agg[name]["active"] += sec

        # IDLE TIME
        for tid, sec in self.sitting_time.items():
            name = self.id_to_name.get(tid, f"Unknown_{tid}")
            agg[name]["idle"] += sec

        for tid, sec in self.standing_time.items():
            name = self.id_to_name.get(tid, f"Unknown_{tid}")
            agg[name]["idle"] += sec

        for tid, sec in self.unknown_time.items():
            name = self.id_to_name.get(tid, f"Unknown_{tid}")
            agg[name]["idle"] += sec

        # 🔁 UPSERT INTO DB
        for worker, t in agg.items():
            self.db.upsert_productivity(
                self.camera_id,
                worker,
                t["active"],
                t["idle"]
            )


    # --------------------------------------------------
    # Angle helpers
    # --------------------------------------------------
    @staticmethod
    def compute_angle(A, B, C):
        try:
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)
            BA = A - B
            BC = C - B
            cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        except:
            return None

    def classify_angles(self, kpts):
        if kpts is None:
            return False, False

        L_SHO, R_SHO = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANK, R_ANK = 15, 16

        standing_votes, sitting_votes = [], []
        legs = [(L_SHO, L_HIP, L_KNEE, L_ANK), (R_SHO, R_HIP, R_KNEE, R_ANK)]

        for SHO, HIP, KNEE, ANK in legs:
            sx, sy, sc = kpts[SHO]
            hx, hy, hc = kpts[HIP]
            kx, ky, kc = kpts[KNEE]
            ax, ay, ac = kpts[ANK]

            if min(sc, hc, kc, ac) < 0.2:
                continue

            knee = self.compute_angle((hx, hy), (kx, ky), (ax, ay))
            hip = self.compute_angle((sx, sy), (hx, hy), (kx, ky))
            if knee is None or hip is None:
                continue

            standing_votes.append(knee > 150 and hip > 140)
            sitting_votes.append(65 < knee < 150 and 55 < hip < 135)

        return any(standing_votes), any(sitting_votes)

    def classify_walking(self, track_id, x1, y1, x2, y2):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if track_id not in self.prev_centers:
            self.prev_centers[track_id] = (cx, cy)
            return False
        px, py = self.prev_centers[track_id]
        self.prev_centers[track_id] = (cx, cy)
        return np.hypot(cx - px, cy - py) > self.walk_threshold

    def detect_hand_movement(self, track_id, kpts, walking):
        if kpts is None or walking:
            return False

        ids = [5, 6, 7, 8, 9, 10]
        pts = [(kpts[i][0], kpts[i][1]) for i in ids if kpts[i][2] > 0.25]
        if not pts:
            return False

        hx, hy = np.mean(pts, axis=0)

        if track_id not in self.prev_hands:
            self.prev_hands[track_id] = (hx, hy)
            return False

        px, py = self.prev_hands[track_id]
        self.prev_hands[track_id] = (hx, hy)
        return np.hypot(hx - px, hy - py) > self.hand_move_threshold

    # --------------------------------------------------
    # MAIN UPDATE (ROI-aware)
    # --------------------------------------------------
    
    def update(self, track_id, worker_name, x1, y1, x2, y2, kpts, dt, inside_roi=True):

        self.elapsed_video_seconds += max(0.0, dt)
        self._check_daily_rollover(fps=25)

        # -------------------------------
        # Normalize worker name
        # -------------------------------
        if not worker_name or worker_name == "Unknown":
            worker_name = f"Unknown_{track_id}"

        prev_name = self.id_to_name.get(track_id)

        # -------------------------------
        # Unknown → Known identity upgrade
        # -------------------------------
        if prev_name and prev_name.startswith("Unknown_") and worker_name != prev_name:
            # Only update name mapping
            self.id_to_name[track_id] = worker_name

            # Optional: delete old DB row (safe)
            self.db.delete_worker(self.camera_id, prev_name)
        else:
            self.id_to_name[track_id] = worker_name

        # -------------------------------
        # Pose logic (UNCHANGED)
        # -------------------------------
        walking = self.classify_walking(track_id, x1, y1, x2, y2)
        standing_pose, sitting_pose = self.classify_angles(kpts)
        hand_move = self.detect_hand_movement(track_id, kpts, walking)

        self.pose_history[track_id].append((walking, standing_pose, sitting_pose, hand_move))
        if len(self.pose_history[track_id]) > self.pose_window:
            self.pose_history[track_id] = self.pose_history[track_id][-self.pose_window:]

        recent = self.pose_history[track_id]
        walking = sum(p[0] for p in recent) > len(recent) / 2
        standing_pose = sum(p[1] for p in recent) > len(recent) / 2
        sitting_pose = sum(p[2] for p in recent) > len(recent) / 2
        hand_move = sum(p[3] for p in recent) > len(recent) / 2

        if not inside_roi:
            return walking, hand_move, standing_pose, sitting_pose

        # -------------------------------
        # Time accumulation (KEYED BY track_id ONLY)
        # -------------------------------
        if walking:
            self.walking_time[track_id] += dt
        elif sitting_pose and hand_move:
            self.sitting_hand_time[track_id] += dt
        elif sitting_pose:
            self.sitting_time[track_id] += dt
        elif standing_pose and hand_move:
            self.hand_movement_time[track_id] += dt
        elif standing_pose:
            self.standing_time[track_id] += dt
        else:
            self.unknown_time[track_id] += dt

        return walking, hand_move, standing_pose, sitting_pose



    def draw_posture(self, frame, x1, y1, x2, y2,walking, hand_move, standing_pose, sitting_pose):

        if walking:
            text = "Walking"
            color = (0, 255, 0)

        elif sitting_pose and hand_move:
            text = "Sitting+HandMove"
            color = (0, 165, 255)

        elif sitting_pose:
            text = "Sitting"
            color = (0, 100, 255)

        elif standing_pose and hand_move:
            text = "Standing+HandMove"
            color = (0, 200, 255)

        elif standing_pose:
            text = "Standing"
            color = (255, 255, 0)

        else:
            return frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        pad_x = 6
        pad_y = 4
        text_x = x2 - (tw + 2 * pad_x)
        text_y = y1 - 10

        if text_y - th - pad_y < 0:
            text_y = y1 + th + 10

        bg_x1 = text_x - pad_x
        bg_y1 = text_y - th - pad_y
        bg_x2 = text_x + tw + pad_x
        bg_y2 = text_y + pad_y

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y),
                    font, font_scale, color, thickness)


    def finalize(self, fps):

        if self._finalized:
                return
        self._finalized = True

        agg = defaultdict(lambda: {
            "walk": 0.0,
            "stand_hand": 0.0,
            "sit": 0.0,
            "sit_hand": 0.0,
            "stand": 0.0,
            "unknown": 0.0
        })

        def add_time(td, k):
            for tid, sec in td.items():
                raw = self.id_to_name.get(tid, "Unknown")
                name = raw if raw != "Unknown" else f"Unknown_{tid}"
                agg[name][k] += sec

        add_time(self.walking_time, "walk")
        add_time(self.hand_movement_time, "stand_hand")
        add_time(self.sitting_time, "sit")
        add_time(self.sitting_hand_time, "sit_hand")
        add_time(self.standing_time, "stand")
        add_time(self.unknown_time, "unknown")


        
        for worker, d in agg.items():
            active = d["walk"] + d["stand_hand"] + d["sit_hand"]
            idle = d["sit"] + d["stand"] + d["unknown"]

            self.db.upsert_productivity(
                self.camera_id,
                worker,
                active,
                idle
            )

        print("Productivity finalized and stored in DB")