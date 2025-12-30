import os
import sys
import json
import cv2
import numpy as np
import math
import torch
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from productivity_tracker import ProductivityTracker
from utils.perf_profiler import PerfProfiler


# Force-disable StrongSORT as per original script
StrongSORT_Class = None

# Attempt to import face recognizer (original)
try:
    from face_recognizer import FaceRecognizer
except Exception as e:
    print("Failed to import face_recognizer.py. Make sure it is next to this script.")
    raise

try:
    from person_features import upper_body_crop, compute_hsv_hist, hist_to_list, list_to_hist
except Exception as e:
    print("Failed to import person_features.py. Make sure it exists next to inference.py.")
    raise

# -------------------------
# Config loader + device selection
# -------------------------
def load_config(path="config.properties"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {path}")
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg

def parse_input_sources(config):
    if "input_sources" in config:
        return [s.strip() for s in config["input_sources"].split("|") if s.strip()]
    return [config["input_video_path"]]

def select_device(device_pref="auto"):
    device_pref = str(device_pref).lower()

    if device_pref == "cpu":
        return "cpu"

    if device_pref in ("gpu", "cuda", "cuda:0"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    return "cuda:0" if torch.cuda.is_available() else "cpu"


# -------------------------
# Pose drawing helpers (unchanged)
# -------------------------
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12)
]

def draw_keypoints(frame, keypoints, color=(255, 0, 0)):
    if keypoints is None:
        return frame
    for kp in keypoints:
        for x, y, conf in kp:
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return frame

def draw_skeleton(frame, keypoints, color=(0, 255, 255)):
    if keypoints is None:
        return frame
    for kp in keypoints:
        for (i1, i2) in SKELETON:
            x1, y1, c1 = kp[i1]
            x2, y2, c2 = kp[i2]
            if c1 > 0.3 and c2 > 0.3:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return frame

# -------------------------
# Persistence helpers
# -------------------------
def load_id_name_map(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_id_name_map(path, mapping):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)

# ReID map (appearance histograms) persistence
def load_reid_map(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return {k: v if isinstance(v, list) else [] for k, v in data.items()}
    except Exception:
        return {}

def save_reid_map(path, mapping):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)

# -------------------------
# Histogram matching helper
# -------------------------
def match_histogram_to_names(query_hist, reid_map, method=cv2.HISTCMP_CORREL):
    """
    Compare query_hist to all stored hist lists in reid_map.
    Returns (best_name, best_score) or (None, None).
    Uses cv2.compareHist on flattened hist arrays (float32).
    """
    if query_hist is None:
        return None, None

    best_name = None
    best_score = -9999.0

    for name, hlist in reid_map.items():
        for stored in hlist:
            if stored is None or len(stored) == 0:
                continue
            stored_hist = list_to_hist(stored)
            if stored_hist is None:
                continue
            try:
                score = cv2.compareHist(query_hist.astype('float32'), stored_hist.astype('float32'), method)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_name = name

    if best_name is None:
        return None, None
    return best_name, float(best_score)

# -------------------------
# Simple bbox-walking helper (unchanged)
# -------------------------
prev_centers = {}  # track_id → (cx, cy)

def is_walking(track_id, x1, y1, x2, y2, threshold=8):
    """
    Simple walking classifier based on bbox center movement.
    """
    global prev_centers

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if track_id not in prev_centers:
        prev_centers[track_id] = (cx, cy)
        return False  # can't classify yet

    px, py = prev_centers[track_id]
    prev_centers[track_id] = (cx, cy)

    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

    return dist > threshold


def is_inside_polygon_roi(x1, y1, x2, y2, roi_polygon, frame_w, frame_h):
    if not roi_polygon:
        return True

    # Convert normalized polygon → pixel coords
    poly = np.array([
        (int(px * frame_w), int(py * frame_h))
        for px, py in roi_polygon
    ], dtype=np.int32)

    # Person center
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # OpenCV point-in-polygon test
    return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0




# -------------------------
# Main inference function 
# -------------------------

    
def run_inference(config, frame_queue=None):

    """
    Main entry for running inference using the provided config dict.
    Part 1 sets up variables and models; main loop is in Part 2.
    """
    # Required config values (paths)
    pose_model_path = config["pose_model_path"]
    face_detector_model_path = config["face_detector_model_path"]
    face_onnx_path = config["face_onnx_path"]
    face_embeddings_json = config["face_embeddings_json"]
    input_video_path = config["input_video_path"]
    
    # -------------------------------------------------
    # INPUT SOURCE TYPE DETECTION (ADDED)
    # -------------------------------------------------
    src = str(input_video_path)

    is_webcam_index = src.isdigit()
    is_dev_camera = src.startswith("/dev/video")
    is_rtsp = src.startswith("rtsp://")

    is_live = is_webcam_index or is_dev_camera or is_rtsp

    
    from datetime import datetime

    # ---- RUN TIMESTAMPS ----
    run_start_epoch = int(time.time())
    current_date_str = datetime.now().strftime("%d-%m-%Y")

    
    camera_id = config.get("camera_id", "cam_1")
    base_output = os.path.join(config["output_folder"], camera_id)


    video_dir = os.path.join(base_output, "output_videos")
    json_dir = os.path.join(base_output, "json_files")
    
    def build_daily_paths(date_str):
        id_map = os.path.join(json_dir, f"{date_str}_id_name_map.json")
        reid_map = os.path.join(json_dir, f"{date_str}_reid_map.json")
        return id_map, reid_map

    
    report_dir = os.path.join(base_output, "reports")

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)


    # Detection & face thresholds
    conf = float(config.get("confidence_threshold", 0.5))
    face_conf = float(config.get("face_confidence_threshold", 0.45))
    face_recog_threshold = float(config.get("face_recognition_threshold", 0.6))
    device_pref = config.get("device", "auto")

    save_output = config.get("save_output_video", "true").lower() == "true"

    face_nose_relative_thresh = float(config.get("face_nose_relative_thresh", 0.25))

    pop_up_window = config.get("pop_up_window", "false").lower() == "true"
    
    
    if frame_queue is not None:
        pop_up_window = False

    resize_w = int(config.get("resize_window_width", 0))
    resize_h = int(config.get("resize_window_height", 0))

    reid_similarity_threshold = float(config.get("reid_similarity_threshold", 0.60))

    # Additional safety/tuning params
    reid_duplicate_protect_frames = int(config.get("reid_duplicate_protect_frames", 50))
    reid_duplicate_override_threshold = float(config.get("reid_duplicate_override_threshold", 0.92))
    reid_persistence_frames = int(config.get("reid_persistence_frames", 3))

    if pop_up_window:
        print("Live Inference Window: ENABLED")
        if resize_w > 0 and resize_h > 0:
            print(f"Window Resize: {resize_w} x {resize_h}")
        else:
            print("Window Resize: ORIGINAL SIZE")
    else:
        print("Live Inference Window: DISABLED")

    # device
    device = select_device(device_pref)
    print(f"[Inference] Selected device: {device}")


    # -------------------------
    # Load models
    # -------------------------
    pose_model = YOLO(pose_model_path)
    pose_model.to(device)

    face_detector = YOLO(face_detector_model_path)
    face_detector.to(device)
    
    recognizer = FaceRecognizer(
        face_onnx_path,
        face_embeddings_json,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    
    # -------------------------
    # Performance profiler
    # -------------------------
    profiler = PerfProfiler()


    os.makedirs(base_output, exist_ok=True)

    # -------------------------------------------------
    # INPUT SOURCE OPEN (UPDATED)
    # -------------------------------------------------
    if is_webcam_index:
        cap = cv2.VideoCapture(int(src))
    elif is_dev_camera:
        cap = cv2.VideoCapture(src)
    elif is_rtsp:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    else:
        cap = cv2.VideoCapture(src)


    if not cap.isOpened():
        print(f"Could not open source: {input_video_path}")
        return
    
    if is_live:
        print("Live source enabled")
        video_name = "live_stream.mp4"
    else:
        video_name = os.path.basename(str(input_video_path))

    temp_video_path = os.path.join(video_dir, "temp_run.mp4")

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0

    # -------------------------
    # ROI CONFIG
    # -------------------------

    roi_enabled = config.get("roi_enabled", "false").lower() == "true"
    roi_polygon = None

    if roi_enabled:
        roi_polygon = []
        pts = config["roi_polygon"].split("|")
        for p in pts:
            x, y = map(float, p.strip().split(","))
            roi_polygon.append((x, y))

        print(f"Polygon ROI enabled with {len(roi_polygon)} points")
    else:
        print("ROI DISABLED")


    writer = None
    if save_output:
        writer = cv2.VideoWriter(
            temp_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

    id_name_map_path, reid_map_path = build_daily_paths(current_date_str)

    id_name_map = load_id_name_map(id_name_map_path)
    id_cache = {
        int(k): {"name": v, "score": 1.0}
        for k, v in id_name_map.items()
    }

    reid_map = load_reid_map(reid_map_path)



    # productivity tracker
    
    productivity = ProductivityTracker(report_dir,config,run_start_epoch)


    # runtime maps for duplicate protection and persistence
    name_last_assigned = {}       
    reid_temp_counts = {}            

    print(f"Processing {input_video_path} -> {video_dir}")

    frame_idx = 0
    prev_timestamp_ms = None  # will hold the timestamp (ms) of previous processed frame

    try:
        while True:
            ret, frame = cap.read()
            
            
            # DAILY ROLLOVER CHECK
            now_date_str = datetime.now().strftime("%d-%m-%Y")

            if now_date_str != current_date_str:
                print(f"Date changed: {current_date_str} → {now_date_str}")

                current_date_str = now_date_str

                # Switch to new daily JSON files
                id_name_map_path, reid_map_path = build_daily_paths(current_date_str)

                # Reset daily caches
                id_cache.clear()
                reid_map.clear()

                # Load existing files if they already exist
                id_name_map = load_id_name_map(id_name_map_path)
                id_cache.update({
                    int(k): {"name": v, "score": 1.0}
                    for k, v in id_name_map.items()
                })

                reid_map.update(load_reid_map(reid_map_path))

                print(f"Using new daily JSON files: {current_date_str}")

            if not ret:
                break
            frame_idx += 1

            
            # -------------------------------------------------
            # TIME DELTA CALCULATION (FIXED FOR ALL INPUT TYPES)
            # -------------------------------------------------
            now = time.time()

            if prev_timestamp_ms is None:
                dt = 0.0
            else:
                if is_live:
                    # webcam / dev video / RTSP
                    dt = max(0.0, now - prev_timestamp_ms)
                else:
                    # video file
                    video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    dt = max(0.0, video_time_sec - prev_timestamp_ms)

            prev_timestamp_ms = now if is_live else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


            ctx = profiler.start("pose_model")
            # Run pose model with tracking (ByteTrack)
            results = pose_model.track(
                source=frame,
                conf=conf,
                imgsz=640,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )
            profiler.end(ctx)
            r = results[0]

            # Filter persons only
            if r.boxes is not None and len(r.boxes) > 0:
                try:
                    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
                    person_idx = cls_arr == 0
                    r.boxes = r.boxes[person_idx]
                    if r.keypoints is not None:
                        r.keypoints = r.keypoints[person_idx]
                except Exception:
                    # if filtering fails, keep original boxes
                    pass

            annotated = frame.copy()
            annotator = Annotator(annotated, line_width=2)

            if roi_enabled and roi_polygon:
                poly_pts = np.array([
                    (int(x * width), int(y * height))
                    for x, y in roi_polygon
                ], dtype=np.int32)

                cv2.polylines(
                    annotated,
                    [poly_pts],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=4
                )

            # boxes and IDs
            if r.boxes is not None and len(r.boxes) > 0:
                try:
                    boxes = r.boxes.xyxy.cpu().numpy()
                except:
                    boxes = np.array([])
                try:
                    ids_tensor = getattr(r.boxes, "id", None)
                    ids = (ids_tensor.cpu().numpy().astype(int)
                           if ids_tensor is not None
                           else np.array([-1] * len(boxes)))
                except:
                    ids = np.array([-1] * len(boxes))
            else:
                boxes = np.array([])
                ids = np.array([], dtype=int)

            try:
                kpts = (r.keypoints.data.cpu().numpy()
                        if r.keypoints is not None else None)
            except:
                kpts = None

            # Process each detected person
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(ids[i]) if i < len(ids) else -1

                inside_roi = True
                    
                if roi_enabled:
                    ctx = profiler.start("roi_check")
                    inside_roi = is_inside_polygon_roi(x1, y1, x2, y2,roi_polygon,width,height)
                    profiler.end(ctx)
                    
                annotator.box_label([x1, y1, x2, y2], None, color=(0, 255, 0))

                cached_name = id_cache.get(track_id, {}).get("name")

                # Compute a top crop region to search for faces
                pw = max(10, x2 - x1)
                ph = max(10, y2 - y1)
                pad_x = int(0.05 * pw)
                pad_y = int(0.05 * ph)
                cx1 = max(0, x1 + pad_x)
                cy1 = max(0, y1 + pad_y)
                cx2 = min(width, x2 - pad_x)
                cy2 = min(height, int(y1 + 0.65 * (y2 - y1)))

                face_name = None
                face_score = 0.0
                fx1_full = fy1_full = fx2_full = fy2_full = None

                # Extract nose if keypoints exist (used for pose-based face validation)
                nose_x = nose_y = nose_conf = None
                if kpts is not None and i < len(kpts):
                    try:
                        nx, ny, nconf = kpts[i][0]
                        nose_x, nose_y, nose_conf = float(nx), float(ny), float(nconf)
                    except:
                        nose_x = nose_y = nose_conf = None

                # Run face detector on the top region
                if cx2 > cx1 and cy2 > cy1:
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        ctx = profiler.start("face_detector")
                        face_results = face_detector(
                            crop, conf=face_conf,imgsz=1024,
                            device=device, verbose=False
                        )
                        profiler.end(ctx)

                        best = None
                        best_conf = 0
                        for fr in face_results:
                            if fr.boxes is not None:
                                for b in fr.boxes:
                                    xy = b.xyxy[0].cpu().numpy()
                                    fx1, fy1, fx2, fy2 = map(int, xy)
                                    fconf = float(b.conf[0])
                                    if fconf > best_conf:
                                        best_conf = fconf
                                        best = (fx1, fy1, fx2, fy2)

                        if best:
                            fx1, fy1, fx2, fy2 = best
                            fx1_full = cx1 + fx1
                            fy1_full = cy1 + fy1
                            fx2_full = cx1 + fx2
                            fy2_full = cy1 + fy2

                            # Check face size and acceptance by estimated nose position
                            if (fx2_full - fx1_full > 10 and fy2_full - fy1_full > 10):
                                face_fully_inside_top = (
                                    fx1_full >= cx1 and fy1_full >= cy1 and
                                    fx2_full <= cx2 and fy2_full <= cy2
                                )

                                if face_fully_inside_top:
                                    accept_by_pose = True
                                    if nose_x is not None and nose_conf > 0.25:
                                        fcx = (fx1_full + fx2_full) / 2
                                        fcy = (fy1_full + fy2_full) / 2
                                        dist = np.sqrt(
                                            (fcx - nose_x) ** 2 +
                                            (fcy - nose_y) ** 2
                                        )
                                        rel = dist / float(max(1.0, ph))
                                        accept_by_pose = (rel <= face_nose_relative_thresh)

                                    if accept_by_pose:
                                        face_crop = frame[fy1_full:fy2_full,
                                                          fx1_full:fx2_full]
                                        ctx = profiler.start("face_recognition")
                                        
                                        name, score = recognizer.recognize(
                                            face_crop,
                                            threshold=face_recog_threshold
                                        )
                                        profiler.end(ctx)
                                        
                                        face_name = name
                                        face_score = score

                                        # If face recognized, update id_cache and name_last_assigned
                                        if name != "Unknown":
                                            id_cache[track_id] = {"name": name, "score": score}
                                            try:
                                                saved_map = {str(k): v["name"] for k, v in id_cache.items()}
                                                save_id_name_map(id_name_map_path, saved_map)
                                            except Exception:
                                                pass
                                            # update last assigned mapping
                                            try:
                                                name_last_assigned[name] = (track_id, frame_idx)
                                            except Exception:
                                                pass

                                        # Save appearance histogram for recognized face
                                        try:
                                            upper_crop = upper_body_crop(frame, x1, y1, x2, y2, frac=0.45)
                                            qhist = compute_hsv_hist(upper_crop)
                                            if face_name and face_name != "Unknown" and qhist is not None:
                                                entry = reid_map.get(face_name, [])
                                                entry.append(hist_to_list(qhist))
                                                MAX_HISTS_PER_PERSON = 5
                                                if len(entry) > MAX_HISTS_PER_PERSON:
                                                    entry = entry[-MAX_HISTS_PER_PERSON:]
                                                reid_map[face_name] = entry
                                                try:
                                                    save_reid_map(reid_map_path, reid_map)
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass

                # Also, if we have a cached name for this track, strengthen appearance map
                try:
                    if cached_name and cached_name != "Unknown":
                        upper_crop = upper_body_crop(frame, x1, y1, x2, y2, frac=0.45)
                        qhist = compute_hsv_hist(upper_crop)
                        if qhist is not None:
                            entry = reid_map.get(cached_name, [])
                            entry.append(hist_to_list(qhist))
                            if len(entry) > 5:
                                entry = entry[-5:]
                            reid_map[cached_name] = entry
                            try:
                                save_reid_map(reid_map_path, reid_map)
                            except Exception:
                                pass
                except Exception:
                    pass

                # Determine display_name with fallback to appearance matching
                display_name = None
                if cached_name:
                    display_name = cached_name
                if face_name:
                    # face_name has higher priority than cached_name (face detection is high-confidence)
                    display_name = face_name

                # -------------------------
                # Safer appearance matching (with duplicate protection + persistence)
                # -------------------------
                if display_name is None or display_name == "Unknown":
                    try:
                        upper_crop = upper_body_crop(frame, x1, y1, x2, y2, frac=0.45)
                        qhist = compute_hsv_hist(upper_crop)
                        if qhist is not None and len(reid_map) > 0:
                            best_name, best_score = match_histogram_to_names(qhist, reid_map, method=cv2.HISTCMP_CORREL)

                            if best_name and best_score is not None:
                                # Basic threshold check
                                if best_score >= reid_similarity_threshold:
                                    conflict = False
                                    # Check recent owner
                                    owner = name_last_assigned.get(best_name)
                                    if owner is not None:
                                        owner_tid, owner_frame = owner
                                        if owner_tid != track_id and (frame_idx - owner_frame) < reid_duplicate_protect_frames:
                                            conflict = True

                                    # Allow override only if extremely confident
                                    if conflict:
                                        if best_score >= reid_duplicate_override_threshold:
                                            conflict = False

                                    if not conflict:
                                        # Temporal persistence: count consecutive frames for same candidate
                                        key = (track_id, best_name)
                                        prev_count = reid_temp_counts.get(key, 0)
                                        reid_temp_counts[key] = prev_count + 1

                                        # Clear other candidates' counts for this track
                                        for k in list(reid_temp_counts.keys()):
                                            if k[0] == track_id and k[1] != best_name:
                                                reid_temp_counts.pop(k, None)

                                        if reid_temp_counts.get(key, 0) >= reid_persistence_frames:
                                            # commit identity
                                            display_name = best_name
                                            id_cache[track_id] = {"name": display_name, "score": float(best_score)}
                                            name_last_assigned[display_name] = (track_id, frame_idx)
                                            try:
                                                saved_map = {str(k): v["name"] for k, v in id_cache.items()}
                                                save_id_name_map(id_name_map_path, saved_map)
                                            except Exception:
                                                pass
                                    else:
                                        # conflict: do not assign, reset temp counter for this candidate
                                        reid_temp_counts.pop((track_id, best_name), None)
                                else:
                                    # below threshold -> clear any persistence for this candidate
                                    reid_temp_counts.pop((track_id, best_name), None)
                    except Exception:
                        pass

                if not display_name:
                    display_name = "Unknown"

                label = f"({display_name})"
                
                ctx = profiler.start("productivity_update")
                walking, hand_move, standing_pose, sitting_pose = productivity.update(
                    track_id,
                    display_name,
                    x1, y1, x2, y2,
                    kpts[i] if kpts is not None else None,
                    dt,
                    inside_roi
                )
                profiler.end(ctx)

                productivity.draw_posture(
                    annotated,
                    x1, y1, x2, y2,
                    walking, hand_move, standing_pose, sitting_pose
                )

                # Put text label
                cv2.putText(
                    annotated, label,
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                # Draw face rectangle + name if available
                if fx1_full is not None:
                    cv2.rectangle(annotated,
                                  (fx1_full, fy1_full),
                                  (fx2_full, fy2_full),
                                  (255, 200, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{display_name} {face_score:.2f}",
                        (fx1_full, max(15, fy1_full - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 200, 0), 2
                    )

            # End of boxes loop

            productivity.save_partial()
            
            # draw skeleton & keypoints
            if kpts is not None:
                annotated = draw_skeleton(annotated, kpts)
                annotated = draw_keypoints(annotated, kpts)

            # Cleanup stale name_last_assigned entries occasionally
            stale_names = []
            for nm, (tid, last_f) in name_last_assigned.items():
                if (frame_idx - last_f) > (reid_duplicate_protect_frames * 2):
                    stale_names.append(nm)
            for nm in stale_names:
                name_last_assigned.pop(nm, None)

            # 🔁 Send latest frame to viewer (multiprocessing-safe)
            if frame_queue is not None:
                try:
                    frame_queue.put_nowait((camera_id, annotated))
                except:
                    pass


            # live display
            if pop_up_window:
                display_frame = (cv2.resize(annotated, (resize_w, resize_h))
                                 if resize_w > 0 and resize_h > 0
                                 else annotated)
                cv2.imshow("Live Inference", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_output and writer is not None:
                writer.write(annotated)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")

    # Finalization, saving, and main guard

    finally:
        
        # 🔔 Notify viewer that this camera is done
        if frame_queue is not None:
            try:
                frame_queue.put((camera_id, None))
            except:
                pass

        
        cap.release()

        # Set total video seconds using last observed timestamp (more robust than frames/fps)
        # prev_timestamp_ms contains POS_MSEC of last processed frame
        # -------------------------------------------------
        # FINAL TOTAL VIDEO TIME (FIXED)
        # -------------------------------------------------
        if is_live:
            productivity.total_video_seconds = productivity.elapsed_video_seconds
        else:
            productivity.total_video_seconds = frame_idx / (fps if fps > 0 else 25.0)


        productivity.finalize(fps)
        
        import json

        perf_out = profiler.summary()

        with open(
            os.path.join(base_output, "performance_summary.json"), "w"
        ) as f:
            json.dump(perf_out, f, indent=2)

        print("Performance summary saved")

        

        if writer is not None:
            writer.release()
            
            run_end_epoch = int(time.time())

            if save_output and os.path.exists(temp_video_path):
                final_video_name = f"{run_start_epoch}-{run_end_epoch}.mp4"
                final_video_path = os.path.join(video_dir, final_video_name)
                os.rename(temp_video_path, final_video_path)

                print("Output video saved to:", final_video_path)

        else:
            print("Output saving disabled. No video created.")

        cv2.destroyAllWindows()

        # Save id -> name map
        try:
            save_id_name_map(id_name_map_path,
                             {str(k): v["name"] for k, v in id_cache.items()})
        except Exception:
            pass

        # Persist reid_map
        try:
            save_reid_map(reid_map_path, reid_map)
        except Exception:
            pass

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Standalone single-camera debug mode ONLY
    cfg = load_config("config.properties")
    run_inference(cfg, frame_queue=None)