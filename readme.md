# 🏭 AI-Based Worker Productivity Monitoring System

An **end-to-end real-time worker monitoring and productivity analytics system** built using **YOLOv8 Pose Estimation, Face Recognition, Appearance Re-Identification, and Multi-Camera Processing**.

The system identifies workers, tracks them across frames and cameras, classifies postures and activities, and computes **accurate active vs idle working time** using **real elapsed time (not frame counting)**.  
It is designed for **industrial environments**, **warehouse floors**, and **manufacturing units**.

---

## 🚀 Key Capabilities

### 🎯 Core Features
- Real-time **multi-camera inference**
- **YOLOv8 Pose Estimation** with ByteTrack
- **Face recognition** using ONNX models
- **Appearance-based Re-Identification (HSV histograms)**
- Accurate **activity classification**
  - Walking
  - Standing
  - Sitting
  - Hand movement
- **Active vs Idle time calculation** using real timestamps (`dt`)
- **Polygon-based ROI filtering**
- **Live multi-camera grid viewer**
- **Daily identity persistence & rollover handling**
- **Database-backed productivity storage**
- **Performance profiling (CPU & latency)**

---

## 🧠 System Architecture Overview

---

## 🧩 Detailed Module Breakdown

### 1️⃣ main.py — System Orchestrator
- Launches **multiple camera processes**
- Spawns a **central viewer process**
- Uses **multiprocessing (spawn mode)** for CUDA safety
- Handles graceful shutdown of the full system

---

### 2️⃣ cam_worker.py — Camera Process Wrapper
- Runs inference for **one camera/source**
- Isolates failures so one camera crash does **not affect others**

---

### 3️⃣ inference.py — Core Intelligence Engine
The **heart of the system**.

**Responsibilities**
- Loads pose, face detection, and face recognition models
- Supports:
  - Webcam
  - RTSP streams
  - Video files
- Performs:
  - Person detection & tracking (ByteTrack)
  - Face recognition
  - Appearance-based ReID fallback
- Maintains:
  - Daily identity maps
  - ReID feature persistence
- Applies:
  - ROI polygon filtering
  - Duplicate identity protection
- Produces:
  - Annotated output video
  - Live frames for viewer
  - Performance metrics (JSON)

---

### 4️⃣ face_recognizer.py — Identity via Face Embeddings
- ONNX-based face embedding inference
- Cosine similarity matching
- Embedding normalization
- Threshold-based recognition
- Persistent JSON-based embedding storage

**Purpose:**  
Provides **high-confidence identity assignment** when faces are visible.

---

### 5️⃣ person_features.py — Appearance Re-Identification
- Upper-body cropping
- HSV histogram extraction
- JSON-serializable feature storage
- Used as a **fallback identity mechanism** when face recognition fails

---

### 6️⃣ productivity_tracker.py — Activity Intelligence
Tracks productivity using **real elapsed time (`dt`)**.

**Tracked Activities**
- Walking
- Standing
- Sitting
- Standing + Hand Movement
- Sitting + Hand Movement
- Unknown

**Key Design Decisions**
- Time-based (not frame-based)
- ROI-aware accumulation
- Identity upgrade handling (`Unknown → Known`)
- Safe daily rollover at midnight
- Periodic DB upserts
- Guaranteed finalization via `atexit`

---

### 7️⃣ utils/db_manager.py — Database Abstraction
- Safe enable/disable via configuration
- Day-based epoch keys
- Atomic UPSERT logic
- Prevents duplicate daily records
- Supports identity cleanup when names change

---

### 8️⃣ utils/perf_profiler.py — Runtime Performance Analysis
- Per-module execution timing
- CPU usage tracking
- Aggregated performance summaries
- Exported automatically as JSON

---

### 9️⃣ utils/grid_view.py — Multi-Camera Layout Builder
- Computes optimal grid rows/columns dynamically
- Uniform frame resizing
- Padding for missing camera feeds
- Produces a clean composite display

---

### 🔟 viewer.py — Live Multi-Camera Dashboard
- Receives frames via multiprocessing queue
- Displays synchronized grid view
- Auto-scales output to screen size
- Handles graceful camera termination
- Manual exit support (`q`)

---

## 🧪 Productivity Logic (Active vs Idle)

| Activity                         | Category |
|----------------------------------|----------|
| Walking                          | Active   |
| Standing + Hand Movement         | Active   |
| Sitting + Hand Movement          | Active   |
| Sitting                          | Idle     |
| Standing                         | Idle     |
| Unknown                          | Idle     |

✔ All calculations are based on **real time delta (`dt`)**, ensuring accuracy across:
- Variable FPS
- RTSP jitter
- Dropped frames

---

## 🛡️ Reliability & Safety Features

- Daily identity isolation
- Duplicate ReID protection
- Temporal persistence checks
- Graceful exit finalization
- Process isolation per camera
- GPU-safe multiprocessing

---

## 📈 Output Artifacts (High-Level)

- Annotated output videos
- Live multi-camera viewer
- Daily JSON identity maps
- Performance summary reports
- Database productivity records

*(Exact paths, credentials, configs intentionally excluded)*

---

## 🧠 Ideal Use Cases

- Factory worker monitoring
- Warehouse productivity analysis
- Safety & compliance tracking
- Multi-camera industrial AI systems
- Smart manufacturing analytics

---

## 🔮 Future Enhancements

- Web dashboard (Streamlit / React)
- Alerting system (idle thresholds)
- Cross-day identity linking
- Heatmaps & zone analytics
- Shift-based productivity reports
- Cloud & edge deployment

---

## 👨‍💻 Author

**Manoj R**  
AI Engineer | Computer Vision | Real-Time Systems  

Focused on **production-grade AI pipelines for industrial environments**