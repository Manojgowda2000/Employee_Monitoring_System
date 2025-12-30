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

