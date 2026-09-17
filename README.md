# 🛡️ Face Recognition–Based Attendance Portal

A modern, biometric desktop & web-based attendance system powered by Computer Vision. Features automated face detection, 128-dimensional biometric embeddings, instant check-in verification, duplicate-prevention, comprehensive roster reporting (Present vs. Absent), and a clean **Light Theme** user interface.

Built for **VITyarthi — Build Your Own Project** (Compliant with PRD `Face_Attendance_System_PRD.md`).

---

## ✨ Key Features

- **📷 Live Biometric Scanner (Kiosk Mode)**:
  - **Instant Snapshot & Photo Scanner**: Zero latency, browser-native camera capture with real-time bounding boxes, match confidence %, and immediate check-in feedback.
  - **Continuous Live Stream Feed**: Smooth OpenCV video stream with automated background verification and duplicate suppression.
  - **Live Check-in Stream**: Real-time ticker showing members checking in today with time badges.
- **🧑 Face Enrollment & Roster Management**:
  - Live face detection quality validator (ensures single-face clarity before enrolling).
  - Multi-photo training support for higher matching accuracy under varying angles and lighting.
  - Registered members directory with search and delete actions.
- **📊 Attendance Intelligence & Reports**:
  - Filter by presets: *Today*, *Yesterday*, *Last 7 Days*, *This Month*, *All Time*, or custom range.
  - Real-time KPI Cards: Total Enrolled, Present Count, Absent Count, and Attendance Rate (%).
  - **Present vs. Absent Roster**: Easily see who attended and who is missing for any given date.
  - Interactive Daily Trend & Member Frequency visual charts.
  - One-click CSV export with timestamped filenames.
  - Admin manual check-in override.
- **🛡️ Enterprise Reliability & Duplicate Prevention**:
  - Automatic suppression of duplicate check-ins for the same person on the same day.
  - Fallback engine: Seamlessly runs on `face_recognition` (dlib) or built-in OpenCV multi-feature embeddings without crashing.
  - Persistent SQLite storage and audit logging in `logs/app.log`.

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│              Presentation Layer (Streamlit Light UI)           │
│  [📷 Live Kiosk]   [🧑 Face Enrollment]   [📊 Reports & Stats] │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│                   Application Service Layer                   │
│  ┌──────────────────┐ ┌───────────────────┐ ┌───────────────┐ │
│  │ face_service.py  │ │ enrollment_       │ │ attendance_   │ │
│  │ (Module 1: CV)   │ │ service.py (Mod 2)│ │ service.py    │ │
│  │ 128-d Embeddings │ │ CRUD & Matching   │ │ (Module 3)    │ │
│  └──────────────────┘ └───────────────────┘ └───────────────┘ │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│                        Data Layer                             │
│       SQLite (attendance.db)  │  /data/known_faces_images/    │
└───────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python -m streamlit run app.py
```

### 3. Run Automated Tests
```bash
python -m pytest
```

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `RECOGNITION_THRESHOLD` | `0.6` | Maximum Euclidean distance for match (Lower = stricter) |
| `DETECTION_MODEL` | `hog` | Face detection model (`hog` CPU-fast or `cnn` GPU-accurate) |
| `FRAME_SCALE` | `0.25` | Downscale factor for real-time video processing |
| `PROCESS_EVERY_N_FRAMES` | `2` | Process every Nth frame for ≥10 FPS performance |
| `FACE_BACKEND` | `auto` | `auto`, `face_recognition`, or `opencv` |

---

## 🧪 Test Suite

The project includes 13 unit and integration tests covering:
- Deterministic 128-d face embedding generation and distance calculation.
- Known face enrollment, multiple photos, single-face validation, and deletion.
- Duplicate attendance prevention on the same day.
- Daily roster calculation (Present list vs. Absent list).
- Manual attendance overrides and CSV export.

