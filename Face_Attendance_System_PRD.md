# Product Requirements Document (PRD)
## Face Recognition–Based Attendance System

**Project Type:** Computer Vision Application
**Prepared for:** VITyarthi — Build Your Own Project
**Build Tool:** Google Antigravity (AI coding agent)
**Version:** 1.0

---

## 1. Overview

A desktop/web-based attendance system that uses face recognition to automatically identify enrolled users from a live camera feed or uploaded image, and logs their attendance with a timestamp. The system replaces manual roll-call with automated recognition and provides a reporting dashboard to view attendance history.

---

## 2. Problem Statement

Manual attendance marking (roll call, sign-in sheets) is time-consuming, prone to proxy attendance (one person marking for another), and difficult to audit or analyze. Small classrooms, training sessions, and offices need a lightweight, low-cost system that can automatically recognize known individuals and log their presence without manual intervention or expensive hardware (no biometric scanners required — a standard webcam is enough).

---

## 3. Objectives

- Automatically detect and recognize enrolled faces from a webcam feed or image.
- Allow an admin to enroll new users by capturing/uploading reference face images.
- Prevent duplicate attendance entries for the same person on the same day.
- Store attendance records persistently and allow filtering/reporting by date, person, or range.
- Provide clear, real-time visual feedback (bounding box + name) during recognition.

---

## 4. Target Users

- **Admin/Instructor** — enrolls users, views/exports reports, manages the known-faces database.
- **End User (student/employee)** — walks in front of the camera; system marks them present automatically.

---

## 5. Scope

### In Scope
- Face detection and encoding using the `face_recognition` library (built on dlib).
- Enrollment flow (add new face + name to the known-faces store).
- Real-time recognition from webcam OR batch recognition from an uploaded photo/group photo.
- Attendance logging to CSV/SQLite with duplicate-prevention per day.
- A simple report view (table + basic filters, optional chart).

### Out of Scope (mention in report as "Future Enhancements")
- Multi-camera / multi-room support.
- Cloud deployment, authentication/login system for admin.
- Liveness detection (anti-spoofing against photos held up to the camera).
- Mobile app version.

---

## 6. Functional Requirements (Modules)

### **Module 1: Face Detection + Encoding**
**Purpose:** Core CV engine — detect faces in an image/frame and generate a 128-d encoding vector for each face.

| Requirement | Detail |
|---|---|
| FR1.1 | Detect all faces in a given frame/image using `face_recognition.face_locations()`. |
| FR1.2 | Generate a 128-dimension encoding per detected face using `face_recognition.face_encodings()`. |
| FR1.3 | Support both HOG (fast, CPU-friendly) and CNN (more accurate, slower) detection models, selectable via config. |
| FR1.4 | Handle frames with zero faces, one face, or multiple faces gracefully (no crash). |
| FR1.5 | Downscale frames before processing (e.g., 0.25x) to maintain real-time performance, then scale bounding boxes back up for display. |

**Output:** List of `(bounding_box, encoding_vector)` per frame.

---

### **Module 2: Enrollment + Recognition Matching**
**Purpose:** CRUD-style management of known faces, and matching live faces against them.

| Requirement | Detail |
|---|---|
| FR2.1 | **Enroll (Create):** Admin provides a name + one or more face images (webcam capture or file upload). System computes and stores the encoding(s) tagged with that name. |
| FR2.2 | **List (Read):** Admin can view all currently enrolled users. |
| FR2.3 | **Delete:** Admin can remove an enrolled user from the known-faces store. |
| FR2.4 | **Update:** Admin can re-enroll (add another reference photo) to improve accuracy for a person. |
| FR2.5 | **Recognition:** For each detected face's encoding, compute Euclidean distance against all known encodings using `face_recognition.compare_faces()` / `face_distance()`; assign the best match below a configurable threshold (default 0.6), else label "Unknown." |
| FR2.6 | Known-faces store persists encodings + names on disk (pickle/JSON file or SQLite table) so it survives app restarts. |

**Output:** Per detected face → `{name, confidence/distance, bounding_box}` or `Unknown`.

---

### **Module 3: Attendance Logging + Report View**
**Purpose:** Persist recognition events as attendance records and expose them via a report UI.

| Requirement | Detail |
|---|---|
| FR3.1 | On a successful recognition (confidence within threshold), log `{name, date, time}` to storage (CSV or SQLite). |
| FR3.2 | **Duplicate prevention:** if the person is already marked present today, do not create a second entry (or optionally log a "last seen" timestamp update instead). |
| FR3.3 | Report view: table of attendance records, filterable by date, date range, or person name. |
| FR3.4 | Summary stats: total present today, attendance % per person over a date range. |
| FR3.5 | Export filtered report to CSV. |
| FR3.6 | (Optional/bonus) Simple bar chart of attendance count per day using matplotlib. |

**Output:** Attendance database/CSV + a viewable/exportable report.

---

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | Recognition loop should process at ≥10 FPS on a standard laptop CPU (achieved via frame downscaling + processing every 2nd–3rd frame). |
| **Reliability** | System should not crash on: no face detected, poor lighting, camera disconnect, or corrupted image upload — all handled with try/except and user-facing messages. |
| **Usability** | Simple UI (CLI menu or Streamlit web UI) with clear prompts: Enroll / Start Recognition / View Report. Visual bounding box + name label overlay during recognition. |
| **Security** | Known-faces data and attendance logs stored locally only; no images transmitted externally. Basic file-path validation on uploads. |
| **Scalability** | Design should support at least 50 enrolled users without significant recognition slowdown (encoding comparison is O(n), acceptable at this scale). |
| **Maintainability** | Modular code: separate files for detection/encoding, enrollment/matching, and attendance/reporting (matches the 3-module structure). Config values (threshold, detection model, file paths) centralized in a `config.py`. |
| **Logging/Monitoring** | Application-level logging (not just attendance logging) — log errors, recognition events, and enrollment actions to a `logs/app.log` file using Python's `logging` module. |

*(This satisfies VITyarthi's requirement of ≥4 non-functional requirements — 7 are listed here for depth.)*

---

## 8. Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Face Detection/Encoding | `face_recognition` (dlib-based) |
| Image/Video Handling | OpenCV (`opencv-python`) |
| Data Storage | SQLite (`sqlite3`, built-in) — recommended over CSV for proper CRUD + query support |
| UI | Streamlit (fast to build, good for demo) — CLI version as fallback |
| Reporting/Charts | pandas + matplotlib |
| Testing | `pytest` for unit tests on matching logic and DB operations |
| Version Control | Git + GitHub |

> **Why SQLite over CSV:** Easier duplicate-checking (`SELECT` queries), atomic writes, and cleanly supports the "CRUD operations" functional example from the assignment brief.

---

## 9. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Presentation Layer                   │
│         (Streamlit UI: Enroll | Recognize | Report)       │
└───────────────────────┬───────────────────────────────────┘
                         │
┌───────────────────────▼───────────────────────────────────┐
│                   Application/Service Layer                │
│  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────┐ │
│  │ face_service.py │ │ enrollment_       │ │ attendance_ │ │
│  │ (Module 1)      │ │ service.py        │ │ service.py  │ │
│  │ detect + encode │ │ (Module 2)        │ │ (Module 3)  │ │
│  │                 │ │ enroll + match    │ │ log + report│ │
│  └─────────────────┘ └──────────────────┘ └─────────────┘ │
└───────────────────────┬───────────────────────────────────┘
                         │
┌───────────────────────▼───────────────────────────────────┐
│                       Data Layer                            │
│   known_faces.db (encodings table)  │  attendance.db (logs) │
│   /known_faces_images/ (raw reference photos)                │
└───────────────────────────────────────────────────────────┘
```

---

## 10. Database Schema

**Table: `known_faces`**
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| name | TEXT | Enrolled person's name |
| encoding | BLOB | Pickled 128-d numpy array |
| enrolled_on | TIMESTAMP | Enrollment date |

**Table: `attendance`**
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| name | TEXT | FK-like reference to known_faces.name |
| date | DATE | Attendance date |
| time | TIME | First-seen time |
| confidence | FLOAT | Match distance score |

*(ER Diagram: two tables, no direct FK constraint needed since `name` is the natural join key — mention as a simplification/future improvement to use `face_id` FK in your report.)*

---

## 11. Workflow / Process Flow

1. **Enrollment flow:**
   Admin opens "Enroll" → enters name → captures/uploads photo → Module 1 detects+encodes face → Module 2 saves encoding to `known_faces` DB.

2. **Recognition/attendance flow:**
   User stands in front of camera → Module 1 detects face(s) each frame → generates encoding → Module 2 compares against all `known_faces` encodings → best match found → Module 3 checks if already logged today → if not, inserts record into `attendance` DB → UI shows green box + name + "Marked Present ✅".

3. **Reporting flow:**
   Admin opens "Report" → selects date/range/person filter → Module 3 queries `attendance` DB → displays table + summary stats + optional chart → export to CSV on request.

---

## 12. Suggested Folder Structure (for GitHub repo)

```
face-attendance-system/
├── README.md
├── statement.md
├── requirements.txt
├── config.py
├── app.py                     # Streamlit entry point
├── services/
│   ├── face_service.py        # Module 1: detection + encoding
│   ├── enrollment_service.py  # Module 2: enroll + match
│   └── attendance_service.py  # Module 3: log + report
├── database/
│   ├── db_setup.py            # creates tables
│   └── attendance.db
├── data/
│   └── known_faces_images/    # raw reference photos
├── tests/
│   ├── test_face_service.py
│   ├── test_enrollment_service.py
│   └── test_attendance_service.py
├── logs/
│   └── app.log
└── docs/
    ├── architecture_diagram.png
    ├── er_diagram.png
    ├── use_case_diagram.png
    └── sequence_diagram.png
```
This gives you **7 core Python files** — comfortably within the 5–10 module requirement.

---

## 13. Testing Plan

| Test | Type | Description |
|---|---|---|
| Encoding consistency | Unit | Same face image → same encoding vector each run. |
| Match threshold | Unit | Known face distance < 0.6; unrelated face distance > 0.6. |
| Duplicate prevention | Unit | Second recognition of the same person on the same day does not create a new row. |
| No-face frame | Edge case | Passing a blank/no-face image doesn't crash the pipeline. |
| DB CRUD | Integration | Enroll → appears in known_faces; delete → removed; attendance insert → retrievable via report query. |

---

## 14. Deliverables Checklist (mapped to VITyarthi requirements)

- [ ] GitHub repo with README.md, statement.md, source code (folder structure above)
- [ ] Problem Statement, Objectives, Functional & Non-functional Requirements (this doc → report)
- [ ] System Architecture Diagram (Section 9)
- [ ] Process/Workflow Diagram (Section 11)
- [ ] Use Case Diagram (Admin: enroll/view report; User: get recognized)
- [ ] Class/Component Diagram (3 service classes + DB layer)
- [ ] Sequence Diagram (recognition flow, per Section 11 step 2)
- [ ] ER Diagram (Section 10)
- [ ] PDF Project Report (cover page → references, per VITyarthi's 15-point structure)
- [ ] Working demo + screenshots

---

## 15. Notes for Building in Antigravity

When prompting Antigravity to scaffold this, feed it section-by-section rather than the whole PRD at once for best results:
1. Start with folder structure + `requirements.txt` + `config.py`.
2. Build `face_service.py` (Module 1) first and test it standalone with a sample image.
3. Build `enrollment_service.py` + DB setup (Module 2), test enroll/list/delete.
4. Build `attendance_service.py` (Module 3) with duplicate-check logic.
5. Wire everything into `app.py` (Streamlit UI) last, once each service works independently.
6. Ask Antigravity to generate the unit tests in `tests/` after each service is stable, not all at once at the end.

This incremental order avoids one giant unreviewable code dump and matches how the rubric rewards "modular and clean implementation."
