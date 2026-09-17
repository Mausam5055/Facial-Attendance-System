# Build Your Own Project — Project Statement
## Face Recognition–Based Attendance System

**Problem:** Manual attendance (roll call, sign-in sheets) is slow, allows proxy
attendance, and is hard to audit.

**Solution:** A computer-vision app that detects and recognizes enrolled faces from a
webcam or uploaded photo and logs `{name, date, time}` automatically, with
duplicate-prevention per day and a filterable report dashboard with CSV export.

**Modules (per PRD):**
1. Face Detection + Encoding — `face_recognition` (HOG/CNN) with OpenCV fallback;
   0.25x downscaling + every-2nd-frame processing for ≥10 FPS.
2. Enrollment + Recognition Matching — enroll/list/delete/re-enroll; Euclidean
   matching with 0.6 threshold; SQLite-persisted encodings.
3. Attendance Logging + Report View — deduped logging, filters, summary stats,
   CSV export, per-day bar chart.

**Stack:** Python 3.10+, OpenCV, Streamlit, SQLite, pandas, matplotlib, pytest.
**Non-functional:** performance, reliability (never crash on no-face/camera loss),
usability, local-only security, 50-user scalability, modular maintainability,
file logging.
