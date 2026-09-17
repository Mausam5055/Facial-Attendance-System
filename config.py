"""Central configuration for the Face Recognition Attendance System (PRD Sec. 6-8)."""
import logging
import os
from pathlib import Path

# --- Base paths ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "database" / "attendance.db")
KNOWN_FACES_IMAGE_DIR = str(BASE_DIR / "data" / "known_faces_images")
LOG_FILE = str(BASE_DIR / "logs" / "app.log")
ASSETS_DIR = str(BASE_DIR / "assets")

# --- Face recognition settings (FR1.3, FR1.5, FR2.5) ---
RECOGNITION_THRESHOLD: float = 0.6   # max Euclidean distance for a match
DETECTION_MODEL: str = "hog"          # "hog" (fast/CPU) or "cnn" (accurate/slow)
FRAME_SCALE: float = 0.25             # downscale factor before processing
PROCESS_EVERY_N_FRAMES: int = 2       # process every 2nd frame for >=10 FPS
FACE_BACKEND: str = "auto"            # "auto" | "face_recognition" | "opencv"

# --- Attendance settings (FR3.2) ---
UPDATE_LAST_SEEN: bool = False  # if True, update time on re-recognition same day

# Ensure necessary directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(KNOWN_FACES_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def setup_logging() -> logging.Logger:
    """Configure application-level logging to logs/app.log + console."""
    logger = logging.getLogger("attendance")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = setup_logging()

