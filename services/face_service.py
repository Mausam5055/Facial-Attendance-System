"""Module 1: Face Detection + Encoding (PRD FR1.1-FR1.5).

Primary backend: `face_recognition` (dlib, 128-d encodings) when installed.
Fallback backend: OpenCV Haar cascade + normalized 128-d embedding so the
app runs on any machine (e.g. Python 3.12+ where dlib wheels are missing).

Public output per frame: list of (bounding_box, encoding_vector) where
bounding_box = (top, right, bottom, left) in ORIGINAL image coordinates.
"""
from __future__ import annotations

import cv2
import numpy as np

import config

log = config.log

try:
    import face_recognition  # type: ignore
    _HAS_FR = True
except Exception as exc:  # noqa: BLE001 - optional dependency
    face_recognition = None  # type: ignore
    _HAS_FR = False
    log.warning("face_recognition not available (%s); using OpenCV fallback.", exc)


def active_backend(override: str | None = None) -> str:
    """Resolve which backend to use: 'face_recognition' or 'opencv'."""
    want = (override or config.FACE_BACKEND).lower()
    if want == "face_recognition":
        if not _HAS_FR:
            log.warning("face_recognition requested but not installed; falling back to opencv.")
            return "opencv"
        return "face_recognition"
    if want == "opencv":
        return "opencv"
    return "face_recognition" if _HAS_FR else "opencv"  # "auto"


def is_fr_available() -> bool:
    """Check if face_recognition (dlib) library is available."""
    return _HAS_FR



class FaceService:
    """Detection + encoding engine (Module 1)."""

    def __init__(self, model: str | None = None, scale: float | None = None,
                 backend: str | None = None):
        self.model = (model or config.DETECTION_MODEL).lower()  # hog | cnn
        self.scale = scale or config.FRAME_SCALE
        self.backend = active_backend(backend)
        self._haar = None
        if self.backend == "opencv":
            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError("This opencv build lacks CascadeClassifier. "
                                   "Install 'opencv-python<5' (pip install 'opencv-python<5').")
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade)
            if self._haar.empty():
                raise RuntimeError(f"Could not load Haar cascade from {cascade}")
        log.info("FaceService initialized: backend=%s model=%s scale=%s",
                 self.backend, self.model, self.scale)

    # -- public API -----------------------------------------------------
    def detect_and_encode(self, frame_bgr: np.ndarray) -> list[tuple[tuple, np.ndarray]]:
        """Detect faces and encode each one.

        Returns: [((top, right, bottom, left), 128-d vector), ...].
        Never raises on zero faces / bad frames — returns [] instead.
        """
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return []
            if self.backend == "face_recognition":
                return self._detect_fr(frame_bgr)
            return self._detect_opencv(frame_bgr)
        except Exception as exc:  # noqa: BLE001 - reliability NFR: never crash
            log.error("detect_and_encode failed: %s", exc)
            return []

    # -- face_recognition backend --------------------------------------
    def _detect_fr(self, frame_bgr: np.ndarray):
        small = cv2.resize(frame_bgr, (0, 0), fx=self.scale, fy=self.scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        assert face_recognition is not None
        locations = face_recognition.face_locations(rgb, model=self.model)
        encodings = face_recognition.face_encodings(rgb, locations)
        out = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            box = (int(top / self.scale), int(right / self.scale),
                   int(bottom / self.scale), int(left / self.scale))
            out.append((box, np.asarray(enc, dtype=np.float64)))
        return out

    # -- OpenCV fallback backend ----------------------------------------
    def _detect_opencv(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=self.scale, fy=self.scale)
        faces = self._haar.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        h, w = frame_bgr.shape[:2]
        out = []
        for (x, y, fw, fh) in faces:
            top = int(max(0, y / self.scale))
            right = int(min(w, (x + fw) / self.scale))
            bottom = int(min(h, (y + fh) / self.scale))
            left = int(max(0, x / self.scale))
            crop = gray[top:bottom, left:right]
            out.append(((top, right, bottom, left), self._embed(crop)))
        return out

    @staticmethod
    def _embed(face_gray: np.ndarray) -> np.ndarray:
        """Deterministic 128-d unit-length embedding from a face crop.

        Combines normalized thumbnail intensity grid (64-d) with spatial
        gradient orientation features (64-d), L2-normalized.
        """
        if face_gray is None or face_gray.size == 0:
            face_gray = np.zeros((64, 64), dtype=np.uint8)
        
        # Standardize size & equalize lighting
        std_face = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
        std_face = cv2.equalizeHist(std_face)
        
        # 1. Intensity grid representation (8x8 = 64 features)
        thumb = cv2.resize(std_face, (8, 8), interpolation=cv2.INTER_AREA)
        feat_intensity = thumb.astype(np.float64).flatten() / 255.0
        
        # 2. Gradient orientation / edge map representation (8x8 = 64 features)
        gx = cv2.Sobel(std_face, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(std_face, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_thumb = cv2.resize(mag, (8, 8), interpolation=cv2.INTER_AREA)
        feat_grad = mag_thumb.flatten()
        grad_norm = np.linalg.norm(feat_grad)
        if grad_norm > 1e-9:
            feat_grad = feat_grad / grad_norm
            
        vec = np.concatenate([feat_intensity, feat_grad])
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.astype(np.float64)  # shape (128,)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def face_distance(known: np.ndarray, candidate: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(known) - np.asarray(candidate)))

    @staticmethod
    def crop_face(frame_bgr: np.ndarray, box: tuple, pad_ratio: float = 0.2) -> np.ndarray | None:
        """Extract a cropped face from a frame with optional boundary padding."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        top, right, bottom, left = box
        h, w = frame_bgr.shape[:2]
        
        box_w = right - left
        box_h = bottom - top
        pad_x = int(box_w * pad_ratio)
        pad_y = int(box_h * pad_ratio)
        
        y1 = max(0, top - pad_y)
        y2 = min(h, bottom + pad_y)
        x1 = max(0, left - pad_x)
        x2 = min(w, right + pad_x)
        
        crop = frame_bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    @staticmethod
    def draw_boxes(frame_bgr: np.ndarray, items: list[tuple[str, tuple, float]],
                   marked_status: dict[str, bool] | None = None) -> np.ndarray:
        """Overlay sleek, modern bounding boxes + badge labels.
        
        items = [(name, box, dist)].
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
            
        out = frame_bgr.copy()
        marked_status = marked_status or {}
        
        for name, (top, right, bottom, left), dist in items:
            is_unknown = (name == "Unknown")
            is_marked = marked_status.get(name, False)
            
            # Palette: Emerald for Recognized/Marked, Amber for Recognized but already recorded, Rose for Unknown
            if is_unknown:
                border_color = (68, 68, 239)   # Rose Red (#EF4444) in BGR
                bg_color = (68, 68, 239)
                label_text = "Unknown"
            elif is_marked:
                border_color = (129, 185, 16)  # Emerald Green (#10B981) in BGR
                bg_color = (129, 185, 16)
                label_text = f"✓ {name}"
            else:
                border_color = (235, 168, 52)  # Sky/Indigo (#34A8EB) in BGR
                bg_color = (235, 168, 52)
                label_text = f"{name}"
                
            if dist is not None and dist != float("inf") and not is_unknown:
                score_pct = max(0, int((1.0 - min(dist, 1.0)) * 100))
                sub_text = f"{score_pct}% match"
            else:
                sub_text = ""

            # Bounding box with clean corner accents
            cv2.rectangle(out, (left, top), (right, bottom), border_color, 2, cv2.LINE_AA)
            
            # Corner accents
            c_len = min(20, (right - left) // 4, (bottom - top) // 4)
            if c_len > 4:
                # Top-left
                cv2.line(out, (left, top), (left + c_len, top), border_color, 4, cv2.LINE_AA)
                cv2.line(out, (left, top), (left, top + c_len), border_color, 4, cv2.LINE_AA)
                # Top-right
                cv2.line(out, (right, top), (right - c_len, top), border_color, 4, cv2.LINE_AA)
                cv2.line(out, (right, top), (right, top + c_len), border_color, 4, cv2.LINE_AA)
                # Bottom-left
                cv2.line(out, (left, bottom), (left + c_len, bottom), border_color, 4, cv2.LINE_AA)
                cv2.line(out, (left, bottom), (left, bottom - c_len), border_color, 4, cv2.LINE_AA)
                # Bottom-right
                cv2.line(out, (right, bottom), (right - c_len, bottom), border_color, 4, cv2.LINE_AA)
                cv2.line(out, (right, bottom), (right, bottom - c_len), border_color, 4, cv2.LINE_AA)

            # Label pill background above or below box
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            thickness = 1
            
            full_display = f"{label_text} {f'({sub_text})' if sub_text else ''}".strip()
            (tw, th), baseline = cv2.getTextSize(full_display, font, font_scale, thickness)
            
            badge_h = th + baseline + 12
            badge_w = tw + 16
            
            badge_top = max(0, top - badge_h)
            badge_bottom = badge_top + badge_h
            badge_right = min(out.shape[1], left + badge_w)
            
            # Draw solid badge background with soft dark shadow
            cv2.rectangle(out, (left + 1, badge_top + 1), (badge_right + 1, badge_bottom + 1),
                          (30, 30, 30), cv2.FILLED)
            cv2.rectangle(out, (left, badge_top), (badge_right, badge_bottom),
                          bg_color, cv2.FILLED)
            
            # White text
            text_x = left + 8
            text_y = badge_top + th + 6
            cv2.putText(out, full_display, (text_x, text_y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
                        
        return out

