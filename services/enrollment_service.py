"""Module 2: Enrollment + Recognition Matching (PRD FR2.1-FR2.6)."""
from __future__ import annotations

import datetime
import glob
import os
import pickle
import shutil

import cv2
import numpy as np

import config
from database.db_setup import get_connection, init_db
from services.face_service import FaceService

log = config.log


class EnrollmentService:
    """CRUD for known faces + best-match recognition."""

    def __init__(self, db_path: str | None = None, face_service: FaceService | None = None,
                 threshold: float | None = None):
        self.db_path = db_path or config.DB_PATH
        init_db(self.db_path)
        self.faces = face_service or FaceService()
        self.threshold = threshold if threshold is not None else config.RECOGNITION_THRESHOLD

    @staticmethod
    def _safe_dirname(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())

    def _get_user_dir(self, name: str) -> str:
        safe_name = self._safe_dirname(name)
        user_dir = os.path.join(config.KNOWN_FACES_IMAGE_DIR, safe_name)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    # -- Create / Update -------------------------------------------------
    def enroll(self, name: str, image_bgr: np.ndarray,
               save_image: bool = True, allow_multi_face: bool = False) -> int:
        """Enroll one reference photo for `name`. Returns #faces stored (1).

        Raises ValueError if no face is found, or if multiple faces are found
        when allow_multi_face is False.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("Person name must not be empty.")
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid or empty image provided.")

        results = self.faces.detect_and_encode(image_bgr)
        if not results:
            log.warning("Enroll failed for '%s': no face detected.", name)
            raise ValueError("No face detected in the image. Please face the camera directly with good lighting.")
        
        if len(results) > 1 and not allow_multi_face:
            log.warning("Enroll rejected for '%s': multiple faces detected (%d).", name, len(results))
            raise ValueError(f"Found {len(results)} faces in the photo. Please ensure only ONE person is in the frame.")

        box, encoding = results[0]  # First / primary face
        conn = get_connection(self.db_path)
        try:
            conn.execute("INSERT INTO known_faces (name, encoding) VALUES (?, ?)",
                         (name, pickle.dumps(np.asarray(encoding, dtype=np.float64))))
            conn.commit()
        finally:
            conn.close()

        if save_image:
            user_dir = self._get_user_dir(name)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = os.path.join(user_dir, f"ref_{stamp}.jpg")
            cv2.imwrite(full_path, image_bgr)
            
            # Save cropped avatar thumbnail for clean UI previews
            crop = self.faces.crop_face(image_bgr, box, pad_ratio=0.25)
            if crop is not None:
                avatar_path = os.path.join(user_dir, "avatar.jpg")
                cv2.imwrite(avatar_path, crop)

        log.info("Enrolled '%s' (backend=%s).", name, self.faces.backend)
        return 1

    # -- Read -------------------------------------------------------------
    def list_users(self) -> list[dict]:
        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT name, COUNT(*) AS photos, MIN(enrolled_on) AS enrolled_on "
                "FROM known_faces GROUP BY name ORDER BY name").fetchall()
            result = []
            for r in rows:
                item = dict(r)
                avatar_path = self.get_user_avatar_path(item["name"])
                item["avatar_path"] = avatar_path
                result.append(item)
            return result
        finally:
            conn.close()

    def count(self) -> int:
        conn = get_connection(self.db_path)
        try:
            return conn.execute("SELECT COUNT(DISTINCT name) FROM known_faces").fetchone()[0]
        finally:
            conn.close()

    def get_user_avatar_path(self, name: str) -> str | None:
        user_dir = self._get_user_dir(name)
        avatar = os.path.join(user_dir, "avatar.jpg")
        if os.path.exists(avatar):
            return avatar
        photos = glob.glob(os.path.join(user_dir, "ref_*.jpg"))
        return photos[0] if photos else None

    def get_user_photos(self, name: str) -> list[str]:
        user_dir = self._get_user_dir(name)
        return sorted(glob.glob(os.path.join(user_dir, "ref_*.jpg")), reverse=True)

    # -- Delete -----------------------------------------------------------
    def delete_user(self, name: str) -> int:
        conn = get_connection(self.db_path)
        try:
            cur = conn.execute("DELETE FROM known_faces WHERE name = ?", (name,))
            conn.commit()
            n = cur.rowcount
        finally:
            conn.close()

        # Clean up image files
        safe_name = self._safe_dirname(name)
        user_dir = os.path.join(config.KNOWN_FACES_IMAGE_DIR, safe_name)
        if os.path.exists(user_dir):
            try:
                shutil.rmtree(user_dir)
            except Exception as e:
                log.warning("Could not delete user dir %s: %s", user_dir, e)

        log.info("Deleted user '%s' (%d encoding records).", name, n)
        return n

    # -- Recognition -------------------------------------------------------
    def _load_known(self) -> tuple[list[str], np.ndarray | None]:
        conn = get_connection(self.db_path)
        try:
            rows = conn.execute("SELECT name, encoding FROM known_faces").fetchall()
        finally:
            conn.close()
        if not rows:
            return [], None
        names = [r["name"] for r in rows]
        matrix = np.stack([pickle.loads(bytes(r["encoding"])) for r in rows])
        return names, matrix

    def recognize(self, encoding: np.ndarray) -> tuple[str, float]:
        """Match one encoding -> (name or 'Unknown', best distance)."""
        names, matrix = self._load_known()
        if matrix is None:
            return "Unknown", float("inf")
        dists = np.linalg.norm(matrix - np.asarray(encoding, dtype=np.float64), axis=1)
        best = int(np.argmin(dists))
        best_dist = float(dists[best])
        if best_dist <= self.threshold:
            return names[best], best_dist
        return "Unknown", best_dist

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[tuple[str, tuple, float]]:
        """Full per-frame pipeline -> [(name, box, distance)]."""
        out = []
        for box, enc in self.faces.detect_and_encode(frame_bgr):
            name, dist = self.recognize(enc)
            out.append((name, box, dist))
        return out

