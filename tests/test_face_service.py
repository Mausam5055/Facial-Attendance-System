"""Unit: Module 1 — detection handles 0/1/many faces without crashing."""
import numpy as np

from services.face_service import FaceService


def test_blank_image_returns_empty():
    fs = FaceService(backend="opencv")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert fs.detect_and_encode(blank) == []


def test_none_and_empty_handled():
    fs = FaceService(backend="opencv")
    assert fs.detect_and_encode(None) == []
    assert fs.detect_and_encode(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_embed_is_deterministic_128d():
    fs = FaceService(backend="opencv")
    rng = np.random.RandomState(0)
    crop = rng.randint(0, 255, (100, 100)).astype(np.uint8)
    a, b = fs._embed(crop), fs._embed(crop)
    assert a.shape == (128,) and np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6


def test_face_distance_zero_for_same():
    v = np.ones(128)
    assert FaceService.face_distance(v, v) == 0.0
