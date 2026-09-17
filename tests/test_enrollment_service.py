"""Unit + integration: Module 2 — enroll/list/match/delete (FR2.1-FR2.6)."""
import numpy as np

from services.enrollment_service import EnrollmentService
from services.face_service import FaceService


def _svc(tmp_path):
    fs = FaceService(backend="opencv")
    return EnrollmentService(db_path=str(tmp_path / "t.db"), face_service=fs)


def _fake_detect(vec):
    return [((10, 50, 50, 10), np.asarray(vec, dtype=np.float64))]


def test_enroll_list_recognize_delete(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    alice = np.ones(128) / np.linalg.norm(np.ones(128))
    monkeypatch.setattr(svc.faces, "detect_and_encode", lambda _img: _fake_detect(alice))
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    assert svc.enroll("Alice", img, save_image=False) == 1
    assert [u["name"] for u in svc.list_users()] == ["Alice"]
    assert svc.count() == 1

    name, dist = svc.recognize(alice)  # exact match
    assert name == "Alice" and dist < 0.6

    far = np.zeros(128)  # distant vector -> Unknown
    far[0] = 5.0
    assert svc.recognize(far)[0] == "Unknown"

    assert svc.delete_user("Alice") == 1
    assert svc.list_users() == []
    assert svc.count() == 0


def test_enroll_no_face_raises(tmp_path):
    svc = _svc(tmp_path)
    try:
        svc.enroll("Bob", np.zeros((50, 50, 3), dtype=np.uint8), save_image=False)
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "No face" in str(e)


def test_enroll_multi_face_raises(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    # Mock finding 2 faces
    monkeypatch.setattr(
        svc.faces,
        "detect_and_encode",
        lambda _img: [
            ((10, 50, 50, 10), np.ones(128)),
            ((60, 100, 100, 60), np.ones(128)),
        ]
    )
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    try:
        svc.enroll("GroupPhoto", img, save_image=False, allow_multi_face=False)
        raise AssertionError("should have rejected multi-face photo")
    except ValueError as e:
        assert "Found 2 faces" in str(e)

