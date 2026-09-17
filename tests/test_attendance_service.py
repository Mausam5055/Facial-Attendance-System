"""Unit: Module 3 — logging, duplicate prevention, reporting (FR3.1-FR3.5)."""
from services.attendance_service import AttendanceService


def _svc(tmp_path):
    return AttendanceService(db_path=str(tmp_path / "a.db"))


def test_mark_and_duplicate_prevention(tmp_path):
    svc = _svc(tmp_path)
    ok1, msg1 = svc.mark_attendance("Alice", 0.3)
    assert ok1 is True
    assert "marked present" in msg1.lower()

    ok2, msg2 = svc.mark_attendance("Alice", 0.2)  # same day duplicate
    assert ok2 is False
    assert "already marked" in msg2.lower()
    assert len(svc.get_report()) == 1


def test_unknown_never_logged(tmp_path):
    svc = _svc(tmp_path)
    ok, msg = svc.mark_attendance("Unknown", 0.9)
    assert ok is False
    assert svc.get_report().empty


def test_report_filters_and_summary(tmp_path):
    svc = _svc(tmp_path)
    svc.mark_attendance("Alice", 0.3, date="2026-09-01", time="09:00:00")
    svc.mark_attendance("Bob", 0.4, date="2026-09-01", time="09:05:00")
    svc.mark_attendance("Alice", 0.3, date="2026-09-02", time="09:01:00")

    assert len(svc.get_report(date="2026-09-01")) == 2
    assert len(svc.get_report(name="Alice")) == 2
    assert len(svc.get_report(start="2026-09-02", end="2026-09-02")) == 1

    s = svc.summary()
    assert s["total_records"] == 3 and s["unique_people"] == 2 and s["days"] == 2
    assert s["per_person"]["Alice"] == 2


def test_daily_roster_present_and_absent(tmp_path):
    svc = _svc(tmp_path)
    svc.mark_attendance("Alice", 0.25, date="2026-09-17", time="08:30:00")
    
    roster = svc.get_daily_roster(enrolled_names=["Alice", "Bob", "Charlie"], date="2026-09-17")
    assert roster["total_enrolled"] == 3
    assert roster["present_count"] == 1
    assert roster["absent_count"] == 2
    assert roster["absent_names"] == ["Bob", "Charlie"]
    assert len(roster["present_records"]) == 1
    assert roster["present_records"][0]["name"] == "Alice"


def test_manual_mark_and_delete_record(tmp_path):
    svc = _svc(tmp_path)
    ok, _ = svc.manual_mark("David", date="2026-09-17", time="10:00:00")
    assert ok is True
    
    rep = svc.get_report(name="David")
    assert len(rep) == 1
    rec_id = int(rep.iloc[0]["id"])
    
    deleted = svc.delete_record(rec_id)
    assert deleted is True
    assert len(svc.get_report(name="David")) == 0


def test_export_csv(tmp_path):
    svc = _svc(tmp_path)
    svc.mark_attendance("Alice", 0.3)
    out = str(tmp_path / "rep.csv")
    svc.export_csv(out)
    with open(out) as f:
        assert "Alice" in f.read()

