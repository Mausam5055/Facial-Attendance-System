"""Module 3: Attendance Logging + Report View (PRD FR3.1-FR3.6)."""
from __future__ import annotations

import datetime

import pandas as pd

import config
from database.db_setup import get_connection, init_db

log = config.log


class AttendanceService:
    """Persistent attendance log with per-day duplicate prevention and analytics."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.DB_PATH
        init_db(self.db_path)

    # -- logging -----------------------------------------------------------
    @staticmethod
    def _today() -> str:
        return datetime.date.today().isoformat()

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    def is_marked_today(self, name: str, date: str | None = None) -> bool:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute("SELECT 1 FROM attendance WHERE name=? AND date=?",
                               (name, date or self._today())).fetchone()
            return row is not None
        finally:
            conn.close()

    def get_attendance_entry(self, name: str, date: str | None = None) -> dict | None:
        """Get the specific check-in record for a person on a given date."""
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT id, name, date, time, confidence FROM attendance WHERE name=? AND date=?",
                (name, date or self._today())
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def mark_attendance(self, name: str, confidence: float | None = None,
                        date: str | None = None, time: str | None = None) -> tuple[bool, str]:
        """Insert {name, date, time}.
        
        Returns: (is_new_marked: bool, message: str).
        'Unknown' names are never logged.
        """
        if not name or name == "Unknown":
            return False, "Unknown face detected — not recorded."
            
        date = date or self._today()
        time = time or self._now()
        conn = get_connection(self.db_path)
        try:
            existing = conn.execute("SELECT time FROM attendance WHERE name=? AND date=?",
                                    (name, date)).fetchone()
            if existing:
                log.info("Duplicate suppressed for '%s' on %s (logged at %s).", name, date, existing["time"])
                return False, f"Already marked present today at {existing['time']}"

            conn.execute("INSERT INTO attendance (name, date, time, confidence) "
                         "VALUES (?, ?, ?, ?)", (name, date, time, confidence))
            conn.commit()
            log.info("Marked present: %s @ %s %s.", name, date, time)
            return True, f"Successfully marked present at {time}"
        except Exception as exc:  # noqa: BLE001
            log.error("mark_attendance failed: %s", exc)
            return False, f"Database error: {exc}"
        finally:
            conn.close()

    def manual_mark(self, name: str, date: str | None = None,
                    time: str | None = None) -> tuple[bool, str]:
        """Admin override to manually mark a person present."""
        return self.mark_attendance(name=name, confidence=1.0, date=date, time=time)

    def delete_record(self, record_id: int) -> bool:
        """Delete an individual attendance log entry."""
        conn = get_connection(self.db_path)
        try:
            cur = conn.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
            conn.commit()
            success = (cur.rowcount > 0)
            if success:
                log.info("Deleted attendance record ID %d", record_id)
            return success
        finally:
            conn.close()

    # -- reporting & analytics -----------------------------------------------
    def get_report(self, date: str | None = None, start: str | None = None,
                   end: str | None = None, name: str | None = None) -> pd.DataFrame:
        query = "SELECT id, name, date, time, confidence FROM attendance WHERE 1=1"
        params: list = []
        if date:
            query += " AND date = ?"
            params.append(date)
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        if name and name != "All":
            query += " AND name = ?"
            params.append(name)
        query += " ORDER BY date DESC, time DESC"
        conn = get_connection(self.db_path)
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def get_daily_roster(self, enrolled_names: list[str],
                         date: str | None = None) -> dict:
        """Calculate Present vs Absent roster for a specific day."""
        target_date = date or self._today()
        report_df = self.get_report(date=target_date)
        
        present_names = set(report_df["name"].tolist()) if not report_df.empty else set()
        
        present_records = []
        if not report_df.empty:
            for _, row in report_df.iterrows():
                present_records.append({
                    "id": row["id"],
                    "name": row["name"],
                    "time": row["time"],
                    "confidence": row["confidence"]
                })
                
        all_enrolled_set = set(enrolled_names)
        absent_names = sorted(list(all_enrolled_set - present_names))
        
        total_enrolled = len(enrolled_names)
        present_count = len(present_records)
        rate = round((present_count / total_enrolled * 100), 1) if total_enrolled > 0 else 0.0
        
        return {
            "date": target_date,
            "total_enrolled": total_enrolled,
            "present_count": present_count,
            "absent_count": len(absent_names),
            "attendance_rate": rate,
            "present_records": present_records,
            "absent_names": absent_names
        }

    def summary(self, start: str | None = None, end: str | None = None) -> dict:
        df = self.get_report(start=start, end=end)
        total_records = len(df)
        unique_people = int(df["name"].nunique()) if total_records else 0
        per_person = df["name"].value_counts().to_dict() if total_records else {}
        per_day = df["date"].value_counts().sort_index().to_dict() if total_records else {}
        days = len(per_day)
        return {
            "total_records": total_records,
            "unique_people": unique_people,
            "days": days,
            "per_person": per_person,
            "per_day": per_day
        }

    def today_count(self) -> int:
        return len(self.get_report(date=self._today()))

    def export_csv(self, path: str, **filters) -> str:
        df = self.get_report(**filters)
        df.to_csv(path, index=False)
        log.info("Exported %d rows to %s.", len(df), path)
        return path

