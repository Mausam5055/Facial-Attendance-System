"""SQLite setup for known_faces + attendance tables (PRD Sec. 10)."""
import sqlite3

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS known_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL,
    enrolled_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_known_faces_name ON known_faces(name);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date DATE NOT NULL,
    time TIME NOT NULL,
    confidence FLOAT,
    UNIQUE(name, date)
);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_attendance_name ON attendance(name);
"""


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> str:
    """Create tables if they don't exist. Returns the db path used."""
    path = db_path or config.DB_PATH
    conn = get_connection(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    config.log.info("Database initialised at %s", path)
    return path


if __name__ == "__main__":
    init_db()
    print(f"DB ready at {config.DB_PATH}")
