"""Case history in SQLite (stdlib sqlite3): cases survive restarts. One row per case: the evidence record, the activity
log and, for uploads, the path of the stored array."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class CaseStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL, rec_id TEXT,
                record TEXT NOT NULL, activity TEXT NOT NULL, array_path TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def put(self, record: dict, activity: list, array_path: str | None = None) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (record["case_id"], record["created_at"], record["source"], record["input"]["rec_id"],
                       json.dumps(record), json.dumps(activity), array_path))

    def get(self, case_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT record, activity, array_path FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            return None
        return {"record": json.loads(row[0]), "activity": json.loads(row[1]), "array_path": row[2]}

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT case_id, created_at, source, rec_id FROM cases ORDER BY created_at DESC LIMIT ?",
                             (limit,)).fetchall()
        return [{"case_id": r[0], "created_at": r[1], "source": r[2], "rec_id": r[3]} for r in rows]
