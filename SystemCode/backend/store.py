"""Case history in SQLite (stdlib sqlite3): cases survive restarts. One row per case: the evidence record, the activity
log, for uploads the path of the stored array, and (App-2a.3) the case it re-runs, if any."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

COLUMNS = ("case_id", "created_at", "source", "rec_id", "record", "activity", "array_path", "rerun_of")


class CaseStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL, rec_id TEXT,
                record TEXT NOT NULL, activity TEXT NOT NULL, array_path TEXT)""")
            if "rerun_of" not in [r[1] for r in c.execute("PRAGMA table_info(cases)")]:
                c.execute("ALTER TABLE cases ADD COLUMN rerun_of TEXT")  # databases created before App-2a.3
            c.execute("CREATE TABLE IF NOT EXISTS explanations (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")  # App-3
            c.execute("""CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
                created_at TEXT NOT NULL, payload TEXT NOT NULL)""")  # App-7: every assistant question and response
            migrate(c)


    def _conn(self):
        return sqlite3.connect(self.path)

    def put(self, record: dict, activity: list, array_path: str | None = None, rerun_of: str | None = None) -> None:
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO cases ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                      (record["case_id"], record["created_at"], record["source"], record["input"]["rec_id"],
                       json.dumps(record), json.dumps(activity), array_path, rerun_of))

    def get(self, case_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT record, activity, array_path, rerun_of FROM cases WHERE case_id = ?",
                            (case_id,)).fetchone()
        if row is None:
            return None
        return {"record": json.loads(row[0]), "activity": json.loads(row[1]), "array_path": row[2], "rerun_of": row[3]}

    def put_explanation(self, case_id: str, payload: dict) -> None:
        """The latest explanation of a case (a regeneration replaces it; every call is also logged to the JSONL file)."""
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO explanations VALUES (?, ?)", (case_id, json.dumps(payload, ensure_ascii=False)))

    def get_explanation(self, case_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM explanations WHERE case_id = ?", (case_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put_question(self, case_id: str, created_at: str, payload: dict) -> int:
        """One assistant question and its response (App-7); returns its id. Rejected drafts are never stored."""
        with self._conn() as c:
            cur = c.execute("INSERT INTO questions (case_id, created_at, payload) VALUES (?, ?, ?)",
                            (case_id, created_at, json.dumps(payload, ensure_ascii=False)))
            return int(cur.lastrowid)

    def questions(self, case_id: str) -> list[dict]:
        """The assistant questions of a case, oldest first."""
        with self._conn() as c:
            rows = c.execute("SELECT id, created_at, payload FROM questions WHERE case_id = ? ORDER BY id",
                             (case_id,)).fetchall()
        return [{"id": r[0], "case_id": case_id, "created_at": r[1], **json.loads(r[2])} for r in rows]

    def questions_with_conversation(self, case_id: str) -> list[dict]:
        """Every stored question of a case with its conversation id and mode (App-9), oldest first."""
        with self._conn() as c:
            rows = c.execute("SELECT id, created_at, payload, conversation_id, mode FROM questions WHERE case_id = ? ORDER BY id",
                             (case_id,)).fetchall()
        return [{"id": r[0], "case_id": case_id, "created_at": r[1], **json.loads(r[2]), "conversation_id": r[3], "mode": r[4]}
                for r in rows]

    # ---------------------------------------------------------------------------------------- App-9: conversations
    def conversation(self, case_id: str) -> int:
        """The case's current conversation id (0 = the conversation every case starts in; no row needed)."""
        with self._conn() as c:
            row = c.execute("SELECT MAX(id) FROM conversations WHERE case_id = ?", (case_id,)).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def new_conversation(self, case_id: str, created_at: str) -> int:
        """Start a new conversation for the case: later questions no longer see earlier turns as context. Nothing is
        deleted (the diagnosis, the stored questions and any notes stay)."""
        with self._conn() as c:
            return int(c.execute("INSERT INTO conversations (case_id, created_at) VALUES (?, ?)", (case_id, created_at)).lastrowid)

    def put_turn(self, case_id: str, conversation_id: int, created_at: str, payload: dict) -> int:
        """One conversation-mode question and its response (only what the user saw, plus the compact tool audit)."""
        with self._conn() as c:
            cur = c.execute("INSERT INTO questions (case_id, created_at, payload, conversation_id, mode) VALUES (?, ?, ?, ?, ?)",
                            (case_id, created_at, json.dumps(payload, ensure_ascii=False), conversation_id, "conversation"))
            return int(cur.lastrowid)

    def turns(self, case_id: str, conversation_id: int) -> list[dict]:
        """The conversation-mode turns of ONE case and ONE conversation, oldest first."""
        with self._conn() as c:
            rows = c.execute("SELECT id, created_at, payload FROM questions WHERE case_id = ? AND conversation_id = ? "
                             "AND mode = 'conversation' ORDER BY id", (case_id, conversation_id)).fetchall()
        return [{"id": r[0], "case_id": case_id, "created_at": r[1], "conversation_id": conversation_id, **json.loads(r[2])}
                for r in rows]

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT case_id, created_at, source, rec_id, rerun_of FROM cases ORDER BY created_at DESC "
                             "LIMIT ?", (limit,)).fetchall()
        return [{"case_id": r[0], "created_at": r[1], "source": r[2], "rec_id": r[3], "rerun_of": r[4]} for r in rows]


# ------------------------------------------------------------------------------------------------------ migrations
# Numbered, applied once each, recorded in schema_migrations; existing databases are upgraded in place and existing rows
# keep their meaning (App-7 questions become mode 'single', conversation 0).
MIGRATIONS = {
    1: ["CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, created_at TEXT NOT NULL)",
        "ALTER TABLE questions ADD COLUMN conversation_id INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE questions ADD COLUMN mode TEXT NOT NULL DEFAULT 'single'",
        "CREATE INDEX IF NOT EXISTS questions_case_conv ON questions (case_id, conversation_id, id)",
        "CREATE INDEX IF NOT EXISTS conversations_case ON conversations (case_id, id)"],
}


def migrate(c) -> list[int]:
    """Apply every migration not yet recorded; returns the versions applied now."""
    c.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    done = {r[0] for r in c.execute("SELECT version FROM schema_migrations")}
    applied = []
    for v in sorted(MIGRATIONS):
        if v in done:
            continue
        for stmt in MIGRATIONS[v]:
            c.execute(stmt)
        c.execute("INSERT INTO schema_migrations VALUES (?, datetime('now'))", (v,))
        applied.append(v)
    return applied
