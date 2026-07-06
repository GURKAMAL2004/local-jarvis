"""SQLite-backed memory: chat sessions/messages per persona, and freeform
per-contact notes the agent accumulates over time (used by chat personas now,
and by the WhatsApp customer-chat feature in Phase 5)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from deskbot import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS contact_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_contact_notes_contact ON contact_notes(contact_id);
"""


@dataclass
class Message:
    role: str
    content: str


class Memory:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or paths.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # --- sessions -----------------------------------------------------

    def latest_session_id(self, persona: str) -> int | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE persona = ? ORDER BY id DESC LIMIT 1",
                (persona,),
            ).fetchone()
            return row[0] if row else None

    def create_session(self, persona: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("INSERT INTO sessions (persona) VALUES (?)", (persona,))
            return cur.lastrowid

    def get_or_create_session(self, persona: str, resume: bool = True) -> int:
        if resume:
            existing = self.latest_session_id(persona)
            if existing is not None:
                return existing
        return self.create_session(persona)

    def touch_session(self, session_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET last_active_at = datetime('now') WHERE id = ?",
                (session_id,),
            )

    # --- messages -------------------------------------------------------

    def add_message(self, session_id: int, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
        self.touch_session(session_id)

    def get_messages(self, session_id: int, limit: int = 40) -> list[Message]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [Message(role=r, content=c) for r, c in rows]

    # --- contact notes ----------------------------------------------------

    def add_contact_note(self, contact_id: str, note: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO contact_notes (contact_id, note) VALUES (?, ?)",
                (contact_id, note),
            )

    def get_contact_notes(self, contact_id: str, limit: int = 50) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT note FROM contact_notes WHERE contact_id = ? ORDER BY id DESC LIMIT ?",
                (contact_id, limit),
            ).fetchall()
        return [r[0] for r in rows]
