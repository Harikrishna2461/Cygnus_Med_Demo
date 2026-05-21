"""
SQLite-backed storage for chat sessions, messages, and clinical feedback.
All user interactions are persisted for later review.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from config import DB_PATH


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT 'New Consultation',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id   TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                role         TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                content      TEXT NOT NULL,
                metadata     TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id       TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL,
                doctor_question   TEXT NOT NULL,
                ai_response       TEXT NOT NULL,
                doctor_feedback   TEXT,
                doctor_rating     INTEGER CHECK(doctor_rating >= 1 AND doctor_rating <= 5),
                created_at        TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)
        conn.commit()


def _now() -> str:
    return datetime.now().isoformat()


# ── Sessions ─────────────────────────────────────────────────────────────────

def create_session(title: str = "New Consultation") -> str:
    sid = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (sid, title, now, now),
        )
        conn.commit()
    return sid


def update_session_title(session_id: str, title: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE session_id=?",
            (title[:80], _now(), session_id),
        )
        conn.commit()


def get_sessions() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Messages ─────────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, metadata: dict | None = None) -> str:
    mid = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            (mid, session_id, role, content, json.dumps(metadata or {}), now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE session_id=?",
            (now, session_id),
        )
        conn.commit()
    return mid


def get_messages(session_id: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except Exception:
            d["metadata"] = {}
        result.append(d)
    return result


# ── Feedback ─────────────────────────────────────────────────────────────────

def save_feedback(
    session_id: str,
    doctor_question: str,
    ai_response: str,
    doctor_feedback: str = "",
    doctor_rating: int | None = None,
) -> str:
    fid = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                fid,
                session_id,
                doctor_question,
                ai_response,
                doctor_feedback or None,
                doctor_rating,
                _now(),
            ),
        )
        conn.commit()
    return fid


def get_all_feedback() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    return [dict(r) for r in rows]
