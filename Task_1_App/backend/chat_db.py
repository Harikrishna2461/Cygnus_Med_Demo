"""
SQLite-backed storage for chat sessions, messages, and clinical feedback.
All user interactions are persisted for later review.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from config import DB_PATH


def _migrate_sessions_table():
    """Migrate sessions table to add missing columns (mode, hidden)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("PRAGMA table_info(sessions)")
            columns = {row[1] for row in cursor.fetchall()}

            if "mode" not in columns:
                conn.executescript("""
                    BEGIN TRANSACTION;

                    CREATE TABLE sessions_new (
                        session_id   TEXT PRIMARY KEY,
                        title        TEXT NOT NULL DEFAULT 'New Consultation',
                        mode         TEXT NOT NULL DEFAULT 'clinical' CHECK(mode IN ('clinical','general')),
                        created_at   TEXT NOT NULL,
                        updated_at   TEXT NOT NULL
                    );

                    INSERT INTO sessions_new (session_id, title, mode, created_at, updated_at)
                    SELECT session_id, title, 'clinical', created_at, updated_at FROM sessions;

                    DROP TABLE sessions;

                    ALTER TABLE sessions_new RENAME TO sessions;

                    COMMIT;
                """)
                cursor = conn.execute("PRAGMA table_info(sessions)")
                columns = {row[1] for row in cursor.fetchall()}

            if "hidden" not in columns:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0"
                )
                conn.commit()
    except Exception as e:
        print(f"Migration warning: {e}")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT 'New Consultation',
                mode         TEXT NOT NULL DEFAULT 'clinical' CHECK(mode IN ('clinical','general')),
                hidden       INTEGER NOT NULL DEFAULT 0,
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

    # Migrate existing database if mode column doesn't exist
    _migrate_sessions_table()


def _now() -> str:
    return datetime.now().isoformat()


# ── Sessions ─────────────────────────────────────────────────────────────────

def create_session(title: str = "New Consultation", mode: str = "clinical") -> str:
    sid = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, title, mode, hidden, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
            (sid, title, mode, now, now),
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


def get_sessions(mode: str | None = None) -> list[dict]:
    """Get non-hidden sessions, optionally filtered by mode (clinical or general)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if mode:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE mode=? AND hidden=0 ORDER BY updated_at DESC LIMIT 50",
                (mode,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE hidden=0 ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()
    return [dict(r) for r in rows]


def hide_session(session_id: str):
    """Mark a session as hidden (UI-only delete — data is preserved)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET hidden=1 WHERE session_id=?",
            (session_id,),
        )
        conn.commit()


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
