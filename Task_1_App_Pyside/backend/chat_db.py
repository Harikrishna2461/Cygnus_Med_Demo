"""
SQLite-backed storage for chat sessions, messages, clinical feedback, and users.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash
from config import DB_PATH, ADMIN_USERNAME, ADMIN_PASSWORD
import sheets_logger as _sheets


def _migrate_sessions_table():
    """Migrate sessions table to add missing columns (mode, hidden, user_id)."""
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

            if "user_id" not in columns:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN user_id TEXT"
                )
                conn.commit()
    except Exception as e:
        print(f"Migration warning: {e}")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       TEXT PRIMARY KEY,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin      INTEGER NOT NULL DEFAULT 0,
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT 'New Consultation',
                mode         TEXT NOT NULL DEFAULT 'clinical' CHECK(mode IN ('clinical','general')),
                hidden       INTEGER NOT NULL DEFAULT 0,
                user_id      TEXT,
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

    _migrate_sessions_table()
    _ensure_default_admin()


def _now() -> str:
    return datetime.now().isoformat()


# -- Users --------------------------------------------------------------------

def create_user(username: str, password: str = "", is_admin: bool = False) -> str:
    uid = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO users (user_id, username, password_hash, is_admin, is_active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
            (uid, username.lower(), "", int(is_admin), now),
        )
        conn.commit()
    return uid


def get_user_by_username(username: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1",
            (username.lower(),)
        ).fetchone()
    return dict(row) if row else None


def get_all_users() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT user_id, username, is_admin, is_active, created_at FROM users ORDER BY created_at ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def deactivate_user(user_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET is_active=0 WHERE user_id=?", (user_id,))
        conn.commit()


def update_user_password(user_id: str, new_password: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE user_id=?",
            (generate_password_hash(new_password), user_id),
        )
        conn.commit()


def _ensure_default_admin():
    """Create the default admin account if no users exist."""
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        try:
            create_user(ADMIN_USERNAME, ADMIN_PASSWORD, is_admin=True)
            print(f"Default admin '{ADMIN_USERNAME}' created.")
        except Exception as e:
            print(f"Could not create default admin: {e}")


# -- Sessions -----------------------------------------------------------------

def create_session(title: str = "New Consultation", mode: str = "clinical", user_id: str | None = None) -> str:
    sid = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, title, mode, hidden, user_id, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?, ?)",
            (sid, title, mode, user_id, now, now),
        )
        conn.commit()
    _sheets.log_session(sid, title, mode, now, now)
    return sid


def update_session_title(session_id: str, title: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE session_id=?",
            (title[:80], _now(), session_id),
        )
        conn.commit()


def get_sessions(mode: str | None = None, user_id: str | None = None) -> list[dict]:
    """Get non-hidden sessions, optionally filtered by mode and user_id."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        base = "SELECT * FROM sessions WHERE hidden=0"
        params: list = []
        if mode:
            base += " AND mode=?"
            params.append(mode)
        if user_id:
            base += " AND user_id=?"
            params.append(user_id)
        base += " ORDER BY updated_at DESC LIMIT 50"
        rows = conn.execute(base, params).fetchall()
    return [dict(r) for r in rows]


def hide_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE sessions SET hidden=1 WHERE session_id=?", (session_id,))
        conn.commit()


# -- Messages -----------------------------------------------------------------

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
    _sheets.log_message(mid, session_id, role, content, now)
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


# -- Feedback -----------------------------------------------------------------

def save_feedback(
    session_id: str,
    doctor_question: str,
    ai_response: str,
    doctor_feedback: str = "",
    doctor_rating: int | None = None,
) -> str:
    fid = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fid, session_id, doctor_question, ai_response, doctor_feedback or None, doctor_rating, now),
        )
        conn.commit()
    _sheets.log_feedback(fid, session_id, doctor_question, ai_response, doctor_feedback or "", doctor_rating, now)
    return fid


def get_all_feedback() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    return [dict(r) for r in rows]