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
    """Migrate sessions table to add missing columns (mode, hidden, user_id).
    Usage: called once at startup from init_db(), before any session rows are read or
    written, so a database created before these columns existed keeps working without
    any manual fix-up."""
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


def _migrate_feedback_type():
    """Add feedback_type column to feedback table without losing existing data.
    Usage: called once at startup from init_db(), after _ensure_default_admin(), so a
    database created before this column existed keeps working without any manual
    fix-up."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("PRAGMA table_info(feedback)")
            columns = {row[1] for row in cursor.fetchall()}
            if "feedback_type" not in columns:
                conn.execute(
                    "ALTER TABLE feedback ADD COLUMN feedback_type TEXT NOT NULL DEFAULT 'classification'"
                )
                conn.commit()
    except Exception as e:
        print(f"Migration warning (feedback_type): {e}")


def init_db():
    """Creates the users/sessions/messages/feedback tables if they don't already exist,
    then runs the column migrations and default-admin seeding that must happen after the
    tables exist. Usage: called exactly once at process startup, from app.py's
    _startup()."""
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
                feedback_type     TEXT NOT NULL DEFAULT 'classification',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)
        conn.commit()

    _migrate_sessions_table()
    _ensure_default_admin()
    _migrate_feedback_type()


def _now() -> str:
    """Returns the current local timestamp as an ISO-8601 string. Usage: every
    created_at/updated_at value written anywhere in this module comes from this one
    function — it's the single source of timestamps for the whole database."""
    return datetime.now().isoformat()


# -- Users --------------------------------------------------------------------

def create_user(username: str, password: str = "", is_admin: bool = False) -> str:
    """Inserts a new user row with a fresh UUID user_id. Note: password_hash is always
    stored as an empty string here — this app's login (routes/auth.py's api_login) never
    checks a password at all, so the password argument is accepted for API compatibility
    but not actually used or verified against anything. Usage: called by
    _ensure_default_admin() to seed the default admin accounts at startup, and by
    routes/admin.py's add_user() when an admin creates a new account from the Admin
    Panel."""
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
    """Looks up one active user by username (case-insensitive — the username is
    lowercased before matching). Usage: called by routes/auth.py's api_login() — this
    lookup is the ONLY check performed during login; if a matching row comes back the
    user is logged in immediately, with no password verification of any kind."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1",
            (username.lower(),)
        ).fetchone()
    return dict(row) if row else None


def get_all_users() -> list[dict]:
    """Returns every user's public fields only (password_hash is deliberately excluded
    from the SELECT). Usage: called by routes/admin.py's list_users() to populate the
    Admin Panel's user list."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT user_id, username, is_admin, is_active, created_at FROM users ORDER BY created_at ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def deactivate_user(user_id: str):
    """Soft-deletes a user by setting is_active=0 rather than actually deleting the row
    — this keeps the row around so old sessions/messages that reference this user_id
    stay valid instead of becoming orphaned. Usage: called by routes/admin.py's
    remove_user() when an admin removes an account from the Admin Panel."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET is_active=0 WHERE user_id=?", (user_id,))
        conn.commit()


def update_user_password(user_id: str, new_password: str):
    """Hashes and stores a new password for a user. Usage: NOT currently called from
    anywhere in the codebase — no route exposes a change-password action, and login
    itself (get_user_by_username) never checks password_hash regardless. This exists as
    the one place a real password check/reset could be wired in later if login is
    changed to require one."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE user_id=?",
            (generate_password_hash(new_password), user_id),
        )
        conn.commit()


def _ensure_default_admin():
    """Create the default admin account plus the team's admin accounts if no users
    exist yet. Login (see routes/auth.py) only checks that the username exists — there
    is no password check anywhere in this app — so these accounts work by username alone.
    Runs once per fresh database; re-running against an existing DB with users already
    present is a no-op. Usage: called from init_db() at startup, between the two table
    migration steps."""
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        for username in (ADMIN_USERNAME, "krish", "harin", "jeffry"):
            try:
                create_user(username, ADMIN_PASSWORD, is_admin=True)
                print(f"Default admin '{username}' created.")
            except Exception as e:
                print(f"Could not create default admin '{username}': {e}")


# -- Sessions -----------------------------------------------------------------

def create_session(title: str = "New Consultation", mode: str = "clinical", user_id: str | None = None) -> str:
    """Creates a new chat session row (mode is either "clinical" or "general") and
    mirrors it to the optional Google Sheets log. Usage: called by routes/sessions.py's
    api_new_session() whenever a user starts a new conversation from either the
    Clinical Assistant or General Chat tab."""
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
    """Renames a session and bumps its updated_at (the title is truncated to 80 chars).
    Usage: called two ways — automatically by routes/clinical.py and routes/general.py
    to auto-title a session from the user's first message, and manually by
    routes/sessions.py's api_rename_session() when a user renames a session themselves
    from the sidebar."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE session_id=?",
            (title[:80], _now(), session_id),
        )
        conn.commit()


def get_sessions(mode: str | None = None, user_id: str | None = None) -> list[dict]:
    """Get non-hidden sessions, optionally filtered by mode and user_id. Usage: called
    by routes/sessions.py's api_list_sessions() to populate the session sidebar,
    scoped to the logged-in user's own sessions plus any session with no owner
    (user_id IS NULL)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        base = "SELECT * FROM sessions WHERE hidden=0"
        params: list = []
        if mode:
            base += " AND mode=?"
            params.append(mode)
        if user_id:
            base += " AND (user_id=? OR user_id IS NULL)"
            params.append(user_id)
        base += " ORDER BY updated_at DESC LIMIT 50"
        rows = conn.execute(base, params).fetchall()
    return [dict(r) for r in rows]


def hide_session(session_id: str):
    """Marks a session as hidden — a soft-archive, not a delete; the row and all its
    messages stay in the database. Usage: called by routes/sessions.py's
    api_hide_session() when a user archives a session from the sidebar."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE sessions SET hidden=1 WHERE session_id=?", (session_id,))
        conn.commit()


# -- Messages -----------------------------------------------------------------

def save_message(session_id: str, role: str, content: str, metadata: dict | None = None) -> str:
    """Inserts one chat message (role is "user" or "assistant"), bumps the parent
    session's updated_at, and mirrors the write to the optional Google Sheets log.
    Usage: this is the single write path for every message in both the Clinical
    (/api/chat) and General Chat (/api/general-chat) flows — called repeatedly
    throughout routes/clinical.py and routes/general.py for every user message and
    every assistant reply, including all of the follow-up/sufficiency-gate
    questions."""
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
    """Fetches a session's full message history in chronological order, parsing each
    message's metadata column back from JSON into a dict. Usage: called by
    routes/clinical.py and routes/general.py to reconstruct conversation history/context
    for the current turn, and by routes/sessions.py's api_get_messages() to let the
    frontend redisplay a past session when it's reopened."""
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
    feedback_type: str = "classification",
) -> str:
    """Records one feedback entry (a clinician's rating/comment on a classification or
    ligation result) tied to a session, and mirrors it to the optional Google Sheets
    log. Usage: called by routes/feedback.py's api_submit_feedback()."""
    fid = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO feedback (feedback_id, session_id, doctor_question, ai_response, doctor_feedback, doctor_rating, created_at, feedback_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fid, session_id, doctor_question, ai_response, doctor_feedback or None, doctor_rating, now, feedback_type),
        )
        conn.commit()
    _sheets.log_feedback(fid, session_id, doctor_question, ai_response, doctor_feedback or "", doctor_rating, now)
    return fid


def get_all_feedback() -> list[dict]:
    """Returns the most recent 500 feedback entries, newest first. Usage: called by
    routes/feedback.py's api_get_feedback() to populate the Feedback Log view — the
    session audit trail shown in the UI."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    return [dict(r) for r in rows]


def get_db_export() -> dict:
    """Serializes the entire database (users, sessions, messages, feedback) into one
    dict, parsing each message's metadata back from JSON along the way. Usage: called
    by routes/admin.py's export_db() to produce the Admin Panel's full-database export
    file."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        users = [dict(r) for r in conn.execute(
            "SELECT user_id, username, is_admin, is_active, created_at FROM users ORDER BY created_at ASC"
        ).fetchall()]
        sessions = [dict(r) for r in conn.execute(
            "SELECT * FROM sessions ORDER BY created_at ASC"
        ).fetchall()]
        messages = []
        for r in conn.execute("SELECT * FROM messages ORDER BY created_at ASC").fetchall():
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.get("metadata") or "{}")
            except Exception:
                d["metadata"] = {}
            messages.append(d)
        feedback = [dict(r) for r in conn.execute(
            "SELECT * FROM feedback ORDER BY created_at ASC"
        ).fetchall()]
    return {
        "export_timestamp": _now(),
        "users": users,
        "sessions": sessions,
        "messages": messages,
        "feedback": feedback,
    }