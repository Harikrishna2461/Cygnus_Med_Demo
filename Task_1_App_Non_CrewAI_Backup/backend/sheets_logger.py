"""
Google Sheets logger — mirrors DB writes to a shared Google Sheet.

Three tabs (auto-created with headers if missing):
  - Sessions  : session_id, title, mode, created_at, updated_at
  - Messages  : message_id, session_id, role, content, created_at
  - Feedback  : feedback_id, session_id, doctor_question, ai_response,
                doctor_feedback, doctor_rating, created_at

All calls are fire-and-forget (background thread) so Sheets latency
never blocks the Flask response. Failures are logged as warnings only.

Setup:
  1. Create a Google Cloud project, enable Sheets API + Drive API.
  2. Create a Service Account, download the JSON key.
  3. Share your Google Sheet with the service account email (Editor).
  4. Set GOOGLE_SHEETS_CREDENTIALS_FILE and GOOGLE_SHEETS_ID in config.py
     (or as environment variables).
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Tab definitions ────────────────────────────────────────────────────────────
_TABS = {
    "Sessions": ["session_id", "title", "mode", "created_at", "updated_at"],
    "Messages": ["message_id", "session_id", "role", "content", "created_at"],
    "Feedback": [
        "feedback_id", "session_id", "doctor_question", "ai_response",
        "doctor_feedback", "doctor_rating", "created_at",
    ],
}

_client = None
_sheet  = None
_lock   = threading.Lock()
_ready  = False


def _init():
    """Initialise gspread client once. Called lazily on first log call."""
    global _client, _sheet, _ready
    try:
        import gspread
        from config import GOOGLE_SHEETS_CREDENTIALS_FILE, GOOGLE_SHEETS_ID

        creds_path = Path(GOOGLE_SHEETS_CREDENTIALS_FILE)
        if not creds_path.exists():
            logger.warning(f"Sheets: credentials file not found at {creds_path} — logging disabled")
            return

        if not GOOGLE_SHEETS_ID:
            logger.warning("Sheets: GOOGLE_SHEETS_ID not set — logging disabled")
            return

        _client = gspread.service_account(filename=str(creds_path))
        _sheet  = _client.open_by_key(GOOGLE_SHEETS_ID)

        # Ensure every tab exists with the correct header row
        existing = {ws.title for ws in _sheet.worksheets()}
        for tab, headers in _TABS.items():
            if tab not in existing:
                ws = _sheet.add_worksheet(title=tab, rows=1000, cols=len(headers))
                ws.append_row(headers, value_input_option="RAW")
            else:
                ws = _sheet.worksheet(tab)
                first = ws.row_values(1)
                if first != headers:
                    ws.insert_row(headers, index=1, value_input_option="RAW")

        _ready = True
        logger.info("Sheets: connected and tabs verified")

    except ImportError:
        logger.warning("Sheets: gspread not installed — run: pip install gspread")
    except AttributeError:
        pass  # config keys not set yet — silently skip
    except Exception as e:
        logger.warning(f"Sheets: init failed ({e}) — logging disabled")


def _append(tab: str, row: list):
    """Append one row to the given tab. Runs in a background thread."""
    def _do():
        global _ready
        with _lock:
            if _client is None:
                _init()
            if not _ready:
                return
            try:
                ws = _sheet.worksheet(tab)
                ws.append_row(row, value_input_option="RAW")
            except Exception as e:
                logger.warning(f"Sheets: failed to append to {tab}: {e}")
                _ready = False  # force reinit on next call

    threading.Thread(target=_do, daemon=True).start()


# ── Public API ─────────────────────────────────────────────────────────────────

def log_session(session_id: str, title: str, mode: str, created_at: str, updated_at: str):
    _append("Sessions", [session_id, title, mode, created_at, updated_at])


def log_message(message_id: str, session_id: str, role: str, content: str, created_at: str):
    # Only log user and assistant turns — skip system messages
    if role == "system":
        return
    _append("Messages", [message_id, session_id, role, content[:2000], created_at])


def log_feedback(
    feedback_id: str,
    session_id: str,
    doctor_question: str,
    ai_response: str,
    doctor_feedback: str,
    doctor_rating,
    created_at: str,
):
    _append("Feedback", [
        feedback_id, session_id,
        doctor_question[:1000], ai_response[:2000],
        doctor_feedback or "", doctor_rating or "", created_at,
    ])
