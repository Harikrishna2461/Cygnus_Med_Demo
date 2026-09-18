import logging
from flask import Blueprint, jsonify, request, session as flask_session
from auth import login_required
from chat_db import create_session, update_session_title, get_sessions, get_messages, hide_session

logger = logging.getLogger(__name__)
bp = Blueprint("sessions", __name__)


@bp.route("/api/sessions", methods=["GET"])
@login_required
def api_list_sessions():
    """GET /api/sessions — lists the logged-in user's sessions, optionally filtered by
    ?mode=clinical|general. Usage: called by the frontend to populate the session
    sidebar; wraps chat_db.get_sessions()."""
    mode = request.args.get("mode", None)
    return jsonify(get_sessions(mode=mode, user_id=flask_session["user_id"]))


@bp.route("/api/session", methods=["POST"])
@login_required
def api_new_session():
    """POST /api/session — creates a new session (clinical or general) for the logged-
    in user. Usage: called by the frontend's "New Session" action; wraps
    chat_db.create_session()."""
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        data = {}
    if not data or not isinstance(data, dict):
        data = {}
    title = data.get("title", "New Chat")
    mode = data.get("mode", "clinical")
    sid = create_session(title, mode=mode, user_id=flask_session["user_id"])
    return jsonify({"session_id": sid, "title": title, "mode": mode})


@bp.route("/api/session/<session_id>/messages", methods=["GET"])
@login_required
def api_get_messages(session_id: str):
    """GET /api/session/<session_id>/messages — fetches a session's full message
    history. Usage: called by the frontend when reopening a past session; wraps
    chat_db.get_messages()."""
    return jsonify(get_messages(session_id))


@bp.route("/api/session/<session_id>/title", methods=["PATCH"])
@login_required
def api_rename_session(session_id: str):
    """PATCH /api/session/<session_id>/title — renames a session. Usage: called when a
    user manually renames a session from the sidebar; wraps
    chat_db.update_session_title()."""
    data = request.get_json(force=True, silent=True) or {}
    new_title = (data.get("title") or "").strip()
    if not new_title:
        return jsonify({"error": "title is required"}), 400
    update_session_title(session_id, new_title)
    return jsonify({"session_id": session_id, "title": new_title})


@bp.route("/api/session/<session_id>/hide", methods=["PATCH"])
@login_required
def api_hide_session(session_id: str):
    """PATCH /api/session/<session_id>/hide — archives a session (soft, not deleted).
    Usage: called when a user hides/archives a session from the sidebar; wraps
    chat_db.hide_session()."""
    hide_session(session_id)
    return jsonify({"status": "hidden"})