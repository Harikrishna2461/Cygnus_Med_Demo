import logging
from flask import Blueprint, jsonify, request, session as flask_session
from auth import login_required
from chat_db import create_session, update_session_title, get_sessions, get_messages, hide_session

logger = logging.getLogger(__name__)
bp = Blueprint("sessions", __name__)


@bp.route("/api/sessions", methods=["GET"])
@login_required
def api_list_sessions():
    mode = request.args.get("mode", None)
    return jsonify(get_sessions(mode=mode, user_id=flask_session["user_id"]))


@bp.route("/api/session", methods=["POST"])
@login_required
def api_new_session():
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
    return jsonify(get_messages(session_id))


@bp.route("/api/session/<session_id>/title", methods=["PATCH"])
@login_required
def api_rename_session(session_id: str):
    data = request.get_json(force=True, silent=True) or {}
    new_title = (data.get("title") or "").strip()
    if not new_title:
        return jsonify({"error": "title is required"}), 400
    update_session_title(session_id, new_title)
    return jsonify({"session_id": session_id, "title": new_title})


@bp.route("/api/session/<session_id>/hide", methods=["PATCH"])
@login_required
def api_hide_session(session_id: str):
    hide_session(session_id)
    return jsonify({"status": "hidden"})