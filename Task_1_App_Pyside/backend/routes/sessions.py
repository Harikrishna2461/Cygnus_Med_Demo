import logging
from flask import Blueprint, jsonify, request

from chat_db import (
    create_session, update_session_title,
    get_sessions, get_messages, hide_session,
)

logger = logging.getLogger(__name__)
bp = Blueprint("sessions", __name__)


@bp.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    mode = request.args.get("mode", None)
    return jsonify(get_sessions(mode=mode))


@bp.route("/api/session", methods=["POST"])
def api_new_session():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.error(f"JSON parse error in /api/session: {e}")
        data = None

    if not data or not isinstance(data, dict):
        data = {}

    title = data.get("title", "New Chat")
    mode = data.get("mode", "clinical")
    sid = create_session(title, mode=mode)
    return jsonify({"session_id": sid, "title": title, "mode": mode})


@bp.route("/api/session/<session_id>/messages", methods=["GET"])
def api_get_messages(session_id: str):
    return jsonify(get_messages(session_id))


@bp.route("/api/session/<session_id>/title", methods=["PATCH"])
def api_rename_session(session_id: str):
    data = request.get_json(force=True, silent=True) or {}
    new_title = (data.get("title") or "").strip()
    if not new_title:
        return jsonify({"error": "title is required"}), 400
    update_session_title(session_id, new_title)
    return jsonify({"session_id": session_id, "title": new_title})


@bp.route("/api/session/<session_id>/hide", methods=["PATCH"])
def api_hide_session(session_id: str):
    hide_session(session_id)
    return jsonify({"status": "hidden"})
