import logging
from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash
from chat_db import get_user_by_username

logger = logging.getLogger(__name__)
bp = Blueprint("auth", __name__)


@bp.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = get_user_by_username(username)
    if not user or not check_password_hash(user["password_hash"], password):
        logger.warning(f"Failed login attempt for username: {username}")
        return jsonify({"error": "Invalid username or password"}), 401

    session.clear()
    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    session["is_admin"] = bool(user["is_admin"])
    session.permanent = True

    return jsonify({
        "user_id": user["user_id"],
        "username": user["username"],
        "is_admin": bool(user["is_admin"]),
    })


@bp.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"status": "logged out"})


@bp.route("/api/me", methods=["GET"])
def api_me():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({
        "user_id": session["user_id"],
        "username": session["username"],
        "is_admin": session.get("is_admin", False),
    })