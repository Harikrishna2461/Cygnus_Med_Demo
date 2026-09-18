import logging
from flask import Blueprint, jsonify, request, session
from chat_db import get_user_by_username

logger = logging.getLogger(__name__)
bp = Blueprint("auth", __name__)


@bp.route("/api/login", methods=["POST"])
def api_login():
    """POST /api/login — authenticates by username only (there is no password check
    anywhere in this app) and sets the session cookie on success. Usage: called by
    login.html's login form; looks the user up via chat_db.get_user_by_username()."""
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip().lower()

    if not username:
        return jsonify({"error": "Username is required"}), 400

    user = get_user_by_username(username)
    if not user:
        logger.warning(f"Failed login attempt for username: {username}")
        return jsonify({"error": "Username not recognised"}), 401

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
    """POST /api/logout — clears the session cookie, logging the current user out.
    Usage: called by the frontend's logout button on any page."""
    session.clear()
    return jsonify({"status": "logged out"})


@bp.route("/api/me", methods=["GET"])
def api_me():
    """GET /api/me — returns the currently logged-in user's id/username/admin status
    from the session cookie. Usage: called by the frontend on page load to check who's
    logged in and whether to show admin-only UI."""
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({
        "user_id": session["user_id"],
        "username": session["username"],
        "is_admin": session.get("is_admin", False),
    })