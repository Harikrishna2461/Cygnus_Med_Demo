import logging
from flask import Blueprint, jsonify, request
from auth import admin_required
from chat_db import get_all_users, create_user, deactivate_user, update_user_password

logger = logging.getLogger(__name__)
bp = Blueprint("admin", __name__)


@bp.route("/api/admin/users", methods=["GET"])
@admin_required
def list_users():
    return jsonify(get_all_users())


@bp.route("/api/admin/users", methods=["POST"])
@admin_required
def add_user():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    is_admin = bool(data.get("is_admin", False))

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    try:
        user_id = create_user(username, password, is_admin=is_admin)
        return jsonify({"user_id": user_id, "username": username, "status": "created"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/admin/users/<user_id>", methods=["DELETE"])
@admin_required
def remove_user(user_id):
    deactivate_user(user_id)
    return jsonify({"status": "deactivated"})


@bp.route("/api/admin/users/<user_id>/password", methods=["PATCH"])
@admin_required
def reset_password(user_id):
    data = request.get_json(force=True, silent=True) or {}
    new_password = data.get("password") or ""
    if not new_password:
        return jsonify({"error": "password is required"}), 400
    update_user_password(user_id, new_password)
    return jsonify({"status": "updated"})