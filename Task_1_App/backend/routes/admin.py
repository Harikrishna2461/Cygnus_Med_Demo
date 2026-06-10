import json
import logging
from datetime import datetime
from flask import Blueprint, jsonify, request, Response
from auth import admin_required
from chat_db import get_all_users, create_user, deactivate_user, get_db_export

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
    is_admin = bool(data.get("is_admin", False))

    if not username:
        return jsonify({"error": "username is required"}), 400

    try:
        user_id = create_user(username, is_admin=is_admin)
        return jsonify({"user_id": user_id, "username": username, "status": "created"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/admin/users/<user_id>", methods=["DELETE"])
@admin_required
def remove_user(user_id):
    deactivate_user(user_id)
    return jsonify({"status": "deactivated"})


@bp.route("/api/admin/export-db", methods=["GET"])
@admin_required
def export_db():
    data = get_db_export()
    filename = f"cmed_db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        json.dumps(data, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )