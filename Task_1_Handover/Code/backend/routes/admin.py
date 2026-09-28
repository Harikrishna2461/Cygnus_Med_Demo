import io
import json
from datetime import datetime
import openpyxl
from flask import Blueprint, jsonify, request, Response
from auth import admin_required
from chat_db import get_all_users, create_user, deactivate_user, get_db_export

bp = Blueprint("admin", __name__)


@bp.route("/api/admin/users", methods=["GET"])
@admin_required
def list_users():
    """GET /api/admin/users — returns every user account (admin-only). Usage: called
    by the frontend's Admin Panel to populate the user management list; wraps
    chat_db.get_all_users()."""
    return jsonify(get_all_users())


@bp.route("/api/admin/users", methods=["POST"])
@admin_required
def add_user():
    """POST /api/admin/users — creates a new user account (admin-only). Usage: called
    by the Admin Panel's "add user" form; wraps chat_db.create_user(). Note: since
    login (routes/auth.py) never checks a password, the account works by username
    alone regardless of is_admin."""
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
    """DELETE /api/admin/users/<user_id> — deactivates a user account (admin-only, soft
    delete). Usage: called by the Admin Panel's "remove user" action; wraps
    chat_db.deactivate_user()."""
    deactivate_user(user_id)
    return jsonify({"status": "deactivated"})


@bp.route("/api/admin/export-db", methods=["GET"])
@admin_required
def export_db():
    """GET /api/admin/export-db — exports the entire database (users, sessions,
    messages, feedback) as a downloadable .xlsx workbook, one sheet per table
    (admin-only). Usage: called by the Admin Panel's "export database" button; wraps
    chat_db.get_db_export() and formats the result with openpyxl."""
    data = get_db_export()

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    tables = [
        ("Users", data["users"]),
        ("Sessions", data["sessions"]),
        ("Messages", data["messages"]),
        ("Feedback", data["feedback"]),
    ]

    for sheet_name, rows in tables:
        ws = wb.create_sheet(sheet_name)
        if not rows:
            continue
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append([
                json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for v in row.values()
            ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"cmed_db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return Response(
        buf.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )