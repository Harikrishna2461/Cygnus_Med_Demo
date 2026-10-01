from flask import Blueprint, jsonify

status_bp = Blueprint("status", __name__)


@status_bp.route("/api/status", methods=["GET"])
def status():
    from config import GROQ_API_KEY

    return jsonify({
        "ok": True,
        "groq_configured": bool(GROQ_API_KEY),
    })
