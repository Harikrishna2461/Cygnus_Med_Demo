import logging
from auth import login_required
from flask import Blueprint, jsonify, request

from chat_db import save_feedback, get_all_feedback

logger = logging.getLogger(__name__)
bp = Blueprint("feedback", __name__)


@bp.route("/api/feedback", methods=["GET"])
@login_required
def api_get_feedback():
    return jsonify(get_all_feedback())


@bp.route("/api/feedback", methods=["POST"])
@login_required
def api_submit_feedback():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.error(f"JSON parse error in /api/feedback: {e}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    if not data or not isinstance(data, dict):
        data = {}

    session_id = (data.get("session_id") or "").strip()
    doctor_question = (data.get("doctor_question") or "").strip()
    ai_response = (data.get("ai_response") or "").strip()
    doctor_feedback = (data.get("doctor_feedback") or "").strip()
    doctor_rating = data.get("doctor_rating")
    feedback_type = (data.get("feedback_type") or "classification").strip()
    if feedback_type not in ("classification", "ligation"):
        feedback_type = "classification"

    logger.info(
        f"Feedback: session={session_id[:20] if session_id else 'EMPTY'}, "
        f"question={doctor_question[:50] if doctor_question else 'EMPTY'}"
    )

    if not session_id or not doctor_question or not ai_response:
        logger.error(
            f"Feedback validation failed: session={bool(session_id)}, "
            f"question={bool(doctor_question)}, response={bool(ai_response)}"
        )
        return jsonify({"error": "session_id, doctor_question, and ai_response are required"}), 400

    if doctor_rating:
        try:
            doctor_rating = int(doctor_rating)
            if not (1 <= doctor_rating <= 5):
                return jsonify({"error": "doctor_rating must be between 1 and 5"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "doctor_rating must be an integer"}), 400

    feedback_id = save_feedback(
        session_id=session_id,
        doctor_question=doctor_question,
        ai_response=ai_response,
        doctor_feedback=doctor_feedback,
        doctor_rating=doctor_rating,
        feedback_type=feedback_type,
    )
    return jsonify({"feedback_id": feedback_id, "status": "saved"})
