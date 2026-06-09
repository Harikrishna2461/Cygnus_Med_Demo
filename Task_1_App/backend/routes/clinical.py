import logging
from auth import login_required
from flask import Blueprint, jsonify, request

from chat_db import save_message, get_messages, update_session_title
from config import QDRANT_COLLECTION
from rag_engine import collection_exists
from nl_interpreter import parse_nl_to_clips, build_conversational_response
import services

logger = logging.getLogger(__name__)
bp = Blueprint("clinical", __name__)

_PARENT_MODULE = False
classify_and_plan_ligation_with_llm = None


def set_classification_fn(loaded: bool, fn) -> None:
    global _PARENT_MODULE, classify_and_plan_ligation_with_llm
    _PARENT_MODULE = loaded
    classify_and_plan_ligation_with_llm = fn


@bp.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.error(f"JSON parse error in /api/chat: {e}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON in request body"}), 400

    session_id: str = (data.get("session_id") or "").strip()
    user_message: str = (data.get("message") or "").strip()

    if not session_id or not user_message:
        return jsonify({"error": "session_id and message are required"}), 400

    user_msg_id = save_message(session_id, "user", user_message)

    if not collection_exists():
        msg = (
            f"Qdrant collection '{QDRANT_COLLECTION}' not found. "
            "Ensure backend/qdrant_storage is present and the main app has been run at least once."
        )
        save_message(session_id, "assistant", msg)
        return jsonify({"type": "error", "conversational_response": msg, "message_id": user_msg_id})

    if not _PARENT_MODULE:
        msg = (
            "Core classification module not found. "
            "Copy shunt_classification_and_ligation_llm.py from backend/ into "
            "cmed_demo/backend/ and restart."
        )
        save_message(session_id, "assistant", msg)
        return jsonify({"type": "error", "conversational_response": msg, "message_id": user_msg_id})

    history = [m for m in get_messages(session_id) if m["message_id"] != user_msg_id]

    interpretation = parse_nl_to_clips(user_message, services.call_llm, history)
    is_clinical = interpretation.get("is_clinical", False)
    sufficient = interpretation.get("sufficient_information", True)
    missing_info = interpretation.get("missing_information") or ""
    clips = interpretation.get("clips", [])
    interp_text = interpretation.get("interpretation") or ""

    if is_clinical and not sufficient:
        decline_msg = (
            missing_info
            if missing_info
            else (
                "To classify this case I need a complete flow path description from your duplex scan. "
                "Please describe: (1) where blood enters the superficial system (e.g. SFJ incompetent, perforator entry, direct deep-to-tributary), "
                "(2) how it travels (e.g. forward along GSV, escapes into tributary), and "
                "(3) whether and where reflux occurs (e.g. GSV refluxes backward, tributary drains back, no reflux anywhere)."
            )
        )
        save_message(session_id, "assistant", decline_msg)
        return jsonify({
            "type": "insufficient",
            "missing_info": decline_msg,
            "conversational_response": decline_msg,
            "message_id": user_msg_id,
        })

    if is_clinical and sufficient and not clips:
        # CHIVA interpretation found no pathological flow events — no shunt present.
        no_shunt_payload = {
            "type": "clinical",
            "interpretation": interp_text or "No pathological shunt identified.",
            "findings": [],
            "shunt_type": "No shunt detected",
            "confidence": 0.97,
            "summary": "No pathological venous shunting identified. The venous system appears normal. CHIVA treatment is not indicated.",
            "needs_elim_test": False,
            "ask_branching": False,
            "token_usage": {},
            "conversational_response": None,
            "message_id": user_msg_id,
            "session_title": None,
        }
        save_message(session_id, "assistant", "[Analysis] No shunt detected.", metadata=no_shunt_payload)
        return jsonify(no_shunt_payload)

    if is_clinical and clips:
        try:
            result = classify_and_plan_ligation_with_llm(
                clip_list=clips,
                call_llm_fn=services.call_llm,
                retrieve_ligation_context_fn=services.retrieve_ligation_context,
            )
        except Exception as e:
            logger.error(f"Classification pipeline failed: {e}", exc_info=True)
            error_msg = f"Classification pipeline error: {e}"
            save_message(session_id, "assistant", error_msg)
            return jsonify({"type": "error", "conversational_response": error_msg}), 500

        services.analysis_cache[session_id] = services.format_analysis_for_context(result)

        new_title = None
        if not any(m["role"] == "assistant" for m in history):
            primary_type = result.get("shunt_type") or "Unknown"
            short_input = user_message[:45].rstrip() + ("…" if len(user_message) > 45 else "")
            new_title = f"{primary_type} — {short_input}"
            update_session_title(session_id, new_title)

        response_payload = {
            "type": "clinical",
            "interpretation": interp_text,
            "findings": result.get("findings", []),
            "shunt_type": result.get("shunt_type"),
            "confidence": result.get("confidence"),
            "summary": result.get("summary"),
            "needs_elim_test": result.get("needs_elim_test", False),
            "ask_branching": result.get("ask_branching", False),
            "token_usage": result.get("token_usage", {}),
            "conversational_response": None,
            "message_id": user_msg_id,
            "session_title": new_title,
        }

        save_message(
            session_id, "assistant",
            f"[Analysis] {interp_text or 'Clinical assessment complete.'}",
            metadata=response_payload,
        )
        return jsonify(response_payload)

    else:
        analysis_ctx = services.analysis_cache.get(session_id, "")
        response_text = build_conversational_response(
            user_message=user_message,
            analysis_context=analysis_ctx,
            history=history,
            call_llm_fn=services.call_llm,
        )
        save_message(session_id, "assistant", response_text)
        return jsonify({
            "type": "conversational",
            "conversational_response": response_text,
            "message_id": user_msg_id,
        })
