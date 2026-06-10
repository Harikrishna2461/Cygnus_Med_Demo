import logging
from auth import login_required
from flask import Blueprint, jsonify, request

from chat_db import save_message, get_messages, update_session_title
from config import RERANK_TOP_N
from general_chat_engine import (
    retrieve_general_context,
    collection_exists as general_collection_exists,
    is_domain_relevant,
)
from crew_pipeline import generate_general_response

logger = logging.getLogger(__name__)
bp = Blueprint("general", __name__)


def _clean_text(text: str) -> str:
    if not text:
        return text
    replacements = {
        '–': '-', '—': '-',
        '‘': "'", '’': "'",
        '“': '"', '”': '"',
        ' ': ' ',
        '​': '', '‌': '', '‍': '',
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text


_SYSTEM_PROMPT = """You are a clinical medical assistant specializing in venous disease and CHIVA methodology.
Plain text only. No markdown. No special characters. ASCII only.

EXAMPLE - simple question gets a short paragraph:

Question: What is CHIVA?
Answer: CHIVA stands for Cure Conservatrice et Hemodynamique de l'Insuffisance Veineuse en Ambulatoire. It is a minimally invasive technique for treating venous insufficiency that targets the hemodynamic source of reflux rather than removing veins. Developed to preserve the saphenous vein while eliminating pathological recirculation circuits.
Sources: Zamboni et al. 1998

---

EXAMPLE - complex multi-part question gets structured sections:

Question: Blood refluxes N2 to N3 to N1. What shunt type is this and what is the ligation strategy?
Answer:

CLASSIFICATION
Shunt Type II. Recirculation occurs entirely in the superficial veins. Reflux originates from the saphenous vein (N2), fills a tributary (N3), and re-enters either the saphenous vein or deep system (N1).

SUBTYPES
Type 2A: Tributary (N3) drains back into saphenous vein (N2). Saphenous vein competent above the refluxive junction.
Type 2B: Saphenous vein becomes incompetent just above the refluxive tributary junction.
Type 2C: Tributary (N3) re-enters the deep vein directly via perforator. Saphenous vein incompetent above junction.

LIGATION STRATEGY
Target the point where the tributary (N3) connects to the saphenous vein. Interrupting this eliminates the recirculation while preserving the saphenous vein. For Type 2C, target the perforating vein re-entry point.

SOURCES
Zamboni et al. 1998, Cappelli et al. 2000

---

Use the appropriate format based on question complexity. Do not copy chunks. Reason and answer."""


@bp.route("/api/general-chat", methods=["POST"])
@login_required
def api_general_chat():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.error(f"JSON parse error in /api/general-chat: {e}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON in request body"}), 400

    session_id: str = (data.get("session_id") or "").strip()
    user_message: str = (data.get("message") or "").strip()

    if not session_id or not user_message:
        return jsonify({"error": "session_id and message are required"}), 400

    user_msg_id = save_message(session_id, "user", user_message)

    if not is_domain_relevant(user_message):
        guardrail_msg = (
            "I'm a medical assistant specialised in venous disease, surgical procedures, "
            "and clinical guidelines. I'm not able to help with that topic. "
            "Please ask a medical or clinical question."
        )
        save_message(session_id, "assistant", guardrail_msg)
        return jsonify({
            "type": "guardrail",
            "conversational_response": guardrail_msg,
            "message_id": user_msg_id,
            "session_title": gen_title,
        })

    if not general_collection_exists():
        msg = "General knowledge collection not found. Ensure the final_structured_rag collection has been created."
        save_message(session_id, "assistant", msg)
        return jsonify({"type": "error", "conversational_response": msg, "message_id": user_msg_id})

    try:
        context_chunks = retrieve_general_context(user_message, k=RERANK_TOP_N)
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        context_chunks = []

    cleaned = [_clean_text(c) for c in context_chunks]
    context_str = "\n\n".join(cleaned) if cleaned else "No relevant context found."

    history = [m for m in get_messages(session_id) if m["message_id"] != user_msg_id]

    # Always name the session on the first message
    gen_title = None
    if not history:
        short_input = user_message[:45].rstrip() + ("…" if len(user_message) > 45 else "")
        gen_title = f"Medical Q&A — {short_input}"
        update_session_title(session_id, gen_title)

    system_with_history = _SYSTEM_PROMPT
    if history:
        system_with_history += "\n\nPrevious conversation context:"
        for msg in history[-4:]:
            role = "Doctor" if msg["role"] == "user" else "Assistant"
            system_with_history += f"\n{role}: {msg['content'][:300]}"

    user_prompt = (
        f"KNOWLEDGE BASE:\n{context_str}\n\n"
        f"QUESTION:\n{user_message}\n\n"
        "Answer based on the knowledge base. Simple question = short paragraph. "
        "Complex question = structured sections. End with SOURCES."
    )

    response_text = generate_general_response(system_with_history, user_prompt)

    save_message(session_id, "assistant", response_text)

    return jsonify({
        "type": "general",
        "conversational_response": response_text,
        "context_count": len(context_chunks),
        "message_id": user_msg_id,
        "session_title": gen_title,
    })
