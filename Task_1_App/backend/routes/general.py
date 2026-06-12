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

RESPONSE FORMAT:
- Simple factual question: one short paragraph, no headings.
- Complex or multi-part question: use short ALL-CAPS headings that describe the actual content
  of each section. Choose headings based on what the question asks — do not reuse fixed headings
  from memory. If the question is about anatomy, use anatomy-appropriate headings. If it is about
  a procedure, use procedure-appropriate headings. Never force CLASSIFICATION or LIGATION STRATEGY
  headings onto questions that are not about classification or ligation.
- Always end with: SOURCES: [author last name et al. year, ...] — unique entries only, short format.

Do not copy chunks verbatim. Reason from the knowledge base and answer in your own words."""


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

    # Deduplicate source lines that appear across chunks so the LLM doesn't repeat them
    import re as _re
    _seen_sources: set[str] = set()
    _deduped: list[str] = []
    for chunk in cleaned:
        lines = chunk.splitlines()
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Heuristic: source lines are short, contain a year, and no sentence punctuation
            if _re.search(r'\b(19|20)\d{2}\b', stripped) and len(stripped) < 200 and stripped.count('.') <= 2:
                key = _re.sub(r'\s+', ' ', stripped.lower())
                if key in _seen_sources:
                    continue
                _seen_sources.add(key)
            kept.append(line)
        _deduped.append('\n'.join(kept))
    context_str = "\n\n".join(_deduped) if _deduped else "No relevant context found."

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
        "Answer based on the knowledge base. "
        "End with SOURCES listing only unique references in short format: Author et al. Year. "
        "Do not repeat the same source twice."
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
