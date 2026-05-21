"""
CHIVA Clinical Assistant — Flask Backend
-----------------------------------------
Chat-based clinical decision support for CHIVA venous shunt classification
and ligation planning. Accepts natural language descriptions from surgeons
and returns structured classification + ligation guidance.
"""

import logging
import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SELF_DIR = Path(__file__).resolve().parent
_PARENT_BACKEND = _SELF_DIR.parent.parent / "backend"

# Parent backend added first so it can be overridden by local files below.
# sys.path.insert(0, x) puts x at index 0, so the LAST insert wins.
sys.path.insert(0, str(_PARENT_BACKEND))   # index 1 after next line
sys.path.insert(0, str(_SELF_DIR))         # index 0 — local config.py wins

# ── Imports ───────────────────────────────────────────────────────────────────
import requests as _requests

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from groq import Groq as GroqClient

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    PORT, HOST, DEBUG,
    LOG_FILE, LOG_LEVEL,
    QDRANT_PATH, QDRANT_COLLECTION,
)
_QDRANT_STORAGE_PATH = QDRANT_PATH
_QDRANT_COLLECTION_NAME = QDRANT_COLLECTION
from chat_db import (
    init_db,
    create_session, update_session_title, get_sessions,
    save_message, get_messages,
    save_feedback, get_all_feedback,
)
from rag_engine import (
    retrieve_context, collection_exists, get_collection_size,
    load_bm25_from_qdrant,
)
from nl_interpreter import parse_nl_to_clips, build_conversational_response

# ── Shared prompts/rules from parent module (same as main app) ────────────────
try:
    from shunt_classification_and_ligation_llm import (
        CHIVA_RULES,
        LIGATION_QUERIES,
        classify_and_plan_ligation_with_llm,
    )
    _PARENT_MODULE = True
except ImportError as _err:
    logging.getLogger(__name__).warning(
        f"Parent module not found ({_err}). "
        "For shipping: copy shunt_classification_and_ligation_llm.py into cmed_demo/backend/."
    )
    _PARENT_MODULE = False
    CHIVA_RULES = ""
    LIGATION_QUERIES = {}

    def classify_and_plan_ligation_with_llm(*args, **kwargs):
        raise RuntimeError(
            "shunt_classification_and_ligation_llm.py not found. "
            "Copy it from backend/ into cmed_demo/backend/ and restart."
        )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
_FRONTEND_DIR = _SELF_DIR.parent / "frontend"
app = Flask(__name__, static_folder=None)
CORS(app, origins=["http://localhost:7860", "http://127.0.0.1:7860"])

# ── Groq client ───────────────────────────────────────────────────────────────
_groq = GroqClient(api_key=GROQ_API_KEY)

# Per-session analysis context cache (in-memory, for conversational follow-ups)
_analysis_cache: dict[str, str] = {}


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    #stream: bool = False,
    temperature: float = 0.3,
    max_tokens: int = 1536,
    return_usage: bool = False,
):
    """Call Groq LLM. Returns (text, usage_dict) when return_usage=True."""
    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        if return_usage:
            u = resp.usage
            usage = {
                "prompt_tokens":     getattr(u, "prompt_tokens",     0) if u else 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0,
                "total_tokens":      getattr(u, "total_tokens",      0) if u else 0,
            }
            return text, usage
        return text
    except Exception as e:
        logger.error(f"Groq LLM error: {e}")
        err_msg = f"LLM error: {e}"
        if return_usage:
            return err_msg, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return err_msg


def retrieve_ligation_context(query: str, k: int = 5) -> list[str]:
    return retrieve_context(query, k=k)


def _format_analysis_for_context(result: dict) -> str:
    lines: list[str] = []
    for f in result.get("findings", [result]):
        leg = f.get("leg", "Assessment")
        lines.append(
            f"LEG: {leg} | Shunt: {f.get('shunt_type','?')} "
            f"({f.get('confidence', 0):.0%} confidence)"
        )
        reasoning = f.get("reasoning", [])
        if reasoning:
            lines.append("Reasoning: " + " | ".join(reasoning[:3]))
        steps = f.get("ligation_steps", [])
        if steps:
            lines.append("Ligation steps: " + " -> ".join(steps[:3]))
        rationale = f.get("clinical_rationale", "")
        if rationale:
            lines.append(f"Rationale: {rationale[:300]}")
        lines.append("")
    return "\n".join(lines)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def serve_frontend():
    return send_file(str(_FRONTEND_DIR / "index.html"))


@app.route("/api/status")
def api_status():
    ollama_ok = False
    ollama_model_ok = False
    try:
        r = _requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            models = [m.get("name", "") for m in r.json().get("models", [])]
            ollama_model_ok = any("llama3.2" in m for m in models)
    except Exception:
        pass

    return jsonify({
        "status": "running",
        "parent_module_loaded": _PARENT_MODULE,
        "qdrant": {
            "collection_ready": collection_exists(),
            "document_count": get_collection_size(),
        },
        "ollama": {
            "running": ollama_ok,
            "model_ready": ollama_model_ok,
        },
    })


@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    return jsonify(get_sessions())


@app.route("/api/session", methods=["POST"])
def api_new_session():
    data = request.get_json(force=True) or {}
    title = data.get("title", "New Consultation")
    sid = create_session(title)
    return jsonify({"session_id": sid, "title": title})


@app.route("/api/session/<session_id>/messages", methods=["GET"])
def api_get_messages(session_id: str):
    return jsonify(get_messages(session_id))


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    session_id: str = data.get("session_id", "").strip()
    user_message: str = data.get("message", "").strip()

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

    # Step 1: Interpret the message
    interpretation = parse_nl_to_clips(user_message, call_llm)
    is_clinical = interpretation.get("is_clinical", False)
    clips = interpretation.get("clips", [])
    interp_text = interpretation.get("interpretation") or ""

    if is_clinical and clips:
        # Step 2: Full clinical pipeline
        try:
            result = classify_and_plan_ligation_with_llm(
                clip_list=clips,
                call_llm_fn=call_llm,
                retrieve_ligation_context_fn=retrieve_ligation_context,
            )
        except Exception as e:
            logger.error(f"Classification pipeline failed: {e}", exc_info=True)
            error_msg = f"Classification pipeline error: {e}"
            save_message(session_id, "assistant", error_msg)
            return jsonify({"type": "error", "conversational_response": error_msg}), 500

        _analysis_cache[session_id] = _format_analysis_for_context(result)

        # Auto-title the session
        if not any(m["role"] == "assistant" for m in history):
            primary_type = result.get("shunt_type") or "Unknown"
            short_input = user_message[:45].rstrip() + ("…" if len(user_message) > 45 else "")
            update_session_title(session_id, f"{primary_type} — {short_input}")

        findings = result.get("findings", [])
        response_payload = {
            "type": "clinical",
            "interpretation": interp_text,
            "findings": findings,
            "shunt_type": result.get("shunt_type"),
            "confidence": result.get("confidence"),
            "summary": result.get("summary"),
            "needs_elim_test": result.get("needs_elim_test", False),
            "ask_branching": result.get("ask_branching", False),
            "token_usage": result.get("token_usage", {}),
            "conversational_response": None,
            "message_id": user_msg_id,
        }

        save_message(
            session_id, "assistant",
            f"[Analysis] {interp_text or 'Clinical assessment complete.'}",
            metadata=response_payload,
        )

        return jsonify(response_payload)

    else:
        # Conversational follow-up
        analysis_ctx = _analysis_cache.get(session_id, "")
        response_text = build_conversational_response(
            user_message=user_message,
            analysis_context=analysis_ctx,
            history=history,
            call_llm_fn=call_llm,
        )
        save_message(session_id, "assistant", response_text)
        return jsonify({
            "type": "conversational",
            "conversational_response": response_text,
            "message_id": user_msg_id,
        })


@app.route("/api/feedback", methods=["GET"])
def api_feedback():
    return jsonify(get_all_feedback())


@app.route("/api/feedback", methods=["POST"])
def api_submit_feedback():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "").strip()
    doctor_question = data.get("doctor_question", "").strip()
    ai_response = data.get("ai_response", "").strip()
    doctor_feedback = data.get("doctor_feedback", "").strip()
    doctor_rating = data.get("doctor_rating")

    logger.info(f"Feedback received: session={session_id[:20] if session_id else 'EMPTY'}, "
                f"question={doctor_question[:50] if doctor_question else 'EMPTY'}, "
                f"response={ai_response[:50] if ai_response else 'EMPTY'}")

    if not session_id or not doctor_question or not ai_response:
        logger.error(f"Feedback validation failed: session={bool(session_id)}, "
                     f"question={bool(doctor_question)}, response={bool(ai_response)}")
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
    )
    return jsonify({"feedback_id": feedback_id, "status": "saved"})


# ── Startup ───────────────────────────────────────────────────────────────────

def _open_browser():
    time.sleep(2)
    webbrowser.open(f"http://{HOST}:{PORT}")

def _startup():
    init_db()
    logger.info("Database initialised.")

    if not collection_exists():
        logger.error(
            f"Qdrant collection '{_QDRANT_COLLECTION_NAME}' not found in "
            f"{_QDRANT_STORAGE_PATH}. "
            "Ensure the parent backend/qdrant_storage is intact."
        )
    else:
        # Build BM25 index from the pre-existing collection (no ingestion needed)
        load_bm25_from_qdrant()

    logger.info(f"CHIVA Clinical Assistant ready at http://{HOST}:{PORT}")


if __name__ == "__main__":
    _startup()
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)
