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
    RERANK_TOP_N,
)
_QDRANT_STORAGE_PATH = QDRANT_PATH
_QDRANT_COLLECTION_NAME = QDRANT_COLLECTION
from chat_db import (
    init_db,
    create_session, update_session_title, get_sessions,
    save_message, get_messages,
    save_feedback, get_all_feedback,
    hide_session,
)
from rag_engine import (
    retrieve_context, collection_exists, get_collection_size,
    load_bm25_from_qdrant, set_qdrant_client as set_rag_qdrant_client,
)
from general_chat_engine import (
    retrieve_general_context, collection_exists as general_collection_exists,
    get_collection_size as get_general_collection_size,
    load_bm25_from_collection as load_general_bm25,
    set_qdrant_client as set_general_qdrant_client,
    is_domain_relevant,
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

# ── Shared Qdrant client ───────────────────────────────────────────────────────
from qdrant_client import QdrantClient
_qdrant_shared = QdrantClient(path=_QDRANT_STORAGE_PATH)

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
def serve_landing():
    """Landing page with mode selection (Clinical vs General)."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Assistant</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f172a;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                width: 100%;
            }
            .header {
                text-align: center;
                margin-bottom: 60px;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 12px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: -0.5px;
            }
            .header p {
                font-size: 0.95em;
                color: #94a3b8;
            }
            .modes {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                margin-bottom: 40px;
            }
            @media (max-width: 768px) {
                .modes {
                    grid-template-columns: 1fr;
                }
                .header h1 {
                    font-size: 2em;
                }
            }
            .mode-card {
                background: #1e293b;
                border-radius: 12px;
                padding: 32px;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 1px solid #334155;
                text-decoration: none;
                color: inherit;
                display: flex;
                flex-direction: column;
            }
            .mode-card:hover {
                transform: translateY(-4px);
                border-color: #475569;
                background: #0f172a;
            }
            .mode-icon {
                font-size: 2.5em;
                margin-bottom: 16px;
            }
            .mode-card h2 {
                font-size: 1.5em;
                margin-bottom: 12px;
                color: #ffffff;
                font-weight: 600;
            }
            .mode-card p {
                font-size: 0.9em;
                color: #cbd5e1;
                line-height: 1.6;
                margin-bottom: 20px;
                flex: 1;
            }
            .features {
                list-style: none;
                margin-bottom: 20px;
            }
            .features li {
                padding: 6px 0;
                color: #94a3b8;
                font-size: 0.85em;
            }
            .features li:before {
                content: "• ";
                color: #64748b;
                margin-right: 8px;
            }
            .btn {
                padding: 11px 20px;
                border-radius: 8px;
                font-size: 0.95em;
                font-weight: 600;
                text-decoration: none;
                transition: all 0.2s ease;
                border: none;
                cursor: pointer;
                align-self: flex-start;
            }
            .btn-clinical {
                background: #2563eb;
                color: white;
            }
            .btn-clinical:hover {
                background: #1d4ed8;
            }
            .btn-general {
                background: #2563eb;
                color: white;
            }
            .btn-general:hover {
                background: #1d4ed8;
            }
            .footer {
                text-align: center;
                color: #64748b;
                font-size: 0.85em;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 1px solid #334155;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Medical Assistant</h1>
                <p>Select a mode to begin</p>
            </div>

            <div class="modes">
                <a href="/clinical" class="mode-card">
                    <div class="mode-icon">C</div>
                    <h2>Clinical Support</h2>
                    <p>Specialized analysis for venous shunt classification and ligation planning with clinical decision support.</p>
                    <ul class="features">
                        <li>Shunt classification</li>
                        <li>Ligation guidance</li>
                        <li>Clinical reasoning</li>
                    </ul>
                    <button class="btn btn-clinical">Enter →</button>
                </a>

                <a href="/general" class="mode-card">
                    <div class="mode-icon">+</div>
                    <h2>General Chat</h2>
                    <p>Ask any medical or surgical questions from the comprehensive knowledge base with intelligent search.</p>
                    <ul class="features">
                        <li>Medical research</li>
                        <li>General knowledge</li>
                        <li>Evidence-based answers</li>
                    </ul>
                    <button class="btn btn-general">Enter →</button>
                </a>
            </div>

            <div class="footer">
                <p>Always consult clinical guidelines and specialists for critical decisions</p>
            </div>
        </div>
    </body>
    </html>
    """


@app.route("/clinical")
def serve_clinical_frontend():
    """Serve the clinical decision support interface."""
    return send_file(str(_FRONTEND_DIR / "index.html"))


@app.route("/general")
def serve_general_frontend():
    """Serve the general medical chat interface."""
    return send_file(str(_FRONTEND_DIR / "general.html"))


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
    mode = request.args.get("mode", None)  # clinical, general, or None for all
    return jsonify(get_sessions(mode=mode))


@app.route("/api/session", methods=["POST"])
def api_new_session():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.error(f"JSON parse error in /api/session: {e}")
        data = None

    if not data or not isinstance(data, dict):
        data = {}

    title = data.get("title", "New Consultation")
    mode = data.get("mode", "clinical")  # clinical or general
    sid = create_session(title, mode=mode)
    return jsonify({"session_id": sid, "title": title, "mode": mode})


@app.route("/api/session/<session_id>/messages", methods=["GET"])
def api_get_messages(session_id: str):
    return jsonify(get_messages(session_id))


@app.route("/api/session/<session_id>/hide", methods=["PATCH"])
def api_hide_session(session_id: str):
    hide_session(session_id)
    return jsonify({"status": "hidden"})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.error(f"JSON parse error in /api/chat: {e}")
        logger.error(f"Request content-type: {request.content_type}")
        logger.error(f"Request data: {request.data}")
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


@app.route("/api/general-chat", methods=["POST"])
def api_general_chat():
    """General medical chat with RAG and domain guardrails."""
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        logger.error(f"JSON parse error in /api/general-chat: {e}")
        logger.error(f"Request content-type: {request.content_type}")
        logger.error(f"Request data: {request.data}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON in request body"}), 400

    session_id: str = (data.get("session_id") or "").strip()
    user_message: str = (data.get("message") or "").strip()
    mode: str = (data.get("mode") or "general").strip()

    if not session_id or not user_message:
        return jsonify({"error": "session_id and message are required"}), 400

    user_msg_id = save_message(session_id, "user", user_message)

    # Guardrail: reject off-topic queries before hitting the LLM
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
        })

    # Check collection exists
    if not general_collection_exists():
        msg = (
            f"General knowledge collection not found. "
            "Ensure the final_structured_rag collection has been created."
        )
        save_message(session_id, "assistant", msg)
        return jsonify({"type": "error", "conversational_response": msg, "message_id": user_msg_id})

    # Retrieve relevant context from final_structured_rag
    try:
        context_chunks = retrieve_general_context(user_message, k=RERANK_TOP_N)
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        context_chunks = []

    # Clean context chunks - normalize Unicode and special characters
    def clean_text(text):
        """Remove problematic Unicode characters and normalize."""
        if not text:
            return text
        replacements = {
            '–': '-', '—': '-',        # dashes
            ''': "'", ''': "'",        # quotes
            '"': '"', '"': '"',        # smart quotes
            ' ': ' ',                   # non-breaking space
            '​': '',              # zero-width space
            '‌': '',              # zero-width non-joiner
            '‍': '',              # zero-width joiner
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

    cleaned_chunks = [clean_text(chunk) for chunk in context_chunks]
    context_str = "\n\n".join(cleaned_chunks) if cleaned_chunks else "No relevant context found."

    system_prompt = """You are a clinical medical assistant specializing in venous disease and CHIVA methodology.
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

    user_prompt = f"""KNOWLEDGE BASE:
{context_str}

QUESTION:
{user_message}

Answer based on the knowledge base. Simple question = short paragraph. Complex question = structured sections. End with SOURCES."""

    history = [m for m in get_messages(session_id) if m["message_id"] != user_msg_id]

    # Convert history to conversation format for context
    system_with_history = system_prompt
    if history and len(history) > 0:
        system_with_history += "\n\nPrevious conversation context:"
        for msg in history[-4:]:  # Keep last 4 messages for context
            role = "Doctor" if msg["role"] == "user" else "Assistant"
            system_with_history += f"\n{role}: {msg['content'][:300]}"

    # Call LLM
    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_with_history},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=1536,
        )
        response_text = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        response_text = f"Error generating response: {e}"

    # Auto-title session
    if not any(m["role"] == "assistant" for m in history):
        short_input = user_message[:45].rstrip() + ("…" if len(user_message) > 45 else "")
        update_session_title(session_id, f"Medical Q&A — {short_input}")

    save_message(session_id, "assistant", response_text)

    return jsonify({
        "type": "general",
        "conversational_response": response_text,
        "context_count": len(context_chunks),
        "message_id": user_msg_id,
    })


@app.route("/api/feedback", methods=["GET"])
def api_feedback():
    return jsonify(get_all_feedback())


@app.route("/api/feedback", methods=["POST"])
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

    # Set shared Qdrant client for all engines (single client avoids lock conflicts)
    set_rag_qdrant_client(_qdrant_shared)
    set_general_qdrant_client(_qdrant_shared)

    if not collection_exists():
        logger.error(
            f"Qdrant collection '{_QDRANT_COLLECTION_NAME}' not found in "
            f"{_QDRANT_STORAGE_PATH}. "
            "Ensure the parent backend/qdrant_storage is intact."
        )
    else:
        # Build BM25 index from the pre-existing collection (no ingestion needed)
        load_bm25_from_qdrant()

    # Load general chat collection and BM25 index
    if not general_collection_exists():
        logger.warning(
            f"General collection 'final_structured_rag' not found. "
            "General chat mode will not be available."
        )
    else:
        logger.info(f"General collection ready ({get_general_collection_size()} points)")
        load_general_bm25()

    logger.info(f"Medical Assistant ready at http://{HOST}:{PORT}")


if __name__ == "__main__":
    _startup()
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)
