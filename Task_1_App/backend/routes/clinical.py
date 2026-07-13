import logging
from auth import login_required
from flask import Blueprint, jsonify, request

from chat_db import save_message, get_messages, update_session_title
from config import QDRANT_COLLECTION
from rag_engine import collection_exists
from crew_pipeline import (
    parse_nl_to_clips,
    build_conversational_response,
    classify_and_plan_ligation_with_llm,
)
import services

logger = logging.getLogger(__name__)
bp = Blueprint("clinical", __name__)


def set_classification_fn(loaded: bool, fn) -> None:
    """No-op — CrewAI pipeline is self-contained and needs no external injection."""
    pass


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

    history = [m for m in get_messages(session_id) if m["message_id"] != user_msg_id]

    # Always name the session on the first message, regardless of response type
    new_title = None
    if not history:
        short_input = user_message[:45].rstrip() + ("…" if len(user_message) > 45 else "")
        new_title = short_input
        update_session_title(session_id, new_title)

    interpretation = parse_nl_to_clips(user_message, history=history)
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
            "session_title": new_title,
        })

    if is_clinical and sufficient:
        try:
            result = classify_and_plan_ligation_with_llm(
                clip_list=clips,
                retrieve_ligation_context_fn=services.retrieve_ligation_context,
            )
        except Exception as e:
            logger.error(f"Classification pipeline failed: {e}", exc_info=True)
            error_msg = (
                "I was unable to classify this case. "
                "Please try rephrasing your description and submitting again."
            )
            save_message(session_id, "assistant", error_msg)
            return jsonify({"type": "error", "conversational_response": error_msg, "session_title": new_title}), 500

        # If the elimination test is still needed, route back to dialogue — do not show classification card
        if result.get("needs_elim_test"):
            elim_msg = (
                f"Interpreted so far: {interp_text}\n\n"
                "To distinguish between Type 1+2 and Type 3, I need the elimination test result. "
                "Compress the GSV or SFJ manually and observe what happens to the tributary reflux: "
                "does it disappear (abolished) or remain (persists)? "
                "For example: 'compression of the GSV abolished tributary reflux' or "
                "'reflux in the tributary persisted when the SFJ was compressed'."
            )
            save_message(session_id, "assistant", elim_msg)
            return jsonify({
                "type": "insufficient",
                "missing_info": elim_msg,
                "conversational_response": elim_msg,
                "message_id": user_msg_id,
                "session_title": new_title,
            })

        # For Type 1+2: RP N2→N1 calibre determines CHIVA 1 vs CHIVA 2 — ask before showing card
        if result.get("shunt_type") == "Type 1+2":
            rp_n2n1_clips = [
                c for c in clips
                if c.get("flow") == "RP" and c.get("fromType") == "N2" and c.get("toType") == "N1"
            ]
            has_rp_calibre = any(c.get("calibre") for c in rp_n2n1_clips)
            if rp_n2n1_clips and not has_rp_calibre:
                calibre_msg = (
                    f"Interpreted so far: {interp_text}\n\n"
                    "To determine the correct ligation strategy for Type 1+2, I need to know the calibre "
                    "of the RP N2→N1 re-entry perforator — the point where the GSV trunk returns blood to "
                    "the deep system.\n\n"
                    "Small calibre → CHIVA 2 staged (ligate the tributary junction first, reassess SFJ at "
                    "6–12 months).\n"
                    "Large calibre → CHIVA 1 simultaneous (ligate SFJ and all tributary junctions in the "
                    "same session).\n\n"
                    "Please describe the calibre of the RP N2→N1 perforator — for example: "
                    "'The re-entry perforator at the calf is small calibre' or "
                    "'The RP N2→N1 perforator has large calibre.'"
                )
                save_message(session_id, "assistant", calibre_msg)
                return jsonify({
                    "type": "insufficient",
                    "missing_info": calibre_msg,
                    "conversational_response": calibre_msg,
                    "message_id": user_msg_id,
                    "session_title": new_title,
                })

        # For Type 4: pelvic vs perforating subtype determines surgical approach — ask if unknown
        if result.get("shunt_type") == "Type 4":
            ep_n1n3_clips = [
                c for c in clips
                if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N3"
            ]
            has_source = any(c.get("source") for c in ep_n1n3_clips)
            if ep_n1n3_clips and not has_source:
                source_msg = (
                    f"Interpreted so far: {interp_text}\n\n"
                    "To plan the correct Type 4 ligation approach, I need to know the origin of the "
                    "entry point (EP N1→N3) — these two subtypes require different operations:\n\n"
                    "Perforating subtype: an incompetent perforator at a specific body level (thigh or "
                    "calf) delivers deep blood directly into a tributary. Requires sub-fascial or "
                    "mini-open perforator ligation at that level.\n\n"
                    "Pelvic subtype: a pelvic, pudendal, gluteal, labial, or ovarian vein enters a "
                    "groin tributary, bypassing the SFJ. Requires a groin incision targeting the "
                    "pelvic/pudendal vein; coil embolisation may be needed if reflux persists.\n\n"
                    "Please clarify the source — for example: "
                    "'An incompetent Hunterian-level perforator feeds a thigh tributary directly' or "
                    "'A pudendal vein from the pelvis enters a groin tributary.'"
                )
                save_message(session_id, "assistant", source_msg)
                return jsonify({
                    "type": "insufficient",
                    "missing_info": source_msg,
                    "conversational_response": source_msg,
                    "message_id": user_msg_id,
                    "session_title": new_title,
                })

        # If multiple tributary branches need detail for ligation planning, ask before showing card
        if result.get("ask_branching"):
            has_branching_details = any(
                c.get("calibre") or c.get("notes")
                for c in clips
                if c.get("fromType") == "N3" or c.get("toType") == "N3"
            )
            if not has_branching_details:
                branch_msg = (
                    f"Interpreted so far: {interp_text}\n\n"
                    "Multiple tributary branches are present. To determine the optimal ligation sequence "
                    "I need a few more details about the branches:\n"
                    "1. Which branch has the larger calibre — or are they equal in diameter?\n"
                    "2. Which branch is closer to its nearest perforator?\n"
                    "3. Is there sufficient independent drainage available through either branch?\n\n"
                    "For example: 'The anterior branch is the dominant one — larger calibre and further from "
                    "the perforator. The posterior branch is smaller but has adequate independent drainage.'"
                )
                save_message(session_id, "assistant", branch_msg)
                return jsonify({
                    "type": "insufficient",
                    "missing_info": branch_msg,
                    "conversational_response": branch_msg,
                    "message_id": user_msg_id,
                    "session_title": new_title,
                })

        services.analysis_cache[session_id] = services.format_analysis_for_context(result)

        # Refine title with shunt type on first classification
        prior_classification = any(
            m["role"] == "assistant"
            and isinstance(m.get("metadata"), dict)
            and m["metadata"].get("type") == "clinical"
            for m in history
        )
        if not prior_classification:
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
        )
        save_message(session_id, "assistant", response_text)
        return jsonify({
            "type": "conversational",
            "conversational_response": response_text,
            "message_id": user_msg_id,
            "session_title": new_title,
        })
