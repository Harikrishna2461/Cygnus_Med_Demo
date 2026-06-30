"""
Streaming guidance engine — orchestrator layer.

At each probe_move or clip_mark event, this module:
  1. Logs the probe position to session.scan_log  (feeds history_agent)
  2. Runs VLM frame analysis via agents.vlm_agent (if position moved enough)
  3. Calls agents.history_agent  → LLM-generated scan history narrative
  4. Calls agents.q_state_agent  → Q1-Q4 circuit status from confirmed clips
  5. Calls agents.protocol_agent → zone-specific examination protocol
  6. Builds an enriched state message via agents.guidance_agent
  7. Runs the CrewAI 5-agent crew to produce the final guidance instruction
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2

from agents import history_agent, q_state_agent, protocol_agent, vlm_agent, guidance_agent
from agents import crew_pipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame_at(pos_y_ratio: float) -> Optional[str]:
    """
    Seek to pos_y_ratio × total_frames in the annotated stream video.
    Returns base64 JPEG string, or None on failure.
    """
    from config import STREAM_VIDEO_PATH
    try:
        cap = cv2.VideoCapture(STREAM_VIDEO_PATH)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        idx = min(int(pos_y_ratio * total), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode()
    except Exception as exc:
        logger.error("Frame extraction error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED VLM + LLM STEP — called from route handler
# ─────────────────────────────────────────────────────────────────────────────

def process_probe_state(
    session: "StreamSession",
    region: str,
    pos_y: float,
    surface: str,
    leg: str,
    pos_x: Optional[float] = None,
    is_front: Optional[bool] = None,
    force_vlm: bool = False,
    force_llm: bool = False,
) -> dict:
    """
    Process a probe position update through the full agent pipeline.

    Steps:
      1. Log position to session.scan_log for history_agent coverage tracking.
      2. VLM: run if position changed >= STREAM_VLM_THRESHOLD or force_vlm.
      3. If LLM threshold reached (or force_llm):
           a. Build context from all agents (history, q_state, protocol).
           b. Compose enriched state message.
           c. Run CrewAI 5-agent crew.
      4. Return result dict for WebSocket emit.
    """
    from config import STREAM_VLM_THRESHOLD, STREAM_LLM_THRESHOLD, STREAM_HISTORY_WINDOW

    # ── 1. Log position ───────────────────────────────────────────────────────
    session.log_scan_position(pos_y, region, surface, leg, is_front=is_front)

    # ── 2. VLM ───────────────────────────────────────────────────────────────
    vlm_dict: Optional[dict] = None
    run_vlm = force_vlm or abs(pos_y - session.last_vlm_pos_y) >= STREAM_VLM_THRESHOLD

    if run_vlm:
        frame_b64 = extract_frame_at(pos_y)
        vlm_dict, vlm_summary = vlm_agent.analyze(frame_b64, region, leg)
        if frame_b64:
            session.last_vlm_summary = vlm_summary
            session.last_vlm_pos_y   = pos_y

    vlm_summary = session.last_vlm_summary

    # ── 3. CrewAI pipeline ────────────────────────────────────────────────────
    guidance        = None
    raw             = None
    action          = None
    state_msg       = None
    shunt_confirmed = False
    shunt_type      = None
    shunt_evidence  = None

    run_llm = (
        force_llm
        or abs(pos_y - session.last_llm_pos_y) >= STREAM_LLM_THRESHOLD
        or region != session.last_llm_region
    )

    # Echo last known action/guidance when LLM isn't called (position moved
    # less than STREAM_LLM_THRESHOLD).  Echo for all action states so the
    # client always gets a response and never times out waiting.
    if not run_llm and session.last_guidance and session.last_action:
        guidance = session.last_guidance
        action   = session.last_action

    if run_llm:
        # Only classify clips from the leg currently under the probe
        leg_clips = [c for c in session.clips if c.get("leg", "right") == leg]

        # ── 3a. Gather context from sub-agents ────────────────────────────────
        hist_summary  = history_agent.build_summary(session, pos_y)
        q_state       = q_state_agent.analyze(leg_clips)
        protocol_text = protocol_agent.get_protocol(region, pos_y)

        # ── 3b. Build enriched state message ──────────────────────────────────
        state_msg = guidance_agent.build_state_message(
            region          = region,
            pos_y           = pos_y,
            surface         = surface,
            leg             = leg,
            clips           = leg_clips,
            vlm_summary     = vlm_summary,
            history_summary = hist_summary,
            q_state         = q_state,
            protocol_text   = protocol_text,
            pos_x           = pos_x,
            is_front        = is_front,
        )

        # ── 3c. CrewAI 5-agent crew ───────────────────────────────────────────

        try:
            guidance, raw, action, shunt_found, s_type, s_evidence = crew_pipeline.run_guidance_crew(
                state_message   = state_msg,
                region          = region,
                pos_y           = pos_y,
                surface         = surface,
                leg             = leg,
                clips           = leg_clips,
                vlm_summary     = vlm_summary,
                history_summary = hist_summary,
                q_state         = q_state,
                protocol_text   = protocol_text,
                pos_x           = pos_x,
                is_front        = is_front,
                rejection_notes = list(session.rejection_notes),
            )
            # Fire shunt_confirmed only once per unique leg:type key per session.
            confirmed_key = f"{leg}:{s_type}"
            if shunt_found and confirmed_key not in session.confirmed_shunts:
                session.confirmed_shunts.append(confirmed_key)
                shunt_confirmed = True
                shunt_type      = s_type
                shunt_evidence  = s_evidence

            session.last_llm_pos_y  = pos_y
            session.last_llm_region = region
            session.last_guidance   = guidance
            session.last_action     = action
            session.push_exchange(state_msg, raw, window=STREAM_HISTORY_WINDOW)
            session.push_thinking(pos_y, region, state_msg, raw, guidance)
        except Exception as exc:
            logger.error("Crew pipeline error: %s", exc)
            guidance = "Continue scanning distally to locate anatomical junction."
            raw      = str(exc)
            action   = "move"
            # Store so sub-threshold echo probes can respond instead of silently dropping.
            session.last_guidance   = guidance
            session.last_action     = action
            session.last_llm_pos_y  = pos_y
            session.last_llm_region = region

    return {
        "guidance":         guidance,
        "action":           action,
        "vlm":              vlm_dict,
        "vlm_summary":      vlm_summary,
        "region":           region,
        "pos_y":            pos_y,
        "shunt_confirmed":  shunt_confirmed,
        "shunt_type":       shunt_type,
        "shunt_evidence":   shunt_evidence,
        "thinking": {
            "state": state_msg,
            "raw":   raw,
        } if state_msg else None,
    }
