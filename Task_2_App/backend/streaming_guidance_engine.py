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

def _crop_roi(frame, annotated: bool):
    """
    Crop the raw machine capture to just the ultrasound scan region.

    Annotated video (350×350): already clean — thin black border only.
      Crop: 3px all sides → 344×344 scan area.

    Raw video (720×480): full LOGIQ C9 machine output.
      Scan rectangle sits at x≈157–490, y≈52–430.
      Left of scan: black panel + "LOGIQ C9" label.
      Right of scan: machine params (FR/AO%/Gn/depth markers).
    """
    if annotated:
        return frame[3:348, 3:348]
    else:
        return frame[52:430, 157:490]


def extract_frame_at(pos_y_ratio: float, annotated: bool = True) -> Optional[str]:
    """
    Seek to pos_y_ratio × total_frames in the chosen video, crop to scan ROI.
    annotated=True  → full N1/N2/N3 overlay (saphenous zones)
    annotated=False → raw video (Giacomini / non-saphenous posterior)
    Returns base64 JPEG string, or None on failure.
    """
    from config import STREAM_VIDEO_PATH, STREAM_VIDEO_PATH_RAW
    video_path = STREAM_VIDEO_PATH if annotated else STREAM_VIDEO_PATH_RAW
    try:
        cap = cv2.VideoCapture(video_path)
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
        frame = _crop_roi(frame, annotated)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
    session.log_scan_position(pos_y, region, surface, leg, pos_x=pos_x, is_front=is_front)

    # ── 2. VLM ───────────────────────────────────────────────────────────────
    vlm_dict: Optional[dict] = None
    # Re-run whenever position moved enough OR the anatomical region changed —
    # region change can happen at the same posY (e.g. crossing anterior→posterior).
    run_vlm = (
        force_vlm
        or abs(pos_y - session.last_vlm_pos_y) >= STREAM_VLM_THRESHOLD
        or region != session.last_vlm_region
    )

    if run_vlm:
        # Giacomini (posterior thigh): show & analyse the blank tissue frame at posY=0.0
        # (annotated video, no N1/N2/N3 labels present) so VLM output is consistent with display.
        vlm_frame_y = 0.0 if region == 'Giacomini' else pos_y
        frame_b64 = extract_frame_at(vlm_frame_y)
        vlm_dict, vlm_summary = vlm_agent.analyze(frame_b64, region, leg)
        if frame_b64:
            session.last_vlm_summary = vlm_summary
            session.last_vlm_dict    = vlm_dict
            session.last_vlm_pos_y   = pos_y
            session.last_vlm_region  = region

    vlm_dict    = session.last_vlm_dict
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

        # Extract the last 3 guidance strings from history so Task 4 can avoid
        # repeating them (history stores assistant=guidance, user=state_msg pairs).
        recent_guidance = [
            ex["content"] for ex in session.history
            if ex.get("role") == "assistant"
        ][-3:]

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
                recent_guidance = recent_guidance,
                accepted_shunts = list(session.accepted_shunts),
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
            session.push_exchange(state_msg, guidance or raw, window=STREAM_HISTORY_WINDOW)
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
            "state":  state_msg,
            "agents": raw if isinstance(raw, dict) else {"guidance": raw},
        } if state_msg else None,
    }
