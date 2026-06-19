"""
Streaming guidance engine — orchestrator layer.

At each probe_move or clip_mark event, this module:
  1. Logs the probe position to session.scan_log  (feeds history_agent)
  2. Runs VLM frame analysis via agents.vlm_agent (if position moved enough)
  3. Calls agents.history_agent  → posY-band coverage summary
  4. Calls agents.q_state_agent  → Q1-Q4 circuit status from confirmed clips
  5. Calls agents.protocol_agent → zone-specific examination protocol
  6. Builds an enriched state message via agents.guidance_agent
  7. Evaluates deterministic circuit-state rules (_rule_based_action)
  8. Falls back to LLM (agents.guidance_agent.call_llm) when no rule fires

The old monolithic _STREAM_SYSTEM_PROMPT, build_state_message(), and
call_with_history() have been moved to agents/guidance_agent.py.
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
# RULE-BASED ACTION DETERMINATION (deterministic, evaluated before LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_action(
    clips: list[dict],
    max_visited_pos_y: float = 0.0,
) -> tuple[str | None, str | None]:
    """
    Evaluate deterministic circuit-state rules against the confirmed clip set.

    Returns (action, guidance) if a rule fires, (None, None) to fall through
    to the LLM.  Rules are evaluated in priority order — maneuver before complete.

    Using old_max_visited (the value BEFORE the current probe_move) ensures the
    Type 1/2A gate (Rule 6) requires the probe to have previously visited the
    popliteal zone, not just reached it on the current move.
    """
    def has(flow, ft, tt):
        return any(
            c.get("flow") == flow and c.get("from_type") == ft and c.get("to_type") == tt
            for c in clips
        )

    ep_n1_n2 = has("EP", "N1", "N2")
    ep_n1_n3 = has("EP", "N1", "N3")
    ep_n2_n3 = has("EP", "N2", "N3")
    rp_n2_n1 = has("RP", "N2", "N1")
    rp_n3_n1 = has("RP", "N3", "N1")
    rp_n3_n2 = has("RP", "N3", "N2")

    # First EP N2→N3 clip that HAS an elimTest value.
    ep_n2_n3_elim = next(
        (c.get("elimination_test", "") for c in clips
         if c.get("flow") == "EP" and c.get("from_type") == "N2" and c.get("to_type") == "N3"
         and c.get("elimination_test", "")),
        "",
    )
    ep_n2_n3_no_elim = ep_n2_n3 and not ep_n2_n3_elim

    COMPLETE_MSG = "Circuit mapped — sufficient findings for classification"

    # 1. Maneuver: all three trigger clips present, no elimTest on any EP N2→N3 yet.
    if ep_n2_n3_no_elim and rp_n3_n1 and rp_n2_n1:
        return "maneuver", "Compress tributary — record whether GSV Doppler changes"

    # 2. Type 3 complete: elimination test result = No Reflux.
    if ep_n2_n3_elim == "No Reflux":
        return "complete", COMPLETE_MSG

    # 3. Type 1+2 complete: elimination test result = Reflux.
    if ep_n2_n3_elim == "Reflux":
        return "complete", COMPLETE_MSG

    # 4. Type 4 complete: direct perforator entry + trunk re-entry.
    if ep_n1_n3 and rp_n2_n1:
        return "complete", COMPLETE_MSG

    # 5. Type 6 complete: perforator-to-perforator, no trunk involvement.
    if ep_n1_n3 and rp_n3_n1 and not rp_n2_n1 and not rp_n3_n2:
        return "complete", COMPLETE_MSG

    # 5b. Type 5 complete: EP N1→N3 → RP N3→N2 → EP N2→N3 → RP N3→N1.
    if ep_n1_n3 and rp_n3_n2 and ep_n2_n3 and rp_n3_n1:
        return "complete", COMPLETE_MSG

    # 6. Type 1/2A complete: junction entry + trunk reflux, no escape found.
    # Gate: probe must have previously scanned to popliteal zone (max_visited ≥ 0.48)
    # so the full trunk has been assessed for escape points before declaring complete.
    if ep_n1_n2 and rp_n2_n1 and not ep_n2_n3 and max_visited_pos_y >= 0.48:
        return "complete", COMPLETE_MSG

    # 7. Type 2B/II complete: saphenous trunk is source (no deep junction entry);
    # blood escapes trunk into tributary (EP N2→N3) and re-enters deep (RP N3→N1).
    if ep_n2_n3 and rp_n3_n1 and not ep_n1_n2 and not ep_n1_n3 and not rp_n2_n1:
        return "complete", COMPLETE_MSG

    return None, None


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
      2. VLM: run if position changed ≥ STREAM_VLM_THRESHOLD or force_vlm.
      3. If LLM threshold reached (or force_llm):
           a. Build context from all agents (history, q_state, protocol).
           b. Compose enriched state message.
           c. Evaluate deterministic rules.
           d. Call LLM if no rule fired.
      4. Return result dict for WebSocket emit.
    """
    from config import GROQ_API_KEY, GROQ_TEXT_MODEL
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
        # If no frame extracted, vlm_agent returns "No frame provided." which
        # we still store so the LLM has the latest usable VLM context.

    vlm_summary = session.last_vlm_summary

    # ── 3. LLM gate ──────────────────────────────────────────────────────────
    guidance  = None
    raw       = None
    action    = None
    state_msg = None

    # Update max_visited unconditionally so sub-threshold moves still advance the gate.
    old_max_visited = session.max_visited_pos_y
    session.max_visited_pos_y = max(session.max_visited_pos_y, pos_y)

    run_llm = force_llm or abs(pos_y - session.last_llm_pos_y) >= STREAM_LLM_THRESHOLD
    if run_llm:
        # ── 3a. Gather context from sub-agents ────────────────────────────────
        hist_summary   = history_agent.build_summary(session, pos_y)
        q_state        = q_state_agent.analyze(session.clips)
        protocol_text  = protocol_agent.get_protocol(region, pos_y)

        # ── 3b. Build enriched state message ──────────────────────────────────
        state_msg = guidance_agent.build_state_message(
            region       = region,
            pos_y        = pos_y,
            surface      = surface,
            leg          = leg,
            clips        = session.clips,
            vlm_summary  = vlm_summary,
            history_summary = hist_summary,
            q_state      = q_state,
            protocol_text = protocol_text,
            pos_x        = pos_x,
            is_front     = is_front,
        )

        # ── 3c. Deterministic rule check ──────────────────────────────────────
        # old_max_visited (before this move) is used for the Type 1/2A gate.
        rule_action, rule_guidance = _rule_based_action(session.clips, old_max_visited)

        if rule_action:
            guidance = rule_guidance
            raw      = f"[rule-based] action={rule_action}"
            action   = rule_action
            session.last_llm_pos_y = pos_y
            if action in ("complete", "maneuver"):
                session.last_llm_pos_y = -1.0  # re-trigger LLM on next move
            session.push_thinking(pos_y, region, state_msg, raw, guidance)

        else:
            # ── 3d. CrewAI 3-agent pipeline ───────────────────────────────────
            try:
                guidance, raw, action = crew_pipeline.run_guidance_crew(
                    state_message   = state_msg,
                    region          = region,
                    pos_y           = pos_y,
                    surface         = surface,
                    leg             = leg,
                    clips           = session.clips,
                    vlm_summary     = vlm_summary,
                    history_summary = hist_summary,
                    q_state         = q_state,
                    protocol_text   = protocol_text,
                    pos_x           = pos_x,
                    is_front        = is_front,
                )
                # Safety guard: only rules may authorise "complete" or "maneuver".
                if action in ("complete", "maneuver"):
                    action   = "move"
                    guidance = guidance_agent.fallback_guidance(session.clips)
                session.last_llm_pos_y = pos_y
                session.push_exchange(state_msg, raw, window=STREAM_HISTORY_WINDOW)
                session.push_thinking(pos_y, region, state_msg, raw, guidance)
            except Exception as exc:
                logger.error("Crew pipeline error: %s", exc)
                guidance = "Unable to generate guidance."
                raw      = str(exc)
                action   = "move"

    return {
        "guidance":    guidance,
        "action":      action,
        "vlm":         vlm_dict,
        "vlm_summary": vlm_summary,
        "region":      region,
        "pos_y":       pos_y,
        "thinking": {
            "state": state_msg,
            "raw":   raw,
        } if state_msg else None,
    }
