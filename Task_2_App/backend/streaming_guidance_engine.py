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
# GUIDANCE FRAME PICKER — for VLM analysis, use the same annotated reference
# frames the user sees (guidance/ concat frames > vein_frames/ > streaming video)
# ─────────────────────────────────────────────────────────────────────────────

_REGION_TO_GUIDANCE_DIRS = {
    "SFJ":         ["sfj"],
    "Upper Thigh": ["gsv_thigh"],
    "Hunterian":   ["gsv_thigh"],
    "Dodd":        ["gsv_thigh"],
    "GSV-THI":     ["gsv_thigh"],
    "GSV-CAL":     ["gsv_calf"],
    "SPJ":         ["spj"],
    "SSV":         ["ssv"],
    # Giacomini intentionally omitted — no dedicated guidance folder;
    # Giocomini vein_frames used instead (metadata bypass, no VLM call needed).
    "Calf":        ["gsv_calf"],
}

_REGION_TO_VEIN_DIRS = {
    "SFJ":         ["GSV_Prox", "CFV"],
    "Upper Thigh": ["GSV_Prox", "FV", "AASV"],
    "Hunterian":   ["GSV_Prox", "FV", "Hunt_Perf"],
    "Dodd":        ["GSV_Distal", "FV", "Dodd_Perf"],
    "GSV-THI":     ["GSV_Distal", "FV"],
    "GSV-CAL":     ["GSV_Distal"],
    "SPJ":         ["SSV", "PV"],
    "SSV":         ["SSV", "PASV"],
    "Giacomini":   ["Giocomini", "SSV", "Tributary", "AASV"],
    "Calf":        ["GSV_Distal", "Tributary", "Boyd_Perf", "Cockett_Perf"],
    "UNKNOWN":     ["GSV_Prox"],
}

# Folder name → anatomy metadata (no VLM needed for vein_frames — the folder IS the label)
_VEIN_FOLDER_TO_META: dict[str, dict] = {
    "GSV_Prox":      {"vein_types": ["GSV"],              "n2": True,  "n3": False, "n1": False},
    "GSV_Distal":    {"vein_types": ["GSV"],              "n2": True,  "n3": False, "n1": False},
    "GSV":           {"vein_types": ["GSV"],              "n2": True,  "n3": False, "n1": False},
    "SSV":           {"vein_types": ["SSV"],              "n2": True,  "n3": False, "n1": False},
    "Tributary":     {"vein_types": ["Tributary"],        "n2": False, "n3": True,  "n1": False},
    "Giocomini":     {"vein_types": ["SSV", "Tributary"], "n2": True,  "n3": True,  "n1": False},
    "AASV":          {"vein_types": ["AASV"],             "n2": False, "n3": True,  "n1": False},
    "PASV":          {"vein_types": ["PASV"],             "n2": False, "n3": True,  "n1": False},
    "CFV":           {"vein_types": ["CFV"],              "n2": False, "n3": False, "n1": True},
    "FV":            {"vein_types": ["FV"],               "n2": False, "n3": False, "n1": True},
    "FV_CFV":        {"vein_types": ["FV", "CFV"],        "n2": False, "n3": False, "n1": True},
    "PV":            {"vein_types": ["PV"],               "n2": False, "n3": False, "n1": True},
    "DFV":           {"vein_types": ["DFV"],              "n2": False, "n3": False, "n1": True},
    "Deep_Vein_Calf":{"vein_types": ["FV"],               "n2": False, "n3": False, "n1": True},
    "Boyd_Perf":     {"vein_types": ["Boyd_Perf"],        "n2": False, "n3": True,  "n1": False},
    "Dodd_Perf":     {"vein_types": ["Dodd_Perf"],        "n2": True,  "n3": False, "n1": True},
    "Hunt_Perf":     {"vein_types": ["Hunt_Perf"],        "n2": True,  "n3": False, "n1": True},
    "Cockett_Perf":  {"vein_types": ["Cockett_Perf"],     "n2": False, "n3": True,  "n1": True},
}


def _build_region_sources(region: str, assets: str) -> list:
    """
    Build an ordered list of all available frame sources for a region:
      [(source_type, dir_path, sorted_jpgs, folder_meta), ...]
    guidance folders come first, then every vein_frames folder.
    Caller cycles through this list using pos_y so every vein type gets shown.
    """
    import os
    sources = []

    for subdir in _REGION_TO_GUIDANCE_DIRS.get(region, []):
        d = os.path.join(assets, "guidance", subdir)
        if os.path.isdir(d):
            jpgs = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
            if jpgs:
                sources.append(("guidance", d, jpgs, None))

    for vname in _REGION_TO_VEIN_DIRS.get(region, []):
        d = os.path.join(assets, "vein_frames", vname)
        if os.path.isdir(d):
            jpgs = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
            if jpgs:
                sources.append(("vein_frames", d, jpgs, _VEIN_FOLDER_TO_META.get(vname)))

    return sources


def _get_frame_for_region(
    region: str, pos_y: float
) -> tuple[Optional[str], str, Optional[dict]]:
    """
    Returns (base64_jpeg, source_type, folder_meta).
    Cycles through ALL sources for the region (guidance + every vein folder) using
    pos_y so every vein type gets shown as the probe moves — not just the first folder.
    """
    import os
    assets = os.path.join(os.path.dirname(__file__), "..", "assets")

    sources = _build_region_sources(region, assets)
    if sources:
        bucket = int(round(pos_y * 20))   # one bucket per 0.05 step (= VLM threshold)
        n_src  = len(sources)
        source_type, d, jpgs, meta = sources[bucket % n_src]
        frame_idx = (bucket // n_src) % len(jpgs)
        try:
            with open(os.path.join(d, jpgs[frame_idx]), "rb") as fh:
                return base64.b64encode(fh.read()).decode(), source_type, meta
        except Exception:
            pass

    # Fallback: streaming video
    return extract_frame_at(pos_y), "stream", None


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

    vlm_frame_updated = False

    if run_vlm:
        # Re-pick frame on every VLM run (pos_y changed enough OR region changed).
        # Frame selection is position-based (not random) so moving the probe
        # shows frames from different parts of the anatomical video.
        frame_b64, frame_source, folder_meta = _get_frame_for_region(region, pos_y)

        if frame_source == "vein_frames" and folder_meta:
            from vlm_analyzer import UltrasoundAssessment
            vt = list(folder_meta["vein_types"])
            assessment = UltrasoundAssessment(
                image_quality="good",
                fascial_layer_visible=False,
                n2_in_fascial_compartment=folder_meta["n2"],
                n3_superficial_to_fascia=folder_meta["n3"],
                n1_deep_to_fascia=folder_meta["n1"],
                label_n2_visible=folder_meta["n2"],
                label_n3_visible=folder_meta["n3"],
                label_n1_visible=folder_meta["n1"],
                vein_types=vt,
                frame_note=f"Vein frame: {', '.join(vt)}",
            )
            vlm_dict    = assessment.to_dict()
            vlm_summary = assessment.summary()
        else:
            vlm_dict, vlm_summary = vlm_agent.analyze(frame_b64, region, leg)

        if frame_b64:
            session.last_vlm_summary    = vlm_summary
            session.last_vlm_dict       = vlm_dict
            session.last_vlm_pos_y      = pos_y
            session.last_vlm_region     = region
            session.last_vlm_frame_b64  = frame_b64
            vlm_frame_updated = True

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
        "frame_b64":        session.last_vlm_frame_b64 if vlm_frame_updated else None,
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
