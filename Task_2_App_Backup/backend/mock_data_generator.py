"""
Mock data generator for Task-2 system testing.

Produces 5 realistic examination scenarios, each containing:
  - A timestamped stream of scanner readings (segment_id, segment_dist, is_front, x, y)
  - EP/RP clips injected when the surgeon marks a point during the scan
    (format matches Task-1 clipdata.json exactly)
  - Real annotated ultrasound frames extracted from Task-3 videos, temporally
    aligned so each event's frame matches its position in the examination

Scenarios
---------
S1  Normal right-leg scan          — no pathology, no clips
S2  Type 1 shunt (right leg)       — SFJ incompetence + GSV reflux
S3  Type 2A shunt (left leg)       — GSV-to-tributary escape, SFJ competent
S4  Type 3 shunt (right leg)       — EP at SFJ + N2-to-N3 + RP bypass
S5  Wrong region then correction   — probe on SSV when GSV-THI expected

Frame alignment:
    Each region segment in a scenario has N events spanning t_start..t_end.
    Frames are sampled at evenly spaced positions across the concatenated video
    so that the visual content changes naturally as the probe moves.
"""

from __future__ import annotations

import copy
import json
import os
import random
from typing import Any

from video_processor import extract_frames_at_positions

random.seed(42)


# ── helpers ───────────────────────────────────────────────────────────────────

def _seg_to_posy(segment_id: int, segment_dist: float) -> float:
    is_thigh = segment_id in (0, 2)
    if is_thigh:
        return round(0.04 + segment_dist * 0.43, 3)
    return round(0.47 + segment_dist * 0.51, 3)


def _jitter(val: float, sigma: float = 0.008) -> float:
    return round(max(0.0, min(1.0, val + random.gauss(0, sigma))), 4)


def _scan_readings(
    segment_id: int,
    dist_start: float,
    dist_end: float,
    is_front: bool,
    n: int,
    t_start: float,
    t_end: float,
    x_base: int = 450,
    y_base: int = 300,
) -> list[dict]:
    readings = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        dist = dist_start + (dist_end - dist_start) * frac
        readings.append({
            "timestamp": round(t_start + (t_end - t_start) * frac, 2),
            "segment_id": segment_id,
            "segment_dist": _jitter(dist),
            "is_front": is_front,
            "x": x_base + random.randint(-4, 4),
            "y": y_base + random.randint(-4, 4),
        })
    return readings


def _clip(
    flow: str,
    from_type: str,
    to_type: str,
    segment_id: int,
    segment_dist: float,
    leg: str,
    seq_num: int,
    step: str,
    pos_x: float = 0.275,
) -> dict:
    label = f"{step}_{seq_num:02d}_{flow}_{from_type}-{to_type}"
    return {
        "algoSnapshot": None,
        "clipPath": f"{label}.mp4",
        "flow": flow,
        "fps": 30.0,
        "fromType": from_type,
        "fullPath": f"C:/mock/data_collection/{label}.mp4",
        "legSide": leg,
        "notes": "",
        "posXRatio": pos_x,
        "posYRatio": _seg_to_posy(segment_id, segment_dist),
        "processingProgress": 100,
        "sequenceNumber": seq_num,
        "step": step,
        "toType": to_type,
    }


# ── frame pool: temporally-aligned frames ──────────────────────────────────────

_frame_cache: dict[str, list[str]] = {}


def _fetch_frames(region: str, n: int, pos_start: float = 0.0, pos_end: float = 1.0) -> list[str]:
    """
    Return n frames from the region video between pos_start and pos_end (0..1).
    Caches the full frame list per region and slices on demand.
    Returns list of empty strings if no video available.
    """
    if region not in _frame_cache:
        # Pre-fetch 60 evenly-spaced frames to cover any subsequence request
        positions = [i / 59 for i in range(60)]
        frames = extract_frames_at_positions(region, positions)
        _frame_cache[region] = frames

    all_frames = _frame_cache[region]
    if not all_frames or all(f == "" for f in all_frames):
        return [""] * n

    total = len(all_frames)
    result = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        pos = pos_start + (pos_end - pos_start) * frac
        idx = int(pos * (total - 1))
        result.append(all_frames[min(idx, total - 1)])
    return result


def _build_events(
    readings: list[dict],
    region: str,
    expected_region: str,
    etype: str,
    descs: list[str],
    pos_start: float = 0.0,
    pos_end: float = 1.0,
    clips: list[dict | None] | None = None,
    accumulated_clips: list[list[dict] | None] | None = None,
) -> list[dict]:
    """
    Build events for a contiguous region scan segment.
    Frames are sampled at positions [pos_start..pos_end] across the video.
    readings, descs, clips, accumulated_clips must all have the same length.
    """
    n = len(readings)
    frames = _fetch_frames(region, n, pos_start, pos_end)
    if clips is None:
        clips = [None] * n
    if accumulated_clips is None:
        accumulated_clips = [None] * n

    events = []
    for i, r in enumerate(readings):
        ev: dict[str, Any] = {
            "timestamp": r["timestamp"],
            "type": etype,
            "description": descs[i] if i < len(descs) else etype,
            "scanner": {
                "segment_id": r["segment_id"],
                "segment_dist": r["segment_dist"],
                "is_front": r["is_front"],
                "x": r["x"],
                "y": r["y"],
            },
            "expected_region": expected_region,
            "image_meta": {"region": region},
        }
        if frames[i]:
            ev["image_b64"] = frames[i]
        if clips[i] is not None:
            ev["type"] = "ep_rp_mark"
            ev["new_clip"] = clips[i]
        if accumulated_clips[i] is not None:
            ev["ep_rp_findings"] = {"clips": accumulated_clips[i]}
        else:
            ev["ep_rp_findings"] = None
        events.append(ev)
    return events


# ── scenario builders ──────────────────────────────────────────────────────────

def _s1_normal_right() -> dict:
    """Normal right-leg examination — no pathology. Full protocol scan."""
    leg, side = "right", 2
    events = []

    # SFJ (0..10s) — frames from 0%..100% of SFJ video (sample_data)
    r = _scan_readings(side, 0.01, 0.10, True, 3, 0.0, 10.0)
    events += _build_events(r, "SFJ", "SFJ", "scan",
        ["Arrive at SFJ, B-mode survey",
         "SFJ B-mode — junction appears competent",
         "SFJ Color Doppler augmentation — no reflux"],
        pos_start=0.0, pos_end=1.0)

    # GSV-THI (12..28s) — frames from 0%..80% of GSV-THI video (Moving+Perf)
    r = _scan_readings(side, 0.15, 0.85, True, 4, 12.0, 28.0)
    events += _build_events(r, "GSV-THI", "GSV-THI", "scan",
        ["GSV Thigh proximal — B-mode, no dilation",
         "GSV Thigh mid — Color Doppler, no reflux",
         "GSV Thigh distal — Doppler augmentation normal",
         "Approaching knee — GSV Thigh distal normal"],
        pos_start=0.0, pos_end=0.8)

    # Crossing to calf (30..32s)
    r = _scan_readings(side + 1, 0.01, 0.08, True, 1, 30.0, 32.0)
    events += _build_events(r, "GSV-CAL", "GSV-CAL", "region_transition",
        ["Crossing knee — transitioning to GSV Calf"],
        pos_start=0.0, pos_end=0.1)

    # GSV-CAL (32..42s) — frames 0%..100% of Knee-Ankle video
    r = _scan_readings(side + 1, 0.10, 0.70, True, 3, 32.0, 42.0)
    events += _build_events(r, "GSV-CAL", "GSV-CAL", "scan",
        ["GSV Calf proximal — B-mode normal",
         "GSV Calf mid — Color Doppler, no reflux",
         "GSV Calf distal — normal"],
        pos_start=0.1, pos_end=0.9)

    # Patient turns posterior — SPJ (44..52s)
    r = _scan_readings(side + 1, 0.02, 0.09, False, 2, 44.0, 52.0)
    events += _build_events(r, "SPJ", "SPJ", "scan",
        ["Patient turned — SPJ area, B-mode",
         "SPJ Color Doppler — competent, no reflux"],
        pos_start=0.0, pos_end=1.0)

    # SSV (54..62s)
    r = _scan_readings(side + 1, 0.30, 0.65, False, 2, 54.0, 62.0)
    events += _build_events(r, "SSV", "SSV", "scan",
        ["SSV mid-calf — B-mode normal",
         "SSV Color Doppler — normal antegrade flow"],
        pos_start=0.0, pos_end=1.0)

    return {
        "id": "S1",
        "name": "Normal Right Leg Assessment",
        "description": "Complete right-leg examination with no venous pathology found.",
        "leg": leg,
        "shunt_type": None,
        "events": events,
    }


def _s2_type1_right() -> dict:
    """Type 1 shunt — SFJ incompetence + GSV trunk reflux (right leg)."""
    leg, side = "right", 2
    events = []
    clips_so_far: list[dict] = []

    # SFJ — Doppler shows reflux (0..10s, full SFJ video)
    r_sfj = _scan_readings(side, 0.01, 0.06, True, 3, 0.0, 10.0)
    events += _build_events(r_sfj, "SFJ", "SFJ", "scan",
        ["Arrive at SFJ — GSV appears dilated on B-mode",
         "SFJ Color Doppler — sustained reflux visible",
         "Confirming reflux with calf augmentation"],
        pos_start=0.0, pos_end=1.0)

    # Surgeon marks EP N1→N2 at SFJ (at t=10s, same position as last SFJ reading)
    r_ep = _scan_readings(side, 0.05, 0.05, True, 1, 10.0, 10.0)
    c0 = _clip("EP", "N1", "N2", side, r_ep[0]["segment_dist"], leg, 0, "SFJ-Knee")
    clips_so_far.append(c0)
    events += _build_events(r_ep, "SFJ", "SFJ", "ep_rp_mark",
        ["Surgeon marks EP N1→N2 at SFJ"],
        pos_start=0.95, pos_end=1.0,
        clips=[c0], accumulated_clips=[copy.deepcopy(clips_so_far)])

    # Move down thigh — GSV trunk reflux visible (12..26s)
    # Video: 0%..70% of GSV-THI (Moving video shows probe scanning along GSV)
    r_thi = _scan_readings(side, 0.14, 0.55, True, 4, 12.0, 26.0)
    events += _build_events(r_thi, "GSV-THI", "GSV-THI", "scan",
        ["GSV Thigh upper — retrograde flow along trunk",
         "GSV Thigh mid — reflux continues",
         "GSV Thigh — reflux persists, trunk dilated",
         "GSV Thigh distal — confirming reflux extent"],
        pos_start=0.0, pos_end=0.7)

    # Surgeon marks RP N2→N1 at upper thigh (t=18s)
    r_rp = _scan_readings(side, 0.18, 0.18, True, 1, 18.0, 18.0)
    c1 = _clip("RP", "N2", "N1", side, r_rp[0]["segment_dist"], leg, 1, "SFJ-Knee", pos_x=0.22)
    clips_so_far.append(c1)
    events += _build_events(r_rp, "GSV-THI", "GSV-THI", "ep_rp_mark",
        ["Surgeon marks RP N2→N1 — GSV trunk reflux confirmed"],
        pos_start=0.2, pos_end=0.25,
        clips=[c1], accumulated_clips=[copy.deepcopy(clips_so_far)])

    # Calf and SPJ — no further pathology (28..40s)
    r_cal = _scan_readings(side + 1, 0.10, 0.60, True, 2, 28.0, 36.0)
    events += _build_events(r_cal, "GSV-CAL", "GSV-CAL", "scan",
        ["GSV Calf — no further reflux",
         "GSV Calf distal — normal"],
        pos_start=0.1, pos_end=0.8)

    r_spj = _scan_readings(side + 1, 0.03, 0.08, False, 1, 38.0, 40.0)
    events += _build_events(r_spj, "SPJ", "SPJ", "scan",
        ["SPJ — competent, no reflux"],
        pos_start=0.0, pos_end=1.0)

    return {
        "id": "S2",
        "name": "Type 1 Shunt — Right Leg",
        "description": "SFJ incompetent (EP N1→N2). GSV trunk reflux (RP N2→N1). Classic Type 1 pattern.",
        "leg": leg,
        "shunt_type": "Type 1",
        "events": events,
    }


def _s3_type2a_left() -> dict:
    """Type 2A shunt — GSV-to-tributary escape, SFJ competent (left leg)."""
    leg, side = "left", 0
    events = []
    clips_so_far: list[dict] = []

    # SFJ — competent (0..8s)
    r = _scan_readings(side, 0.03, 0.08, True, 2, 0.0, 8.0)
    events += _build_events(r, "SFJ", "SFJ", "scan",
        ["SFJ — normal flow, junction competent",
         "SFJ Color Doppler — no reflux confirmed"],
        pos_start=0.1, pos_end=0.9)

    # GSV-THI — tributary escape visible (10..22s)
    r_thi1 = _scan_readings(side, 0.15, 0.35, True, 2, 10.0, 16.0)
    events += _build_events(r_thi1, "GSV-THI", "GSV-THI", "scan",
        ["GSV mid-thigh — scanning for escape point",
         "GSV mid-thigh — tributary visualised branching off"],
        pos_start=0.0, pos_end=0.5)

    # Surgeon marks EP N2→N3 at mid-thigh (t=16s)
    r_ep = _scan_readings(side, 0.30, 0.30, True, 1, 16.0, 16.0)
    c0 = _clip("EP", "N2", "N3", side, r_ep[0]["segment_dist"], leg, 0, "SFJ-Knee", pos_x=0.23)
    clips_so_far.append(c0)
    events += _build_events(r_ep, "GSV-THI", "GSV-THI", "ep_rp_mark",
        ["Surgeon marks EP N2→N3 — GSV to tributary escape"],
        pos_start=0.45, pos_end=0.55,
        clips=[c0], accumulated_clips=[copy.deepcopy(clips_so_far)])

    # Continue thigh — no RP (no trunk reflux) (18..26s)
    r_thi2 = _scan_readings(side, 0.35, 0.80, True, 2, 18.0, 26.0)
    events += _build_events(r_thi2, "GSV-THI", "GSV-THI", "scan",
        ["GSV Thigh distal — no trunk reflux below escape",
         "Knee approach — GSV normal"],
        pos_start=0.5, pos_end=0.9)

    # Calf normal (28..36s)
    r_cal = _scan_readings(side + 1, 0.10, 0.55, True, 2, 28.0, 36.0)
    events += _build_events(r_cal, "GSV-CAL", "GSV-CAL", "scan",
        ["GSV Calf — normal B-mode",
         "GSV Calf — Color Doppler normal"],
        pos_start=0.1, pos_end=0.9)

    # SPJ (38..42s)
    r_spj = _scan_readings(side + 1, 0.04, 0.09, False, 1, 38.0, 42.0)
    events += _build_events(r_spj, "SPJ", "SPJ", "scan",
        ["SPJ — competent"],
        pos_start=0.0, pos_end=1.0)

    return {
        "id": "S3",
        "name": "Type 2A Shunt — Left Leg",
        "description": "SFJ competent. EP N2→N3 at mid-thigh (GSV-to-tributary escape). No GSV trunk reflux.",
        "leg": leg,
        "shunt_type": "Type 2A",
        "events": events,
    }


def _s4_type3_right() -> dict:
    """Type 3 shunt — EP N1→N2, EP N2→N3, RP N3→N1 bypass (right leg)."""
    leg, side = "right", 2
    events = []
    clips_so_far: list[dict] = []

    # SFJ — incompetent (0..8s)
    r = _scan_readings(side, 0.03, 0.07, True, 2, 0.0, 8.0)
    events += _build_events(r, "SFJ", "SFJ", "scan",
        ["SFJ — junction appears incompetent on B-mode",
         "SFJ Color Doppler — reflux present at junction"],
        pos_start=0.0, pos_end=1.0)

    # EP N1→N2 at SFJ (t=8s)
    r_ep0 = _scan_readings(side, 0.05, 0.05, True, 1, 8.0, 8.0)
    c0 = _clip("EP", "N1", "N2", side, r_ep0[0]["segment_dist"], leg, 0, "SFJ-Knee")
    clips_so_far.append(c0)
    events += _build_events(r_ep0, "SFJ", "SFJ", "ep_rp_mark",
        ["Surgeon marks EP N1→N2 at SFJ"],
        pos_start=0.9, pos_end=1.0,
        clips=[c0], accumulated_clips=[copy.deepcopy(clips_so_far)])

    # Upper thigh — tributary escape (10..18s)
    r_thi1 = _scan_readings(side, 0.14, 0.20, True, 2, 10.0, 16.0)
    events += _build_events(r_thi1, "GSV-THI", "GSV-THI", "scan",
        ["Upper thigh — tributary escape from GSV trunk visible",
         "Tributary confirmed branching off GSV"],
        pos_start=0.0, pos_end=0.35)

    # EP N2→N3 upper thigh (t=16s)
    r_ep1 = _scan_readings(side, 0.16, 0.16, True, 1, 16.0, 16.0)
    c1 = _clip("EP", "N2", "N3", side, r_ep1[0]["segment_dist"], leg, 1, "SFJ-Knee", pos_x=0.22)
    clips_so_far.append(c1)
    events += _build_events(r_ep1, "GSV-THI", "GSV-THI", "ep_rp_mark",
        ["Surgeon marks EP N2→N3 — tributary escape upper thigh"],
        pos_start=0.3, pos_end=0.38,
        clips=[c1], accumulated_clips=[copy.deepcopy(clips_so_far)])

    # Mid thigh — RP via bypass tributary (18..26s)
    r_thi2 = _scan_readings(side, 0.28, 0.40, True, 2, 18.0, 24.0)
    events += _build_events(r_thi2, "GSV-THI", "GSV-THI", "scan",
        ["Mid thigh — tributary refluxing back toward deep system",
         "Bypass tributary confirmed — no RP N2→N1 on main trunk"],
        pos_start=0.4, pos_end=0.65)

    # RP N3→N1 (t=24s)
    r_rp = _scan_readings(side, 0.32, 0.32, True, 1, 24.0, 24.0)
    c2 = _clip("RP", "N3", "N1", side, r_rp[0]["segment_dist"], leg, 2, "SFJ-Knee", pos_x=0.17)
    clips_so_far.append(c2)
    events += _build_events(r_rp, "GSV-THI", "GSV-THI", "ep_rp_mark",
        ["Surgeon marks RP N3→N1 — bypass reflux to deep system"],
        pos_start=0.6, pos_end=0.68,
        clips=[c2], accumulated_clips=[copy.deepcopy(clips_so_far)])

    # Distal thigh — no RP N2→N1 (26..34s)
    r_thi3 = _scan_readings(side, 0.50, 0.85, True, 2, 26.0, 34.0)
    events += _build_events(r_thi3, "GSV-THI", "GSV-THI", "scan",
        ["Distal thigh — GSV trunk not refluxing (no RP N2→N1)",
         "Knee approach — GSV trunk competent below escape"],
        pos_start=0.7, pos_end=1.0)

    return {
        "id": "S4",
        "name": "Type 3 Shunt — Right Leg",
        "description": (
            "EP N1→N2 at SFJ. EP N2→N3 upper thigh. RP N3→N1 bypass. "
            "No RP N2→N1 — GSV trunk not refluxing. Classic Type 3."
        ),
        "leg": leg,
        "shunt_type": "Type 3",
        "events": events,
    }


def _s5_wrong_region_correction() -> dict:
    """Probe starts in SSV zone when GSV-THI expected — system flags, then corrected."""
    leg, side = "left", 0
    events = []
    clips_so_far: list[dict] = []

    # WRONG: probe on posterior calf when GSV-THI expected (0..6s)
    r_wrong = _scan_readings(side + 1, 0.45, 0.55, False, 2, 0.0, 6.0)
    events += _build_events(r_wrong, "SSV", "GSV-THI", "scan",
        ["WRONG REGION — probe on posterior calf (SSV) but GSV-THI expected",
         "Still in wrong region — sonographer repositioning"],
        pos_start=0.3, pos_end=0.7)

    # Correction — move to thigh anterior (8..12s)
    r_fix = _scan_readings(side, 0.20, 0.28, True, 1, 8.0, 12.0)
    events += _build_events(r_fix, "GSV-THI", "GSV-THI", "region_transition",
        ["Correction — patient faces examiner, probe on anterior thigh"],
        pos_start=0.0, pos_end=0.2)

    # GSV-THI — reflux found (12..22s)
    r_thi = _scan_readings(side, 0.25, 0.45, True, 3, 12.0, 22.0)
    events += _build_events(r_thi, "GSV-THI", "GSV-THI", "scan",
        ["GSV Thigh — reflux detected on Color Doppler",
         "GSV Thigh — confirming reflux extent",
         "GSV Thigh — retrograde flow in trunk confirmed"],
        pos_start=0.15, pos_end=0.6)

    # Back to SFJ for confirmation (24..30s)
    r_sfj = _scan_readings(side, 0.03, 0.07, True, 2, 24.0, 30.0)
    events += _build_events(r_sfj, "SFJ", "SFJ", "scan",
        ["Moved back to SFJ — confirming junction incompetence",
         "SFJ reflux confirmed with augmentation"],
        pos_start=0.0, pos_end=1.0)

    # EP N1→N2 at SFJ (t=30s)
    r_ep = _scan_readings(side, 0.05, 0.05, True, 1, 30.0, 30.0)
    c0 = _clip("EP", "N1", "N2", side, r_ep[0]["segment_dist"], leg, 0, "SFJ-Knee")
    clips_so_far.append(c0)
    events += _build_events(r_ep, "SFJ", "SFJ", "ep_rp_mark",
        ["Surgeon marks EP N1→N2 at SFJ"],
        pos_start=0.85, pos_end=1.0,
        clips=[c0], accumulated_clips=[copy.deepcopy(clips_so_far)])

    # RP N2→N1 at thigh (t=34s)
    r_rp = _scan_readings(side, 0.22, 0.22, True, 1, 34.0, 34.0)
    c1 = _clip("RP", "N2", "N1", side, r_rp[0]["segment_dist"], leg, 1, "SFJ-Knee", pos_x=0.22)
    clips_so_far.append(c1)
    events += _build_events(r_rp, "GSV-THI", "GSV-THI", "ep_rp_mark",
        ["Surgeon marks RP N2→N1 — GSV reflux confirmed"],
        pos_start=0.25, pos_end=0.3,
        clips=[c1], accumulated_clips=[copy.deepcopy(clips_so_far)])

    return {
        "id": "S5",
        "name": "Wrong Region + Correction (Left Leg Type 1)",
        "description": (
            "Probe starts on SSV when GSV-THI expected. "
            "System flags wrong region. Corrected — confirms Type 1 on left leg."
        ),
        "leg": leg,
        "shunt_type": "Type 1",
        "events": events,
    }


# ── public API ─────────────────────────────────────────────────────────────────

_SCENARIO_BUILDERS = {
    "S1": _s1_normal_right,
    "S2": _s2_type1_right,
    "S3": _s3_type2a_left,
    "S4": _s4_type3_right,
    "S5": _s5_wrong_region_correction,
}


def generate_scenario(scenario_id: str) -> dict:
    builder = _SCENARIO_BUILDERS.get(scenario_id)
    if not builder:
        raise ValueError(f"Unknown scenario_id {scenario_id!r}. Choose from {list(_SCENARIO_BUILDERS)}")
    return builder()


def generate_all_scenarios() -> list[dict]:
    return [b() for b in _SCENARIO_BUILDERS.values()]


def save_scenarios(output_dir: str = None) -> str:
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "tests", "mock_scenarios"
        )
    os.makedirs(output_dir, exist_ok=True)
    for sid, builder in _SCENARIO_BUILDERS.items():
        scenario = builder()
        safe_name = scenario["name"].replace(" ", "_").replace("/", "-")
        fname = os.path.join(output_dir, f"{sid}_{safe_name}.json")
        with open(fname, "w") as f:
            json.dump(scenario, f, indent=2)
        n_with_img = sum(1 for e in scenario["events"] if e.get("image_b64"))
        print(f"  {fname}  ({len(scenario['events'])} events, {n_with_img} with real frames)")
    return output_dir


if __name__ == "__main__":
    print("Generating mock scenarios...")
    out = save_scenarios()
    print(f"Done — scenarios written to: {out}")
