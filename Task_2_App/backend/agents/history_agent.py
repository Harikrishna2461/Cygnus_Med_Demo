"""
Session History Summarizer Agent.

Computes raw band-visit and clip data from the session, then calls Groq to
produce a short 2-sentence narrative summary of what has been scanned and
what findings have been confirmed. This narrative is fed into the CrewAI
crew's state message so agents have a human-readable picture of progress.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BANDS: list[tuple[float, float, str]] = [
    (0.00, 0.07, "SFJ/groin"),
    (0.08, 0.20, "upper thigh"),
    (0.21, 0.33, "Hunterian (proximal thigh)"),
    (0.34, 0.47, "Dodd (distal thigh)"),
    (0.48, 0.57, "popliteal/SPJ"),
    (0.58, 0.88, "calf"),
    (0.89, 1.00, "ankle"),
]


def _build_raw_data(session, current_pos_y: float) -> str:
    scan_log: list[dict] = getattr(session, "scan_log", [])
    clips: list[dict]    = getattr(session, "clips", [])

    band_visited: dict[str, bool] = {lbl: False for _, _, lbl in _BANDS}
    band_x_vals: dict[str, list[float]] = {lbl: [] for _, _, lbl in _BANDS}
    for entry in scan_log:
        py = float(entry.get("pos_y", 0.0))
        px = entry.get("pos_x")
        for lo, hi, lbl in _BANDS:
            if lo <= py <= hi:
                band_visited[lbl] = True
                if px is not None:
                    band_x_vals[lbl].append(float(px))
                break

    visited   = [lbl for lbl, v in band_visited.items() if v]
    unvisited = [lbl for lbl, v in band_visited.items() if not v]

    visited_strs = []
    for lbl in visited:
        xs = band_x_vals[lbl]
        if xs:
            visited_strs.append(f"{lbl} (posX {min(xs):.2f}–{max(xs):.2f})")
        else:
            visited_strs.append(lbl)

    lines = []
    lines.append(f"Zones visited: {', '.join(visited_strs) if visited_strs else 'none yet'}")
    lines.append(f"Zones not yet visited: {', '.join(unvisited) if unvisited else 'all covered'}")

    if clips:
        clip_strs = [
            f"{c.get('flow')} {c.get('from_type')}→{c.get('to_type')} "
            f"at posY={float(c.get('pos_y_ratio', 0)):.2f} ({c.get('leg')} leg)"
            + (f" elimTest={c['elimination_test']}" if c.get("elimination_test") else "")
            for c in clips
        ]
        lines.append(f"Confirmed clips: {'; '.join(clip_strs)}")
    else:
        lines.append("Confirmed clips: none")

    return "\n".join(lines)


def build_summary(session, current_pos_y: float) -> str:
    """
    Return a structured scan-progress summary for the CrewAI agents.
    The raw structured data is more useful than an LLM-generated narrative —
    it preserves explicit visited/unvisited lists that the Circuit Analyst reasons from.
    """
    raw_data = _build_raw_data(session, current_pos_y)
    return f"SCAN HISTORY SUMMARY\n{raw_data}"