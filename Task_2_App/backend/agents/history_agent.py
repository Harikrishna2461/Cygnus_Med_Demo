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
    (0.21, 0.33, "mid-thigh/Dodd"),
    (0.34, 0.47, "lower thigh/Hunterian"),
    (0.48, 0.57, "popliteal/SPJ"),
    (0.58, 0.88, "calf"),
    (0.89, 1.00, "ankle"),
]

_SYSTEM_PROMPT = (
    "You are a CHIVA duplex examination assistant. "
    "Given a structured scan log, write exactly 2 short sentences: "
    "sentence 1 — which anatomical zones have been scanned so far; "
    "sentence 2 — which EP/RP findings have been confirmed (or 'No findings confirmed yet'). "
    "Be concise and clinical. No bullet points. No labels. Plain sentences only."
)


def _build_raw_data(session, current_pos_y: float) -> str:
    scan_log: list[dict] = getattr(session, "scan_log", [])
    clips: list[dict]    = getattr(session, "clips", [])

    band_visited: dict[str, bool] = {lbl: False for _, _, lbl in _BANDS}
    for entry in scan_log:
        py = float(entry.get("pos_y", 0.0))
        for lo, hi, lbl in _BANDS:
            if lo <= py <= hi:
                band_visited[lbl] = True
                break

    visited   = [lbl for lbl, v in band_visited.items() if v]
    unvisited = [lbl for lbl, v in band_visited.items() if not v]

    lines = []
    lines.append(f"Zones visited: {', '.join(visited) if visited else 'none yet'}")
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


def _call_llm(raw_data: str) -> str:
    from config import GROQ_API_KEY, GROQ_TEXT_MODEL
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": raw_data},
        ],
        max_tokens=80,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def build_summary(session, current_pos_y: float) -> str:
    """
    Return a short LLM-generated narrative summary of scan progress and findings.
    Falls back to a plain structured string if the LLM call fails.
    """
    raw_data = _build_raw_data(session, current_pos_y)
    try:
        summary = _call_llm(raw_data)
        return f"SCAN HISTORY SUMMARY\n{summary}"
    except Exception as exc:
        logger.warning("History agent LLM call failed, using raw data: %s", exc)
        return f"SCAN HISTORY SUMMARY\n{raw_data}"
