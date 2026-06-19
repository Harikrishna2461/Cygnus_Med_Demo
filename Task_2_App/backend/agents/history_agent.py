"""
Session History Summarizer Agent.

Builds a compact structured summary at each timestep capturing:
  - Which posY bands have been visited this session (and how many times)
  - Which bands have NOT yet been visited
  - All confirmed findings (clips) with their posY, leg, and elimTest

Fed to the guidance LLM as part of the enriched state message so the model
always knows what anatomy has already been covered vs what remains.
"""
from __future__ import annotations

# posY band definitions (boundaries inclusive)
# Sources: DuplexUS 2014 p.33-34 (Dodd/Hunterian), Lee 2017 p.129, Adler 2022
_BANDS: list[tuple[float, float, str]] = [
    (0.00, 0.07, "SFJ/groin               (posY 0.00–0.07)"),
    (0.08, 0.20, "Upper thigh             (posY 0.08–0.20)"),
    (0.21, 0.33, "Mid-thigh / Dodd        (posY 0.21–0.33)"),
    (0.34, 0.47, "Lower thigh / Hunterian (posY 0.34–0.47)"),
    (0.48, 0.57, "Popliteal / SPJ         (posY 0.48–0.57)"),
    (0.58, 0.88, "Calf                    (posY 0.58–0.88)"),
    (0.89, 1.00, "Ankle / lower calf      (posY 0.89–1.00)"),
]


def build_summary(session, current_pos_y: float) -> str:
    """
    Build a structured scan history summary from session.scan_log and session.clips.

    Args:
        session:        StreamSession — provides scan_log and clips attributes.
        current_pos_y:  Current probe position (already logged by caller before this call).

    Returns:
        Multi-line string ready to embed in the LLM state message.
    """
    scan_log: list[dict] = getattr(session, "scan_log", [])
    clips: list[dict] = getattr(session, "clips", [])

    # Tally visits and deepest posY per band
    band_count: dict[str, int]   = {lbl: 0 for _, _, lbl in _BANDS}
    band_max:   dict[str, float] = {lbl: 0.0 for _, _, lbl in _BANDS}

    for entry in scan_log:
        py = float(entry.get("pos_y", 0.0))
        for lo, hi, lbl in _BANDS:
            if lo <= py <= hi:
                band_count[lbl] += 1
                if py > band_max[lbl]:
                    band_max[lbl] = py
                break

    lines = ["SCAN HISTORY SUMMARY"]
    lines.append("Zone coverage this session:")
    for _, _, lbl in _BANDS:
        cnt = band_count[lbl]
        if cnt > 0:
            lines.append(f"  [DONE] {lbl} — {cnt} scan(s), deepest posY {band_max[lbl]:.2f}")
        else:
            lines.append(f"  [    ] {lbl} — NOT YET VISITED")

    # Confirmed clips
    if clips:
        lines.append(f"\nConfirmed findings this session ({len(clips)}):")
        for c in clips:
            flow = c.get("flow", "?")
            ft   = c.get("from_type", "?")
            tt   = c.get("to_type", "?")
            py   = float(c.get("pos_y_ratio", 0.0))
            leg  = c.get("leg", "?")
            elim = c.get("elimination_test", "")
            elim_str = f"  [elimTest={elim}]" if elim else ""
            lines.append(f"  • {flow} {ft}→{tt}  posY={py:.2f}  {leg} leg{elim_str}")
    else:
        lines.append("\nNo clips confirmed yet this session.")

    return "\n".join(lines)
