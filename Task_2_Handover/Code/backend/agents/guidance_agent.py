"""
Guidance agent utilities — state message builder.

build_state_message() assembles the enriched per-turn state message fed
to the CrewAI crew each turn.
Called by streaming_guidance_engine.process_probe_state().

Note: the historical SYSTEM_PROMPT, call_llm() and fallback_guidance()
functions have been removed. The CrewAI 5-agent crew in crew_pipeline.py
is the sole live code path for guidance generation.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE MESSAGE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_state_message(
    region: str,
    pos_y: float,
    surface: str,
    leg: str,
    clips: list[dict],
    vlm_summary: str,
    history_summary: str,
    q_state: str,
    protocol_text: str,
    pos_x: Optional[float] = None,
    is_front: Optional[bool] = None,
) -> str:
    """
    Compose the full enriched state message fed to the CrewAI crew each turn.

    Integrates outputs from all sub-agents:
      history_summary  → from history_agent.build_summary()
      q_state          → from q_state_agent.analyze()
      protocol_text    → from protocol_agent.get_protocol()
      vlm_summary      → from vlm_agent.analyze()
    """
    if clips:
        clip_lines = []
        for c in clips:
            flow = c.get("flow", "?")
            fT   = c.get("from_type", "?")
            tT   = c.get("to_type", "?")
            pY   = float(c.get("pos_y_ratio", 0.0))
            cleg = c.get("leg", "?")
            elim = c.get("elimination_test", "")
            elim_str = f"  [elimTest={elim}]" if elim else ""
            clip_lines.append(f"  • {flow} {fT}→{tT}  posY={pY:.2f}  {cleg} leg{elim_str}")
        clips_text = "\n".join(clip_lines)
    else:
        clips_text = "  None confirmed yet."

    pos_x_str = f" | posX: {pos_x:.2f}" if pos_x is not None else ""
    front_str = (
        f" | is_front: {'yes (anterior face)' if is_front else 'no (posterior face)'}"
        if is_front is not None else ""
    )

    return (
        f"PROBE STATE\n"
        f"Region: {region} | Surface: {surface} | Leg: {leg} | posY: {pos_y:.2f}{pos_x_str}{front_str}\n\n"
        f"CONFIRMED FINDINGS\n{clips_text}\n\n"
        f"VLM FRAME ANNOTATION\n{vlm_summary}\n\n"
        f"{history_summary}\n\n"
        f"{q_state}\n\n"
        f"{protocol_text}"
    )

