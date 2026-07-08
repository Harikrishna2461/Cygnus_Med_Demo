"""
Q1-Q4 Circuit Status Agent.

Derives the current CHIVA diagnostic state from the confirmed clip set.
Ported from _chiva_component_status() in active_guidance_engine.py and
extended to include Type 3 conduit check and elimination test alerts.

Q1: Where does blood enter the superficial system?
Q2: Does blood reflux down the saphenous trunk?
Q3: Does blood escape the trunk into a tributary?
Q4: Where does that tributary re-enter the deep system?
"""
from __future__ import annotations

from typing import Optional


def _level(pos: float) -> str:
    if pos <= 0.07:  return f"SFJ/groin (posY={pos:.2f})"
    if pos <= 0.20:  return f"upper thigh (posY={pos:.2f})"
    if pos <= 0.33:  return f"Hunterian/proximal thigh (posY={pos:.2f})"
    if pos <= 0.47:  return f"Dodd/distal thigh (posY={pos:.2f})"
    if pos <= 0.57:  return f"popliteal/SPJ (posY={pos:.2f})"
    if pos <= 0.88:  return f"calf (posY={pos:.2f})"
    return f"ankle (posY={pos:.2f})"


def analyze(clips: list[dict]) -> str:
    """
    Analyze confirmed clips and return a Q1-Q4 status block.

    Args:
        clips: list of clip dicts from session.clips — each has
               flow, from_type, to_type, pos_y_ratio, leg, elimination_test.

    Returns:
        Multi-line Q1-Q4 status string ready to embed in the LLM state message.
    """
    if not clips:
        return "Q1-Q4 STATUS\nQ1: OPEN\nQ2: N/A\nQ3: N/A\nQ4: N/A"

    def has(flow: str, fT: str, tT: str) -> bool:
        return any(
            c.get("flow") == flow and c.get("from_type") == fT and c.get("to_type") == tT
            for c in clips
        )

    def first(flow: str, fT: str, tT: str) -> Optional[dict]:
        return next(
            (c for c in clips
             if c.get("flow") == flow and c.get("from_type") == fT and c.get("to_type") == tT),
            None,
        )

    has_ep_n1_n2 = has("EP", "N1", "N2")
    has_ep_n2_n2 = has("EP", "N2", "N2")
    has_ep_n2_n3 = has("EP", "N2", "N3")
    has_ep_n1_n3 = has("EP", "N1", "N3")
    has_rp_n2_n1 = has("RP", "N2", "N1")
    has_rp_n3_n2 = has("RP", "N3", "N2")
    has_rp_n3_n1 = has("RP", "N3", "N1")
    has_rp_n3    = has_rp_n3_n2 or has_rp_n3_n1

    elim_done = any(
        c.get("flow") == "EP" and c.get("from_type") == "N2" and c.get("to_type") == "N3"
        and c.get("elimination_test", "")
        for c in clips
    )

    lines = ["Q1-Q4 STATUS"]

    # Q1
    if has_ep_n1_n2:
        c = first("EP", "N1", "N2")
        lines.append(f"Q1: CONFIRMED — EP N1→N2 at {_level(float(c.get('pos_y_ratio', 0.06)))}.")
    elif has_ep_n2_n2:
        c = first("EP", "N2", "N2")
        lines.append(f"Q1: CONFIRMED — EP N2→N2 at {_level(float(c.get('pos_y_ratio', 0.25)))}.")
    elif has_ep_n1_n3:
        c = first("EP", "N1", "N3")
        lines.append(f"Q1: CONFIRMED — EP N1→N3 at {_level(float(c.get('pos_y_ratio', 0.25)))}.")
    elif has_ep_n2_n3:
        lines.append("Q1: PARTIAL — EP N2→N3 found; SFJ/Hunterian not yet assessed.")
    else:
        lines.append("Q1: OPEN")

    # Q2
    if has_ep_n1_n2 or has_ep_n2_n2:
        if has_rp_n2_n1:
            c = first("RP", "N2", "N1")
            lines.append(f"Q2: CONFIRMED — RP N2→N1 at {_level(float(c.get('pos_y_ratio', 0.3)))}.")
        else:
            lines.append("Q2: OPEN")
    else:
        lines.append("Q2: N/A")

    # Q3
    if has_ep_n1_n2 or has_ep_n2_n2:
        if has_ep_n2_n3:
            c = first("EP", "N2", "N3")
            lines.append(f"Q3: CONFIRMED — EP N2→N3 at {_level(float(c.get('pos_y_ratio', 0.3)))}.")
        else:
            lines.append("Q3: OPEN")
    elif has_ep_n2_n3:
        c = first("EP", "N2", "N3")
        lines.append(f"Q3: CONFIRMED — EP N2→N3 at {_level(float(c.get('pos_y_ratio', 0.3)))}.")
    else:
        lines.append("Q3: N/A")

    # Q4
    if has_ep_n2_n3:
        if has_rp_n3:
            c_rp = next((c for c in clips if c.get("flow") == "RP" and c.get("from_type") == "N3"), None)
            direction = "RP N3→N2" if has_rp_n3_n2 else "RP N3→N1"
            rp_level  = _level(float(c_rp.get("pos_y_ratio", 0.5))) if c_rp else "unknown"
            lines.append(f"Q4: CONFIRMED — {direction} at {rp_level}.")
        else:
            lines.append("Q4: OPEN")
    else:
        lines.append("Q4: N/A")

    # EP N1→N3 type hint (factual only)
    if has_ep_n1_n3:
        rp_n2_str = "RP N2→N1 confirmed" if has_rp_n2_n1 else "RP N2→N1 not confirmed"
        rp_n3_str = "RP N3 confirmed" if has_rp_n3 else "RP N3 not confirmed"
        lines.append(f"TYPE 4/5/6 pattern — {rp_n2_str}; {rp_n3_str}.")

    # Perforator circuit factual note
    if has_ep_n2_n2 and not has_ep_n1_n2:
        rp_n3_str = "RP N3 confirmed" if has_rp_n3 else "RP N3 not confirmed"
        rp_n2_str = "RP N2→N1 confirmed" if has_rp_n2_n1 else "RP N2→N1 not confirmed"
        lines.append(f"PERFORATOR ENTRY — {rp_n3_str}; {rp_n2_str}.")

    # Elimination test status (factual only)
    if has_ep_n1_n2 and has_ep_n2_n3 and has_rp_n3_n1 and has_rp_n2_n1:
        if not elim_done:
            lines.append("ELIM TEST: PENDING — four clips present, no elimTest recorded.")
        else:
            lines.append("ELIM TEST: DONE — result recorded on EP N2→N3 clip.")

    return "\n".join(lines)