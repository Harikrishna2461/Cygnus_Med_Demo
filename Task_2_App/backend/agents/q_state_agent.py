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
    if pos <= 0.33:  return f"mid-thigh/Dodd (posY={pos:.2f})"
    if pos <= 0.47:  return f"lower thigh/Hunterian (posY={pos:.2f})"
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
        return (
            "Q1-Q4 STATUS\n"
            "Q1 NOT CONFIRMED — No clips yet.\n"
            "  Next: move probe to groin crease (posY≈0.06), scan transversely at SFJ."
        )

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

    # ── Q1 ────────────────────────────────────────────────────────────────────
    if has_ep_n1_n2:
        c = first("EP", "N1", "N2")
        lines.append(
            f"Q1 CONFIRMED — EP N1→N2 at {_level(float(c.get('pos_y_ratio', 0.06)))}. "
            "Deep blood entering GSV trunk (SFJ or Hunterian incompetent)."
        )
    elif has_ep_n2_n2:
        c = first("EP", "N2", "N2")
        lines.append(
            f"Q1 CONFIRMED — EP N2→N2 at {_level(float(c.get('pos_y_ratio', 0.25)))}. "
            "Perforator feeds GSV directly. SFJ competent."
        )
    elif has_ep_n1_n3:
        c = first("EP", "N1", "N3")
        lines.append(
            f"Q1 CONFIRMED — EP N1→N3 at {_level(float(c.get('pos_y_ratio', 0.25)))}. "
            "Deep blood enters tributary directly — SFJ competent. Pattern: Type 4/5/6."
        )
    elif has_ep_n2_n3:
        lines.append(
            "Q1 PARTIAL — EP N2→N3 confirmed (trunk escape). "
            "SFJ entry NOT yet assessed — move to groin (posY≈0.06) to confirm or exclude N1→N2."
        )
    else:
        lines.append(
            "Q1 NOT CONFIRMED — Move to groin crease (posY≈0.05). "
            "If SFJ competent, also check Dodd zone (posY 0.21–0.33) and Hunterian zone (posY 0.34–0.47) for perforator entry."
        )

    # ── Q2 ────────────────────────────────────────────────────────────────────
    if has_ep_n1_n2 or has_ep_n2_n2:
        if has_rp_n2_n1:
            c = first("RP", "N2", "N1")
            lines.append(
                f"Q2 CONFIRMED — RP N2→N1 at {_level(float(c.get('pos_y_ratio', 0.3)))}. "
                "GSV trunk carries blood backward (toward foot)."
            )
        else:
            ep = first("EP", "N1", "N2") or first("EP", "N2", "N2")
            ep_pos = float(ep.get("pos_y_ratio", 0.06)) if ep else 0.06
            lines.append(
                f"Q2 NOT CONFIRMED — Scan GSV trunk distally from posY≈{ep_pos:.2f} (medial surface). "
                "Apply Paranà: reversed flow = RP N2→N1. No reflux = GSV may be conduit only (Type 3)."
            )

    # ── Q3 ────────────────────────────────────────────────────────────────────
    if has_ep_n1_n2 or has_ep_n2_n2:
        if has_ep_n2_n3:
            c = first("EP", "N2", "N3")
            lines.append(
                f"Q3 CONFIRMED — EP N2→N3 at {_level(float(c.get('pos_y_ratio', 0.3)))}. "
                "Blood escapes GSV into tributary."
            )
        else:
            if has_rp_n2_n1:
                rp = first("RP", "N2", "N1")
                rp_pos = float(rp.get("pos_y_ratio", 0.3)) if rp else 0.3
                lines.append(
                    f"Q3 NOT CONFIRMED — Scan GSV between SFJ and posY≈{rp_pos:.2f} for tributary escape (EP N2→N3). "
                    "N3 visible above fascia while N2 in compartment = escape junction."
                )
            else:
                lines.append(
                    "Q3 NOT CONFIRMED — After Q2 assessment, scan distally along GSV for EP N2→N3."
                )

    # ── Q3 perforator circuit ─────────────────────────────────────────────────
    if has_ep_n2_n2 and not has_ep_n1_n2:
        c = first("EP", "N2", "N2")
        perf_pos = float(c.get("pos_y_ratio", 0.25)) if c else 0.25
        if not has_rp_n3:
            lines.append(
                f"PERFORATOR CIRCUIT — EP N2→N2 at posY≈{perf_pos:.2f}. "
                f"Scan for tributary reflux (RP N3) at posY {max(0.0, perf_pos-0.10):.2f}–{min(1.0, perf_pos+0.15):.2f}. "
                "Also check RP N2→N1 (distinguishes Type 2B from 2C)."
            )

    # ── Q4 ────────────────────────────────────────────────────────────────────
    if has_ep_n2_n3:
        escape = first("EP", "N2", "N3")
        esc_pos = float(escape.get("pos_y_ratio", 0.3)) if escape else 0.3
        if has_rp_n3:
            c_rp = next(
                (c for c in clips if c.get("flow") == "RP" and c.get("from_type") == "N3"),
                None,
            )
            direction = "toward GSV (RP N3→N2)" if has_rp_n3_n2 else "into deep system (RP N3→N1)"
            rp_level = _level(float(c_rp.get("pos_y_ratio", 0.5))) if c_rp else "unknown level"
            lines.append(
                f"Q4 CONFIRMED — RP N3 confirmed: tributary reflux {direction} at {rp_level}."
            )
        else:
            lines.append(
                f"Q4 NOT CONFIRMED — Follow tributary from escape posY≈{esc_pos:.2f}. "
                "Retrograde into deep perforator = RP N3→N1. Retrograde toward GSV = RP N3→N2."
            )

    # ── EP N1→N3 branch (Type 4/5/6) ─────────────────────────────────────────
    if has_ep_n1_n3:
        if has_rp_n2_n1 and not has_rp_n3:
            c = first("EP", "N1", "N3")
            lines.append(
                f"TYPE 4 developing — EP N1→N3 at {_level(float(c.get('pos_y_ratio', 0.25)))} + RP N2→N1. "
                "Trace tributary for RP N3→N2 to confirm full circuit."
            )
        elif not has_rp_n2_n1 and not has_rp_n3:
            c = first("EP", "N1", "N3")
            lines.append(
                f"TYPE 4/5/6 developing — EP N1→N3 at {_level(float(c.get('pos_y_ratio', 0.25)))}. "
                "Trace tributary for re-entry: RP N3→N1 (Type 6, no trunk) or RP N3→N2 then RP N2→N1 (Type 4/5)."
            )

    # ── Type 3 conduit check ──────────────────────────────────────────────────
    if has_ep_n1_n2 and has_ep_n2_n3 and not has_rp_n2_n1:
        escape = first("EP", "N2", "N3")
        esc_pos = float(escape.get("pos_y_ratio", 0.3)) if escape else 0.3
        lines.append(
            f"TYPE 3 CONDUIT CHECK — EP N1→N2 + EP N2→N3 present, NO RP N2→N1. "
            f"GSV may act as conduit to escape at posY≈{esc_pos:.2f}. "
            "Do NOT assume trunk reflux below escape — only record RP N2→N1 if explicitly seen below that posY."
        )

    # ── Elimination test alert ────────────────────────────────────────────────
    if has_ep_n2_n3 and has_rp_n3_n1 and has_rp_n2_n1 and not elim_done:
        lines.append(
            "ELIMINATION TEST REQUIRED — EP N2→N3 + RP N3→N1 + RP N2→N1 all confirmed, no elimTest yet. "
            "Compress the tributary at the escape site. "
            "If GSV Doppler disappears → Type 3 (No Reflux). If persists → Type 1+2 (Reflux)."
        )

    return "\n".join(lines)