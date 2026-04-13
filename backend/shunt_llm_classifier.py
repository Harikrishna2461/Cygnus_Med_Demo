"""
Shunt Classifier — Rule-Based Engine (ShuntManager) + LLM narrative.

Primary classification is done deterministically using the CHIVA rule-based
algorithm (from the clinical specification). The LLM is used ONLY to generate
a short plain-English clinical summary sentence — it is never trusted for
structured output.

Post-assessment batch processing: accepts a clip_list (15–20 points),
optionally splits by legSide, and returns one Finding per leg.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# RULE-BASED SHUNT CLASSIFICATION ENGINE
# Faithfully implements the decision tree from the clinical spec.
# ─────────────────────────────────────────────────────────────

# SFJ / Hunterian region thresholds (normalised y-coordinate)
SFJ_MAX_Y = 0.098
HUNTERIAN_MAX_Y = 0.353


def _classify_group(clips: list[dict]) -> dict:
    """
    Run the CHIVA rule-based classification on a group of clips.
    Returns a structured result dict.
    """
    # ── 1. Collect boolean flags ─────────────────────────────
    ep12 = ep23 = ep22 = False
    rp32 = rp31 = rp21 = rp3 = False
    many_rp3 = many_rp21 = False
    ep12_at_sfj = ep12_at_hunterian = False
    elim_test_done = needs_elim_test = reflux_persists = False

    ep12_indexes: list[int] = []
    ep23_indexes: list[int] = []
    rp21_indexes: list[tuple[int, float]] = []   # (idx, y)
    ep23_y_indexes: list[tuple[int, float]] = []  # for Type 2 highest-EP logic

    for idx, c in enumerate(clips):
        flow = c.get("flow", "")
        ft = c.get("fromType", "")
        tt = c.get("toType", "")
        y = c.get("posYRatio") or 0.0
        elim = (c.get("eliminationTest") or "").strip()

        # EP N1→N2
        if flow == "EP" and ft == "N1" and tt == "N2":
            ep12 = True
            ep12_indexes.append(idx)
            if y <= SFJ_MAX_Y:
                ep12_at_sfj = True
            elif y <= HUNTERIAN_MAX_Y:
                ep12_at_hunterian = True

        # EP N2→N3
        if flow == "EP" and ft == "N2" and tt == "N3":
            ep23 = True
            ep23_indexes.append(idx)
            ep23_y_indexes.append((idx, y))
            if elim:
                elim_test_done = True
                if elim == "Reflux":
                    reflux_persists = True
                elif elim == "No Reflux":
                    reflux_persists = False

        # EP N2→N2 (mid-GSV entry, Type 2)
        if flow == "EP" and ft == "N2" and tt == "N2":
            ep22 = True
            ep23_y_indexes.append((idx, y))

        # RP N3→N2
        if flow == "RP" and ft == "N3" and tt == "N2":
            rp32 = True
            if rp3:
                many_rp3 = True
            else:
                rp3 = True

        # RP N3→N1
        if flow == "RP" and ft == "N3" and tt == "N1":
            rp31 = True
            if rp3:
                many_rp3 = True
            else:
                rp3 = True

        # RP N2→N1
        if flow == "RP" and ft == "N2" and tt == "N1":
            rp21_indexes.append((idx, y))
            if rp21:
                many_rp21 = True
            else:
                rp21 = True

    # ── 2. Decision tree ────────────────────────────────────
    st1 = st3 = st12 = st2a = st2b = st2c = False
    st_detected = True
    reasoning: list[str] = []
    ligation: list[str] = []

    if ep12_at_sfj or ep12_at_hunterian:
        if ep12_at_sfj:
            reasoning.append("EP exists from N1 to N2 at SFJ")
        if ep12_at_hunterian:
            reasoning.append("EP exists from N1 to N2 at Hunterian perforator")

        if not ep23:
            reasoning.append("No EP exists from N2 to N3")
            if rp21 and not (rp31 or rp32):
                st1 = True
                reasoning.append("RP exists along N2, but no RP exists along N3")
            else:
                st_detected = False

        elif ep23:
            reasoning.append("EP exists from N2 to N3")

            if (rp32 or rp31) and not rp21:
                st3 = True
                reasoning.append("RP exists along N3, but no RP exists along N2")

            elif rp32 and rp21:
                st3 = True
                reasoning.append("RP exists along N2, and through N3 to N2")

            elif rp31 and rp21:
                reasoning.append("RP exists along both N2 and N3, and towards N1")
                if not elim_test_done:
                    needs_elim_test = True
                    reasoning.append("Elimination test missing — required to distinguish Type 3 from Type 1+2")
                elif reflux_persists:
                    st12 = True
                    reasoning.append("Reflux persists in elimination test")
                else:
                    st3 = True
                    reasoning.append("Reflux does not persist in elimination test")
            else:
                st_detected = False
        else:
            st_detected = False

    elif ep23 and not ep12:
        st2a = True
        reasoning.append("EP exists from N2 to N3, but no EP exists from N1 to N2")
        if (rp32 or rp31) and not rp21:
            st2b = True
            st2a = False
            reasoning.append("RP exists along N3, but no RP exists along N2")
        elif (rp32 or rp31) and rp21:
            st2c = True
            st2a = False
            reasoning.append("RP exists along both N2 (GSV) and N3")
        else:
            reasoning.append("RP pattern does not match Type 2B or 2C")

    elif ep22 and not ep12:
        reasoning.append("EP starts at mid GSV from N2 to N2")
        if (rp32 or rp31) and not rp21:
            st2b = True
            reasoning.append("RP exists along N3, but no RP exists along N2")
        elif (rp32 or rp31) and rp21:
            st2c = True
            reasoning.append("RP exists along both N2 (GSV) and N3")
        else:
            st_detected = False
    else:
        st_detected = False

    if not st_detected and not needs_elim_test:
        reasoning = ["No pathological reflux pattern detected in this clip group"]

    # ── 3. Ligation plan ───────────────────────────────────
    if st1:
        if ep12_at_sfj:
            ligation.append("Ligation at the SFJ")
        elif ep12_at_hunterian:
            ligation.append("Ligation at the Hunterian Perforator closer to N2")
        if many_rp21:
            ligation.append("Ligate below each RP along N2 except the most distal RP")

    if st12:
        if not many_rp21:
            ligation.append("Apply CHIVA 2: ligate EP at N2→N3 first, then SFJ/Hunterian")
            ligation.append("OR ligate SFJ/Hunterian first and every refluxing tributary except one")
            ligation.append("Once N2 normalises, ligate the last refluxing tributary")
        else:
            ligation.append("Ligate SFJ or Hunterian perforator and every refluxing tributary")
            ligation.append("Ligate below each RP along N2 except the most distal RP")

    if st3:
        if not many_rp3:
            ligation.append("Ligate the EP at N2→N3")
        else:
            ligation.append("Ligate every refluxing tributary where it joins the N2 (CHIVA 2 first step)")
        if ep12_at_sfj:
            ligation.append("Follow up from 6 weeks to 6/12 months to assess reflux at proximal N2")
            ligation.append("If there is reflux, the SFJ must be treated/ligated")
        if ep12_at_hunterian:
            ligation.append("Assess reflux status at Hunterian perforator at follow-up")

    if st2a or st2b or st2c:
        ligation.append("Ligate highest EP at N2→N3")
        if many_rp3:
            ligation.append("If more branching along N3, ligate based on calibre, distance to perforator, and drainage")

    if needs_elim_test:
        ligation = ["Elimination test required at EP N2→N3 before ligation decision"]

    if not st_detected and not needs_elim_test:
        ligation = []

    # ── 4. Determine shunt type label ─────────────────────
    if st1:
        shunt_type = "Type 1"
        confidence = 0.95
    elif st3:
        shunt_type = "Type 3"
        confidence = 0.93
    elif st12:
        shunt_type = "Type 1+2"
        confidence = 0.90
    elif st2c:
        shunt_type = "Type 2C"
        confidence = 0.88
    elif st2b:
        shunt_type = "Type 2B"
        confidence = 0.88
    elif st2a:
        shunt_type = "Type 2A"
        confidence = 0.85
    elif needs_elim_test:
        shunt_type = "Undetermined — Elimination Test Required"
        confidence = 0.50
    else:
        shunt_type = "No shunt detected"
        confidence = 0.92

    return {
        "shunt_type": shunt_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "ligation": ligation,
        "needs_elim_test": needs_elim_test,
        "ask_diameter": st12 and not many_rp21,
        "ask_branching": (st2a or st2b or st2c) and many_rp3,
        "ask_reflux_perforator": st3 and ep12_at_hunterian,
        "flags": {
            "ep12_at_sfj": ep12_at_sfj,
            "ep12_at_hunterian": ep12_at_hunterian,
            "ep23": ep23,
            "rp32": rp32,
            "rp31": rp31,
            "rp21": rp21,
            "many_rp3": many_rp3,
            "many_rp21": many_rp21,
            "elim_test_done": elim_test_done,
            "reflux_persists": reflux_persists,
        },
    }


# ─────────────────────────────────────────────────────────────
# LLM NARRATIVE GENERATOR (single sentence only)
# ─────────────────────────────────────────────────────────────

def _llm_summary(shunt_type: str, reasoning: list[str], ligation: list[str],
                 call_llm_fn) -> str:
    """Ask the LLM for a single clinical summary sentence. Safe: short prompt, short output."""
    if shunt_type in ("No shunt detected",):
        return "No pathological venous reflux pattern was identified in this assessment."

    reasoning_str = "; ".join(reasoning[:3])
    ligation_str = ligation[0] if ligation else "conservative management"

    prompt = (
        f"Write exactly ONE clinical sentence (under 30 words) summarising this venous shunt finding. "
        f"No headers, no bullets, no lists.\n"
        f"Shunt: {shunt_type}. Evidence: {reasoning_str}. Primary treatment: {ligation_str}.\n"
        f"Sentence:"
    )
    try:
        raw = call_llm_fn(prompt, stream=False)
        # Take only first sentence
        sentence = raw.strip().split("\n")[0].strip()
        # Truncate if too long
        if len(sentence) > 250:
            sentence = sentence[:247] + "..."
        return sentence
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        return f"{shunt_type} detected with {len(ligation)} proposed ligation step(s)."


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def classify_shunt_with_llm(
    clip_list: list[dict[str, Any]],
    call_llm_fn,
) -> dict:
    """
    Classify shunts from a post-assessment clip_list.

    Splits clips by legSide (left / right / unknown) and runs the
    rule-based engine on each group independently.  The LLM is called
    once per finding to produce a plain-English summary sentence.

    Returns a dict with:
      - findings: list of per-leg Finding dicts
      - num_clips: total clips processed
      - (top-level keys mirroring the primary finding for backward compat)
    """
    # Split by legSide
    groups: dict[str, list[dict]] = {}
    for c in clip_list:
        side = (c.get("legSide") or c.get("leg_side") or "unspecified").strip().lower()
        groups.setdefault(side, []).append(c)

    # If no legSide info at all, treat as one group
    if list(groups.keys()) == ["unspecified"]:
        groups = {"Assessment": clip_list}

    findings = []
    for leg_label, group in groups.items():
        label = leg_label.capitalize()
        result = _classify_group(group)
        result["leg"] = label
        result["num_clips"] = len(group)

        # LLM summary (short, safe)
        result["summary"] = _llm_summary(
            result["shunt_type"],
            result["reasoning"],
            result["ligation"],
            call_llm_fn,
        )
        findings.append(result)

    # Sort: left first, right second, others last
    _order = {"Left": 0, "Right": 1}
    findings.sort(key=lambda f: _order.get(f["leg"], 99))

    # Build backward-compat top-level keys from primary finding
    primary = findings[0] if findings else {}

    return {
        # Multi-finding
        "findings": findings,
        # Backward-compat single-finding keys
        "shunt_type": primary.get("shunt_type", "No shunt detected"),
        "confidence": primary.get("confidence", 0.0),
        "reasoning": primary.get("reasoning", []),
        "ligation": primary.get("ligation", []),
        "needs_elim_test": primary.get("needs_elim_test", False),
        "ask_diameter": primary.get("ask_diameter", False),
        "ask_branching": primary.get("ask_branching", False),
        "summary": primary.get("summary", ""),
        "num_clips": len(clip_list),
        "num_findings": len(findings),
    }
