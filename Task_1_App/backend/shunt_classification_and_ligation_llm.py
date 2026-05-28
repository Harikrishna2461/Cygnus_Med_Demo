"""
Unified Shunt Classification and Ligation LLM Module
=====================================================

This module separates two distinct tasks:
1. SHUNT CLASSIFICATION — No RAG, only CHIVA rules (embedded in prompt)
2. LIGATION PLANNING — With RAG from ligation_knowledgebase_db

Each task has its own LLM call with separate prompts and configurations.
"""

import json
import re
import logging
from contextlib import suppress
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE CHIVA RULES (embedded knowledge — used in shunt classification only)
# ─────────────────────────────────────────────────────────────────────────────
CHIVA_RULES = """
=== CHIVA VENOUS SHUNT CLASSIFICATION RULES ===

ANATOMY:
    N1 = Deep venous system (femoral / popliteal vein)
    N2 = Great Saphenous Vein (GSV) or Small Saphenous Vein (SSV) trunk
    N3 = Tributaries / superficial branches
    EP = Physiological (forward, antegrade) flow — NORMAL clip
    RP = Retrograde (pathological, reflux) flow — ABNORMAL clip
    SFJ = Saphenofemoral Junction  →  posYRatio ≤ 0.098
    Hunterian Perforator            →  0.098 < posYRatio ≤ 0.353

═══════════════════════════════════════════════════════════
CRITICAL RULE — SFJ COMPETENCE (read before classifying):
    SFJ is INCOMPETENT if and only if a clip has fromType=N1 AND toType=N2 (EP N1→N2).
    EP N2→N2 means blood circulates within the saphenous trunk via a perforator — SFJ REMAINS COMPETENT.
    This is true regardless of posYRatio or step label. Even posYRatio=0.05 with step=SFJ-Knee
    is a perforator entry if the clip reads EP N2→N2, NOT EP N1→N2.
═══════════════════════════════════════════════════════════

STEP 1 — CHECK FOR EP N1→N2:
    Scan ALL clips. Does any clip have flow=EP, fromType=N1, toType=N2?
    YES → SFJ/Hunterian INCOMPETENT → go to Case A or B.
    NO  → SFJ COMPETENT → go to Case C.

─────────────────────────────────────────────────────────
Case A — EP N1→N2 EXISTS (SFJ or Hunterian), NO EP N2→N3
─────────────────────────────────────────────────────────
    If RP N2→N1 present AND no RP at N3 (no RP N3→N2, no RP N3→N1) → TYPE 1
    Ligation: Ligate at SFJ (y≤0.098) or Hunterian (y≤0.353).
            If multiple RP N2→N1: ligate below each except the most distal.

─────────────────────────────────────────────────────────
Case B — EP N1→N2 EXISTS (SFJ or Hunterian) AND EP N2→N3 EXISTS
─────────────────────────────────────────────────────────
    B1: RP N3→N2 or RP N3→N1, NO RP N2→N1               → TYPE 3
    B2: RP N3→N2 AND RP N2→N1                             → TYPE 3
    B3: RP N3→N1 AND RP N2→N1, eliminationTest absent    → UNDETERMINED (set needs_elim_test=true)
    B4: RP N3→N1 AND RP N2→N1, eliminationTest="Reflux"  → TYPE 1+2
    B5: RP N3→N1 AND RP N2→N1, eliminationTest="No Reflux" → TYPE 3

    TYPE 3 Ligation:
        Single RP at N3: Ligate EP at N2→N3. Follow up 6–12 months; if N2 reflux develops, ligate SFJ.
        Multiple RP at N3: Ligate every refluxing tributary at N2 junction (CHIVA 2 step 1). Same follow-up.

    TYPE 1+2 Ligation — depends on RP N2→N1 calibre:
        Small RP N2→N1: Apply CHIVA 2 (ligate EP N2→N3 first, then SFJ/Hunterian).
                        OR ligate SFJ first + all tributaries except one; once N2 normalises ligate last.
        Large / multiple RP N2→N1: Ligate SFJ/Hunterian + every refluxing tributary simultaneously.
                                    Ligate below each RP N2→N1 except the most distal.

─────────────────────────────────────────────────────────
Case C — NO EP N1→N2 ANYWHERE (SFJ COMPETENT)
─────────────────────────────────────────────────────────
    C-Sub-check: what type of EP clip exists?

    ── TYPE 2A ── EP N2→N3 present, NO EP N1→N2
        The defining feature is EP N2→N3 (GSV feeding a tributary) without any SFJ entry.
        RP may or may not be present in early/developing cases.
        Typical pattern: EP N2→N3 + RP N3→N2 or N3→N1. No RP N2→N1.
        Key signal: EP N2→N3 clip exists + NO EP N1→N2 clip exists anywhere.
        If multiple RP at N3 → set ask_branching=true (need calibre/distance/drainage info).
        Ligation: Ligate highest EP at N2→N3 junction.
                    If multiple branching at N3: ligate based on calibre, distance to perforator, drainage.

    ── TYPE 2B ── EP N2→N2 present, NO EP N1→N2, RP at N3, NO RP N2→N1
        Entry is via perforator (fromType=N2, toType=N2 — NOT N1→N2).
        IMPORTANT: EP N2→N2 at ANY posYRatio (even 0.05, SFJ-Knee step) = perforator, NOT SFJ.
        Key signal: EP N2→N2 clip + RP N3→N2 or N3→N1 + NO EP N1→N2 + NO RP N2→N1.
        If multiple RP at N3 → set ask_branching=true.
        Ligation: Ligate the highest EP N2→N2 (perforator entry point).

    ── TYPE 2C ── EP N2→N2 present, NO EP N1→N2, RP at N3, RP N2→N1 ALSO present
        Perforator entry (EP N2→N2) with secondary GSV reflux (RP N2→N1). SFJ still competent.
        IMPORTANT: 2C has EP N2→N2 (perforator), while Type 1+2 has EP N1→N2 (SFJ entry).
        If NO EP N1→N2 but RP N2→N1 exists with EP N2→N2 → TYPE 2C, not Type 1+2.
        Key signal: EP N2→N2 + RP N3 + RP N2→N1 + NO EP N1→N2.
        Ligation: Ligate perforator entry (highest EP N2→N2) AND all RP N2→N1 sites along GSV.

    Case C — NO SHUNT:
        If EP N2→N2 exists but NO RP clips of any kind → NO SHUNT DETECTED.

─────────────────────────────────────────────────────────
Case D — No RP in any clip → NO SHUNT DETECTED. No ligation needed.
─────────────────────────────────────────────────────────

QUICK DECISION TABLE (commit this to memory):
    Has EP N1→N2? YES + no EP N2→N3 + RP N2→N1           → TYPE 1
    Has EP N1→N2? YES + EP N2→N3 + RP N3 only             → TYPE 3
    Has EP N1→N2? YES + EP N2→N3 + RP N3 + RP N2→N1 + eliminationTest absent → UNDETERMINED
    Has EP N1→N2? YES + EP N2→N3 + RP N3 + RP N2→N1 + elim="Reflux"          → TYPE 1+2
    Has EP N1→N2? YES + EP N2→N3 + RP N3 + RP N2→N1 + elim="No Reflux"       → TYPE 3
    Has EP N1→N2? YES + EP N2→N3 + ZERO RP clips          → NO SHUNT (not Type 3 — Type 3 requires RP)
    Has EP N1→N2? YES + no EP N2→N3 + ZERO RP clips       → NO SHUNT
    No EP N1→N2  + EP N2→N3                                → TYPE 2A
    No EP N1→N2  + EP N2→N2 + RP N3 + NO RP N2→N1         → TYPE 2B
    No EP N1→N2  + EP N2→N2 + RP N3 + RP N2→N1            → TYPE 2C
    No EP N1→N2  + EP N2→N2 + NO RP                        → NO SHUNT
    EP N1→N3 + RP N2→N1                                    → TYPE 4
    EP N1→N3 + RP N3→N2 or RP N3→N1                         → TYPE 5
    No RP at all (except Type 2A)                          → NO SHUNT

CONCRETE EXAMPLES (match these patterns exactly):
    Type 1:  [EP N1→N2 y=0.06 SFJ-ENTRY, RP N2→N1 y=0.25]
            → EP N1→N2 present, RP N2→N1, no EP N2→N3, no N3 reflux → TYPE 1
    Type 2A: [EP N2→N3 y=0.20]  OR  [EP N2→N3 y=0.20, RP N3→N2 y=0.47]
            → No EP N1→N2, EP N2→N3 present → TYPE 2A
    Type 2B: [EP N2→N2 y=0.050 step=SFJ-Knee ligation-point-marker, RP N3→N1 y=0.132]
            → No EP N1→N2, EP N2→N2 = perforator, RP N3 only → TYPE 2B
    Type 2C: [EP N2→N2 y=0.050 step=SFJ-Knee ligation-point-marker, RP N3→N1 y=0.132, RP N2→N1 y=0.212]
            → No EP N1→N2, EP N2→N2 = perforator, RP N3 + RP N2→N1 → TYPE 2C
    Type 3:  [EP N1→N2 y=0.05 SFJ-ENTRY, EP N2→N3 y=0.132 ligation-point-marker, RP N3→N1 y=0.212]
            → EP N1→N2 + EP N2→N3 + RP N3→N1, no RP N2→N1 → TYPE 3
        Type 4:  [EP N1→N3 y=0.60, RP N2→N1 y=0.40]
            → EP N1→N3 with N2 return → TYPE 4
        Type 5:  [EP N1→N3 y=0.65, RP N3→N2 y=0.50, RP N3→N1 y=0.75]
            → EP N1→N3 with looping N3 return → TYPE 5
    Type 3 variant 2 (no elim test):
            [EP N1→N2, EP N2→N3, RP N3→N1, RP N2→N1, no eliminationTest] → UNDETERMINED
    Type 1+2:[EP N1→N2, EP N2→N3 eliminationTest="Reflux", RP N3→N1, RP N2→N1] → TYPE 1+2
    No shunt:[EP N1→N2 only, no RP]  OR  [EP N2→N2 only, no RP] → NO SHUNT
    No shunt:[EP N1→N2 + EP N2→N3, ZERO RP clips] → NO SHUNT (cannot be Type 3 without RP clips)

TYPE 2 BRANCHING — ask_branching flag:
    Set ask_branching=true when there are MULTIPLE RP at N3 tributaries in a Type 2A, 2B, or 2C case.
    The ligation choice among multiple N3 branches depends on:
        • Calibre of branches (equal or unequal)
        • Distance of each branch to its perforator
        • Whether drainage through the thinner vessel is possible
    If unequal calibre with drainage possible → ligate the larger vessel.
    If unequal calibre, no drainage → ligate the smaller vessel.
    If equal calibre, unequal distance → ligate the branch with longer distance to perforator.

COORDINATE HINTS (secondary — always check fromType/toType first):
    posYRatio ≤ 0.098   = SFJ region (upper thigh)
    0.099–0.353         = Hunterian / mid-thigh
    0.354–0.60          = Knee / popliteal
    > 0.60              = Calf / ankle (SPJ region for posterior clips)

OUTPUT FLAGS:
    needs_elim_test : true when RP N3→N1 + RP N2→N1 present but eliminationTest is absent (B3)
    ask_branching   : true for Type 2A/2B/2C with multiple RP at N3

CONFIDENCE GUIDE:
    Clear single pattern, no ambiguity         → 0.90–0.97
    Pattern present but some noise clips       → 0.80–0.89
    Ambiguous (needs elimination test)         → 0.50–0.65
    No pattern / insufficient clips            → 0.40–0.55
"""


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: SHUNT CLASSIFICATION (No RAG)
# ─────────────────────────────────────────────────────────────────────────────

_CLIP_LABELS: dict[tuple, str] = {
    ("EP", "N1", "N2"): None,
    ("EP", "N2", "N2"): " [PERFORATOR-ENTRY: N2→N2, SFJ=COMPETENT]",
    ("EP", "N2", "N3"): " [GSV-to-TRIBUTARY-ENTRY: N2→N3]",
    ("RP", "N2", "N1"): " [GSV-TRUNK-REFLUX: N2→N1]",
}


def _posY_to_location(posY: float) -> str:
    """Translate a posYRatio value to a named anatomical location."""
    if posY <= 0.09:
        return "groin / SFJ level"
    elif posY <= 0.20:
        return "upper thigh"
    elif posY <= 0.35:
        return "mid-thigh (Hunterian area)"
    elif posY <= 0.55:
        return "knee / popliteal area"
    elif posY <= 0.80:
        return "calf"
    else:
        return "ankle"


def _compute_ligation_hints(shunt_type: str, clips: list[dict]) -> str:
    """
    Pre-compute the anatomical ligation location from clip posYRatios so the LLM
    outputs location-specific ligation steps instead of generic CHIVA notation.
    """
    hints = []

    if "Type 1" in shunt_type and "1+2" not in shunt_type:
        for c in clips:
            if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N2":
                y = c.get("posYRatio") or 0.0
                if y <= 0.098:
                    hints.append(
                        "EP N1→N2 entry is at the SFJ level "
                        "→ LIGATION POINT: Saphenofemoral Junction (SFJ), at the groin"
                    )
                else:
                    loc = _posY_to_location(y)
                    hints.append(
                        f"EP N1→N2 entry is at the {loc} "
                        f"→ LIGATION POINT: Hunterian Perforator at the {loc} (NOT at the SFJ)"
                    )

    elif shunt_type in ("Type 2A", "Type 2B", "Type 2C"):
        ep_clips = sorted(
            [c for c in clips if c.get("flow") == "EP"],
            key=lambda c: c.get("posYRatio") or 0.0,
        )
        if ep_clips:
            top = ep_clips[0]
            y = top.get("posYRatio") or 0.0
            ft, tt = top.get("fromType", "?"), top.get("toType", "?")
            loc = _posY_to_location(y)
            if ft == "N2" and tt == "N2":
                hints.append(
                    f"EP N2→N2 perforator entry is at the {loc} "
                    f"→ PRIMARY LIGATION POINT: perforator entry into the GSV at the {loc}"
                )
            elif ft == "N2" and tt == "N3":
                hints.append(
                    f"EP N2→N3 branch point is at the {loc} "
                    f"→ PRIMARY LIGATION POINT: GSV-to-tributary junction at the {loc}"
                )
        if shunt_type == "Type 2C":
            rp_n2 = sorted(
                [c for c in clips if c.get("flow") == "RP" and c.get("fromType") == "N2" and c.get("toType") == "N1"],
                key=lambda c: c.get("posYRatio") or 0.0,
            )
            for c in rp_n2:
                y = c.get("posYRatio") or 0.0
                loc = _posY_to_location(y)
                hints.append(f"RP N2→N1 GSV reflux segment at the {loc} → also ligate this GSV segment")

    elif shunt_type == "Type 3":
        for c in clips:
            if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3":
                y = c.get("posYRatio") or 0.0
                loc = _posY_to_location(y)
                hints.append(
                    f"EP N2→N3 branch point is at the {loc} "
                    f"→ PRIMARY LIGATION POINT: GSV-to-tributary junction at the {loc}. "
                    f"Follow up 6–12 months; if GSV reflux develops, ligate the SFJ."
                )
                break

    elif shunt_type == "Type 1+2":
        for c in clips:
            if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N2":
                y = c.get("posYRatio") or 0.0
                if y <= 0.098:
                    hints.append("SFJ entry point at the groin → SFJ ligation is part of the plan")
                else:
                    loc = _posY_to_location(y)
                    hints.append(f"Hunterian entry point at the {loc} → Hunterian ligation is part of the plan")
        for c in clips:
            if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3":
                y = c.get("posYRatio") or 0.0
                loc = _posY_to_location(y)
                hints.append(f"EP N2→N3 tributary branch at the {loc} → tributary junction also requires ligation")
                break

    if not hints:
        return ""
    return (
        "\n=== PRE-COMPUTED LIGATION LOCATIONS (use these anatomical names in your ligation_steps) ===\n"
        + "\n".join(f"  • {h}" for h in hints)
        + "\n"
    )


def _clip_label(flow: str, ft: str, tt: str, y: float) -> str:
    if flow == "EP" and ft == "N1" and tt == "N2":
        if y <= 0.098:
            return " [SFJ-ENTRY=INCOMPETENT]"
        return " [Hunterian-ENTRY=INCOMPETENT]" if y <= 0.353 else " [Deep-to-GSV-ENTRY]"
    if flow == "RP" and ft == "N3":
        return f" [TRIBUTARY-REFLUX: N3→{tt}]"
    return _CLIP_LABELS.get((flow, ft, tt), "")


def _summarise_clips(clips: list[dict]) -> str:
    lines = []
    for i, c in enumerate(clips):
        flow = c.get("flow", "?")
        ft   = c.get("fromType", "?")
        tt   = c.get("toType", "?")
        y    = c.get("posYRatio") or 0.0
        elim_raw = c.get("eliminationTest")
        elim = elim_raw.strip() if isinstance(elim_raw, str) else ""
        step = c.get("step", "")
        has_rect = c.get("ep_ligation_rect2") or c.get("ep_ligation_rect")

        loc = _clip_label(flow, ft, tt, y)
        parts = [f"  Clip {i:02d}: {flow} {ft}→{tt}  y={y:.3f}{loc}"]
        if step:
            parts.append(f"step={step}")
        if has_rect:
            parts.append("[ligation-point-marker]")
        if elim:
            parts.append(f'eliminationTest="{elim}"')
        lines.append("  ".join(parts))
    return "\n".join(lines)


def build_shunt_classification_prompt(clips: list[dict], leg_label: str) -> str:
    """Build prompt for shunt classification — NO RAG context."""
    clips_str = _summarise_clips(clips)

    return f"""{CHIVA_RULES}

=== ASSESSMENT: {leg_label} ({len(clips)} clips) ===
{clips_str}

═══════════════════════════════════════════════════════════════
STEP-BY-STEP DECISION GUIDE (Follow in order)
═══════════════════════════════════════════════════════════════

STEP 1: CHECK FOR EP N1→N3 FIRST (Type 4 / Type 5 path)
    Look for any clip with flow=EP, fromType=N1, toType=N3.
    If YES → this is a direct deep-to-tributary shunt. Go to Type 4/5 branch below.
    If NO  → continue to STEP 2.

    ┌─ EP N1→N3 PATH (direct deep-to-tributary escape):
    │
    ├─ Has RP N2→N1 (return through saphenous trunk)?
    │  └─ YES → TYPE 4 (confidence 0.88)
    │           Key: the return loop goes THROUGH the N2 trunk (RP N2→N1)
    │
    └─ Has RP N3→N1 or RP N3→N2 (return stays in tributaries)?
       └─ YES → TYPE 5 (confidence 0.88)
               Key: the return loop stays within N3 tributaries (RP N3→N1 or N3→N2)
               *** EP N1→N3 + RP N3→N1 = TYPE 5, NOT TYPE 4 ***

STEP 2: CHECK FOR EP N1→N2 (SFJ or Hunterian ENTRY)
    Look for: "EP N1→N2" with y≤0.098 (SFJ) or y≤0.353 (Hunterian)
    If YES with SFJ-ENTRY/Hunterian-ENTRY label → SFJ INCOMPETENT
    If NO  → SFJ COMPETENT (go to Case C)
    ✓ Found EP N1→N2? YES/NO

    STEP 3: IF YES to EP N1→N2, CHECK FOR REFLUX PATTERNS
    3a) ANY RP N3→N2 or RP N3→N1? (tributary reflux)
    3b) ANY RP N2→N1? (GSV reflux)
    3c) ANY RP anywhere else?
    3d) ANY EP N2→N3? (extra antegrade to tributary)

STEP 4: MATCH PATTERN TO TYPE

    ┌─ SFJ INCOMPETENT PATH (has EP N1→N2):
    │
    │  *** FIRST CHECK FOR ALL BRANCHES: Count RP clips. ZERO RP → NO SHUNT immediately. ***
    │
    ├─ NO EP N2→N3:
    │  ├─ ZERO RP clips anywhere → NO SHUNT DETECTED (confidence 0.95)
    │  │   *** EP N1→N2 alone with no reflux = entry without shunt. Do NOT classify as Type 1. ***
    │  └─ Has RP N2→N1, no RP at N3 → TYPE 1 (confidence 0.90)
    │
    └─ YES EP N2→N3 EXISTS:
        ├─ ZERO RP clips → NO SHUNT DETECTED (confidence 0.90)
        │   *** Type 3, 1+2, and Undetermined ALL require at least one RP clip. ***
        │   *** EP N1→N2 + EP N2→N3 with ZERO RP = no reflux = no shunt. ***
        ├─ Has RP N3 (at N2 or N1), NO RP N2→N1 → TYPE 3 (confidence 0.88)
        ├─ Has RP N3 AND RP N2→N1:
        │  ├─ eliminationTest absent → UNDETERMINED (confidence 0.55) [needs_elim_test=true]
        │  ├─ eliminationTest="Reflux" → TYPE 1+2 (confidence 0.80)
        │  └─ eliminationTest="No Reflux" → TYPE 3 (confidence 0.75)

    ┌─ SFJ COMPETENT PATH (NO EP N1→N2, NO EP N1→N3):
    │
    ├─ EP N2→N3 EXISTS:
    │  └─ TYPE 2A (confidence 0.85-0.92)
    │     └─ Multiple RP at N3? → [ask_branching=true]
    │
    └─ ONLY EP N2→N2 (perforator entry):
        ├─ Has RP N3, NO RP N2→N1 → TYPE 2B (confidence 0.84)
        │  └─ Multiple RP at N3? → [ask_branching=true]
        ├─ Has RP N3 AND RP N2→N1 → TYPE 2C (confidence 0.82)
        │  └─ Multiple RP at N3? → [ask_branching=true]
        └─ No RP at all → NO SHUNT (confidence 0.95)

STEP 4: ASSIGN CONFIDENCE
    Clear pattern, no ambiguity → 0.90–0.97
    Pattern present, minor noise → 0.80–0.89
    Ambiguous / needs elimination test → 0.50–0.65
    Insufficient clips → 0.40–0.55

═══════════════════════════════════════════════════════════════
CRITICAL REMINDERS:
    • Check EP N1→N3 FIRST — if present, it is Type 4 or Type 5, not Type 1/2/3
    • TYPE 4 vs TYPE 5 — the ONLY difference is the return path:
        Type 4: EP N1→N3 + RP N2→N1  (return via saphenous TRUNK)
        Type 5: EP N1→N3 + RP N3→N1 or RP N3→N2  (return via TRIBUTARIES)
        EP N1→N3 + RP N3→N1 = TYPE 5 — NEVER classify this as Type 4
    • EP N1→N2 is THE KEY decision point for Types 1/2/3 — check after ruling out N1→N3
    • EP N2→N2 means perforator (SFJ COMPETENT), never confuse with N1→N2
    • Type 2A has EP N2→N3; Type 2B/2C have EP N2→N2 (NOT N2→N3)
    • Type 2C differs from Type 1+2: 2C has EP N2→N2, Type 1+2 has EP N1→N2
    • RP only at N3 (not N2→N1) + EP N1→N2 = TYPE 3 (not 1+2)
    • EP N1→N2 + EP N2→N3 with ZERO RP clips = NO SHUNT — never classify as Type 3 without RP
    • NEVER infer, hypothesize, or assume RP clips that are not listed in the assessment above
═══════════════════════════════════════════════════════════════

=== TASK ===
Follow the Step-by-Step Decision Guide above. Classify the venous shunt for: {leg_label}.

STRICT OUTPUT RULES:
- summary: 1 sentence clinical summary. Do NOT mention "left leg" or "right leg" unless {leg_label} is explicitly Left or Right (i.e. not "Unspecified").
- reasoning: describe each decision step in plain clinical language (e.g. "EP N1→N2 present, indicating SFJ incompetence"). Do NOT reference internal clip indices ("Clip 00", "Clip 01", etc.), y-coordinates, or posYRatio values in any reasoning step.
- STRICT NO-INFERENCE RULE: classify ONLY based on flow findings listed in the assessment above. Do NOT write "RP might be present", "could have reflux", or any similar inference. If no RP finding is listed, no RP exists.
- NEVER use the word "clip" or "clips" anywhere in summary or reasoning. Say "finding", "flow finding", "entry point", "reflux finding", or "EP/RP finding" instead.

Output ONLY the JSON below — no other text, no markdown.

{{
    "shunt_type": "<Type 1 / Type 2A / Type 2B / Type 2C / Type 3 / Type 4 / Type 5 / Type 1+2 / No shunt detected / Undetermined>",
    "confidence": <0.0-1.0>,
    "reasoning": ["<decision step 1>", "<decision step 2>", "..."],
    "ask_branching": <true/false>,
    "summary": "<1 sentence clinical summary>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: LIGATION PLANNING (With RAG)
# ─────────────────────────────────────────────────────────────────────────────

LIGATION_QUERIES_OLD = {
    "Type 1": "SFJ incompetent with circular reflux N1->N2->N1. High ligation tie at saphenofemoral junction. Multiple GSV reflux points management strategy.",
    "Type 2A": "Tributary entry from GSV trunk N2->N3 without SFJ involvement. Ligate highest EP at tributary junction. Branching anatomy considerations.",
    "Type 2B": "Perforator-fed shunt via N2->N2 entry into saphenous trunk. Open distal shunt with tributary reflux N3->N1. Selective perforator ligation.",
    "Type 2C": "Perforator-fed shunt via N2->N2 entry with secondary GSV reflux N2->N1. Selective perforator ligation combined with GSV segment treatment.",
    "Type 3": "SFJ incompetent with dual entries: EP N1->N2 and EP N2->N3. Staged approach: tributary ligation first, then follow-up for SFJ. Six to twelve month reassessment.",
    "Type 4": "N1->N3 perforator or pelvic-point shunt with N2 return via N2->N1. Target the N1->N3 escape/perforator entry and the return path through N2.",
    "Type 5": "N1->N3 shunt with looping return through N3 and complex re-entry path. Target the N1->N3 escape entry and all refluxing N3 return segments.",
    "Type 1+2": "Complex dual entry shunt with SFJ incompetence and tributary involvement. RP N2->N1 diameter determines strategy. CHIVA 2 staged vs simultaneous ligation.",
    "No shunt detected": "No significant shunt detected. Standard compression therapy. No surgical intervention required.",
    "Undetermined": "Unclear shunt classification. Elimination test required to determine type. Defer ligation planning until classification confirmed.",
}

LIGATION_QUERIES = {
    "Type 1": "SFJ incompetent with circular reflux N1->N2->N1.",
    "Type 2A": "Tributary entry from GSV trunk N2->N3 without SFJ involvement.",
    "Type 2B": "Perforator-fed shunt via N2->N2 entry into saphenous trunk.",
    "Type 2C": "Perforator-fed shunt via N2->N2 entry with secondary GSV reflux N2->N1.",
    "Type 3": "SFJ incompetent with dual entries: EP N1->N2 and EP N2->N3.",
    "Type 4": "N1->N3 perforator or pelvic-point shunt with N2 return via N2->N1.",
    "Type 5": "N1->N3 shunt with looping return through N3 and complex re-entry path.",
    "Type 1+2": "Complex dual entry shunt with SFJ incompetence and tributary involvement.",
    "No shunt detected": "No significant shunt detected.",
    "Undetermined": "Unclear shunt classification. Elimination test required to determine type. Defer ligation planning until classification confirmed.",
}


def build_ligation_prompt(shunt_type: str, clips: list[dict], rag_context: str, leg_label: str) -> str:
    """Build prompt for ligation planning — WITH RAG context from ligation database."""
    clips_str = _summarise_clips(clips)
    location_hints = _compute_ligation_hints(shunt_type, clips)

    return f"""=== LIGATION PLANNING FOR VENOUS SHUNT CLASSIFICATION ===

You are an expert vascular surgeon trained in CHIVA (hemodynamic conservative surgery) principles.
Your task is to generate a detailed, evidence-based ligation plan based on the shunt type and clinical findings.

=== RETRIEVED KNOWLEDGE BASE (Ligation & Treatment Guidelines) ===
{rag_context}

=== SHUNT TYPE IDENTIFIED ===
Type: {shunt_type}

=== CLINICAL ASSESSMENT: {leg_label} ===
Number of clips: {len(clips)}
{clips_str}
{location_hints}
=== TASK ===
Based on the shunt type "{shunt_type}", the clinical findings above, and the medical knowledge base provided:

1. Generate a detailed ligation plan with specific anatomical steps — the pre-computed ligation locations section above gives you the EXACT anatomical site name(s) to use. Copy those location names verbatim into your ligation_steps (e.g. "Ligate at the Hunterian Perforator at the upper thigh", "Ligate the GSV-to-tributary junction at the mid-thigh", "Ligate the perforator entry into the GSV at the groin level").
2. Identify any additional clinical information needed.
3. Consider complications and contraindications.
4. Provide follow-up and monitoring recommendations.
5. Consider CHIVA principles (hemodynamic, saphenous-vein-sparing when appropriate).

Important formatting rules:
1. ligation_steps must be a JSON array with one clear action per item.
2. The FIRST ligation step MUST name the specific anatomical location exactly as given in the pre-computed ligation locations above. If the pre-computed section says "Hunterian Perforator at the upper thigh", your step must say "Hunterian Perforator at the upper thigh" — not just "Hunterian Perforator" and not "SFJ". Do NOT use y-coordinates, posYRatio values, or clip indices.
3. Every ligation step that names a vessel or junction must include the anatomical level (e.g. "at the groin", "at the upper thigh", "at the mid-thigh", "at the knee level", "in the calf"). Never say just "ligate the perforator" without specifying where.
3. clinical_rationale must explain why that plan fits the shunt anatomy in plain surgical language — no y-coordinates.
4. additional_info_needed must be [] when there is no meaningful extra information to request.
5. chiva_approach must describe the hemodynamic CHIVA reasoning, even if brief.
6. Do NOT mention "left leg" or "right leg" in any field if {leg_label} is "Unspecified".
7. NEVER include raw clip data, y-values, posYRatio, or coordinate numbers in any output field.
8. NEVER use the word "clip" or "clips" in any output field. Use "finding", "flow finding", "entry", "reflux finding", or "EP/RP finding" instead.

Output ONLY the JSON below — no other text, no markdown:

{{
    "shunt_type": "{shunt_type}",
    "ligation_steps": ["<step 1>", "<step 2>", "..."],
    "clinical_rationale": "<detailed surgical reasoning>",
    "additional_info_needed": ["<info 1>", "<info 2>", "..."],
    "complications_contraindications": ["<complication 1>", "<contraindication 1>", "..."],
    "followup_schedule": "<follow-up timeline and monitoring plan>",
    "chiva_approach": "<CHIVA-specific hemodynamic considerations>",
    "confidence": <0.0-1.0>
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPAIR
# ─────────────────────────────────────────────────────────────────────────────

def _repair_and_parse(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.rstrip())

    with suppress(Exception):
        return json.loads(text)

    start = text.find("{")
    if start == -1:
        return None
    raw = text[start:]

    depth_b = depth_sq = 0
    in_str = esc = False
    for ch in raw:
        if esc:        esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"':  in_str = not in_str; continue
        if in_str:     continue
        if ch == "{":  depth_b  += 1
        elif ch == "}": depth_b  -= 1
        elif ch == "[": depth_sq += 1
        elif ch == "]": depth_sq -= 1

    if in_str:       raw += '"'
    raw += "]" * max(0, depth_sq)
    raw += "}" * max(0, depth_b)

    with suppress(Exception):
        return json.loads(raw)

    result: dict = {}
    for k in ("shunt_type", "summary", "clinical_rationale", "chiva_approach"):
        if m := re.search(rf'"{k}"\s*:\s*"([^"]*)"', raw):
            result[k] = m[1]
    if cm := re.search(r'"confidence"\s*:\s*([\d.]+)', raw):
        result["confidence"] = float(cm[1])

    def ex_list(key):
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', raw)
        return re.findall(r'"([^"]+)"', m[1]) if m else []

    result["ligation_steps"] = ex_list("ligation_steps")
    result["additional_info_needed"] = ex_list("additional_info_needed")
    result["complications_contraindications"] = ex_list("complications_contraindications")

    if "ligation_steps" not in result or len(result["ligation_steps"]) == 0:
        return None
    result["_repaired"] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — UNIFIED INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFICATION_ERROR_RESULT: dict = {
    "shunt_type": "Classification failed",
    "confidence": 0.0,
    "reasoning": ["The LLM did not return a parseable classification response. Please retry."],
    "needs_elim_test": False,
    "ask_branching": False,
    "summary": "Classification unavailable.",
    "_llm_error": True,
}

_LIGATION_ERROR_RESULT: dict = {
    "shunt_type": "Unknown",
    "ligation_steps": ["Ligation planning failed — unable to generate recommendations."],
    "clinical_rationale": "LLM response could not be parsed.",
    "additional_info_needed": [],
    "complications_contraindications": [],
    "followup_schedule": "Consult vascular surgery specialist.",
    "chiva_approach": "Unable to determine.",
    "confidence": 0.0,
    "_llm_error": True,
}

_LEG_ORDER = {"Left": 0, "Right": 1}


def _deterministic_no_shunt_check(group: list[dict]) -> dict | None:
    """
    Return a No Shunt result immediately — without calling the LLM — when the
    clip pattern is unambiguously shunt-free.

    Rules (all require zero RP clips):
      • EP N1→N2 present (with or without EP N2→N3): SFJ incompetent entry but no
        reflux anywhere → No Shunt.
      • EP N2→N2 present with no EP N2→N3: perforator entry, SFJ competent, no
        reflux → No Shunt.

    EP N2→N3 alone with zero RP is left to the LLM because it may be an early
    Type 2A developing case.
    """
    rp_clips = [c for c in group if c.get("flow") == "RP"]
    if rp_clips:
        return None  # RP present — let LLM classify

    ep_n1_n2 = any(
        c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N2"
        for c in group
    )
    ep_n2_n2 = any(
        c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N2"
        for c in group
    )
    ep_n2_n3 = any(
        c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"
        for c in group
    )

    if ep_n1_n2:
        # SFJ/Hunterian entry with zero RP → definitively No Shunt
        reason = (
            "EP N1→N2 present (SFJ/Hunterian incompetent entry) but no retrograde flow "
            "detected anywhere — no shunt by CHIVA definition."
        )
        if ep_n2_n3:
            reason = (
                "EP N1→N2 (SFJ entry) and EP N2→N3 (tributary branch) present, "
                "but no retrograde flow found — no shunt. "
                "Type 3 requires at least one RP (reflux) finding."
            )
        logger.info(f"Deterministic No Shunt: {reason}")
        return {
            "shunt_type": "No shunt detected",
            "confidence": 0.95,
            "reasoning": [reason],
            "needs_elim_test": False,
            "ask_branching": False,
            "summary": "No venous shunt detected. Antegrade entry present but no retrograde flow identified.",
            "_llm_usage": {},
        }

    if ep_n2_n2 and not ep_n2_n3:
        reason = (
            "EP N2→N2 present (perforator entry, SFJ competent) but no retrograde flow "
            "detected anywhere — no shunt by CHIVA definition."
        )
        logger.info(f"Deterministic No Shunt: {reason}")
        return {
            "shunt_type": "No shunt detected",
            "confidence": 0.95,
            "reasoning": [reason],
            "needs_elim_test": False,
            "ask_branching": False,
            "summary": "No venous shunt detected. Perforator entry present but no retrograde flow identified.",
            "_llm_usage": {},
        }

    return None  # fall through to LLM


def _call_llm_for_shunt_classification(group: list[dict], leg_label: str, call_llm_fn: Callable) -> dict:
    """Task 1: Classify shunt type — NO RAG."""
    # Deterministic shortcut: zero-RP patterns are unambiguously No Shunt
    deterministic = _deterministic_no_shunt_check(group)
    if deterministic is not None:
        logger.info(f"Shunt classification for {leg_label}: deterministic No Shunt (skipping LLM)")
        return deterministic

    prompt = build_shunt_classification_prompt(group, leg_label)
    logger.info(f"Shunt classification LLM prompt for {leg_label}: {len(prompt)} chars")
    try:
        raw, usage = call_llm_fn(prompt, return_usage=True)
        logger.info(f"Shunt classification LLM response ({leg_label}): {raw[:300]!r}")
        logger.info(f"Shunt classification tokens ({leg_label}): prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}")
        result = _repair_and_parse(raw)
        if result and "shunt_type" in result:
            result['_llm_usage'] = usage
            return result
    except Exception as e:
        logger.error(f"Shunt classification LLM call failed for {leg_label}: {e}")
    logger.error(f"Shunt classification failed for {leg_label}")
    raise RuntimeError(f"Shunt classification failed for {leg_label}")


def _call_llm_for_ligation(shunt_type: str, group: list[dict], rag_context: str, leg_label: str, call_llm_fn: Callable) -> dict:
    """Task 2: Plan ligation — WITH RAG."""
    prompt = build_ligation_prompt(shunt_type, group, rag_context, leg_label)
    logger.info(f"Ligation planning LLM prompt for {leg_label}: {len(prompt)} chars")
    try:
        raw, usage = call_llm_fn(prompt, return_usage=True)
        logger.info(f"Ligation planning LLM response ({leg_label}): {raw[:300]!r}")
        logger.info(f"Ligation planning tokens ({leg_label}): prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}")
        result = _repair_and_parse(raw)
        if result and "ligation_steps" in result:
            result['_llm_usage'] = usage
            return result
    except Exception as e:
        logger.error(f"Ligation planning LLM call failed for {leg_label}: {e}")
    logger.error(f"Ligation planning failed for {leg_label}")
    raise RuntimeError(f"Ligation planning failed for {leg_label}")


def _retrieve_rag_context_for_ligation(shunt_type: str, retrieve_fn: Callable) -> str:
    """Retrieve ligation-specific RAG context from ligation database."""
    try:
        query = LIGATION_QUERIES.get(shunt_type, f"Ligation planning for {shunt_type}")
        if chunks := retrieve_fn(query, k=3):
            return "\n\n---\n\n".join(str(ch)[:600] for ch in chunks)
    except Exception as e:
        logger.warning(f"RAG retrieval failed for ligation planning ({shunt_type}): {e}")
    return "No RAG context available."


def classify_and_plan_ligation_with_llm(
    clip_list: list[dict[str, Any]],
    call_llm_fn: Callable,
    retrieve_ligation_context_fn: Callable | None = None,
) -> dict:
    """
    Unified API: Classify shunts AND generate ligation plans.

    Workflow:
    1. Group clips by leg
    2. Call LLM for SHUNT CLASSIFICATION (no RAG)
    3. Call LLM for LIGATION PLANNING (with ligation RAG)
    4. Return combined result

    Args:
        clip_list: Raw clip data from assessment
        call_llm_fn: Function to call LLM (returns (response, usage_dict))
        retrieve_ligation_context_fn: Function to retrieve from ligation database

    Returns:
        {
            "findings": [
                {
                    "leg": "Left" | "Right",
                    "shunt_type": str,
                    "confidence": float,
                    "reasoning": [...],
                    "needs_elim_test": bool,
                    "ask_branching": bool,
                    "summary": str,
                    "ligation_steps": [...],
                    "clinical_rationale": str,
                    "additional_info_needed": [...],
                    "complications_contraindications": [...],
                    "followup_schedule": str,
                    "chiva_approach": str,
                    "num_clips": int,
                }
            ],
            "shunt_type": str (primary leg),
            "confidence": float (primary leg),
            "summary": str (primary leg),
            ...
        }
    """
    # Group by leg
    groups: dict[str, list[dict]] = {}
    for c in clip_list:
        side = (c.get("legSide") or c.get("leg_side") or "Assessment").strip().capitalize()
        groups.setdefault(side, []).append(c)

    _NO_LIGATION_RESULT = {
        "No shunt detected": {
            "ligation_steps": [],
            "clinical_rationale": "No pathological venous shunt identified. No surgical intervention is required.",
            "additional_info_needed": [],
            "complications_contraindications": [],
            "followup_schedule": "Routine clinical follow-up. Conservative management with compression therapy if symptomatic.",
            "chiva_approach": "No hemodynamic shunt present — CHIVA intervention is not indicated.",
            "confidence": 0.95,
            "_llm_usage": {},
        },
        "Undetermined": {
            "ligation_steps": [],
            "clinical_rationale": "Shunt classification is undetermined — ligation planning cannot proceed until the elimination test result is available.",
            "additional_info_needed": ["Elimination test result required to confirm shunt type before ligation planning."],
            "complications_contraindications": [],
            "followup_schedule": "Perform elimination test. Return for reassessment once shunt type is confirmed.",
            "chiva_approach": "Defer ligation planning until hemodynamic classification is confirmed via elimination test.",
            "confidence": 0.0,
            "_llm_usage": {},
        },
    }

    findings = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for leg_label, group in groups.items():
        # Step 1: Shunt Classification (NO RAG)
        classification = _call_llm_for_shunt_classification(group, leg_label, call_llm_fn)
        classification_usage = classification.pop("_llm_usage", {})
        shunt_type = classification.get("shunt_type", "Unknown")

        # Step 2: Ligation Planning — skip entirely for types that require no ligation
        if shunt_type in _NO_LIGATION_RESULT:
            ligation = dict(_NO_LIGATION_RESULT[shunt_type])
            ligation_usage = ligation.pop("_llm_usage", {})
        else:
            rag_context = (
                _retrieve_rag_context_for_ligation(shunt_type, retrieve_ligation_context_fn)
                if retrieve_ligation_context_fn else "No RAG context available."
            )
            ligation = _call_llm_for_ligation(shunt_type, group, rag_context, leg_label, call_llm_fn)
            ligation_usage = ligation.pop("_llm_usage", {})

        total_prompt_tokens += classification_usage.get("prompt_tokens", 0) + ligation_usage.get("prompt_tokens", 0)
        total_completion_tokens += classification_usage.get("completion_tokens", 0) + ligation_usage.get("completion_tokens", 0)

        # Merge both results
        finding = {
            "leg": leg_label,
            "num_clips": len(group),

            # Classification results
            "shunt_type": classification.get("shunt_type"),
            "assessment": classification.get("shunt_type"),
            "confidence": classification.get("confidence", 0.0),
            "reasoning": classification.get("reasoning", []),
            "needs_elim_test": classification.get("needs_elim_test", False),
            #"ask_diameter": classification.get("ask_diameter", False),
            "ask_branching": classification.get("ask_branching", False),
            "summary": classification.get("summary", ""),

            # Ligation results
            "ligation_steps": ligation.get("ligation_steps", []),
            "point_of_ligation": ligation.get("ligation_steps", [""])[0] if ligation.get("ligation_steps") else "",
            "clinical_rationale": ligation.get("clinical_rationale", ""),
            "additional_info_needed": ligation.get("additional_info_needed", []),
            "complications_contraindications": ligation.get("complications_contraindications", []),
            "followup_schedule": ligation.get("followup_schedule", ""),
            "chiva_approach": ligation.get("chiva_approach", ""),
            "classification_llm_usage": classification_usage,
            "ligation_llm_usage": ligation_usage,
        }
        findings.append(finding)

    findings.sort(key=lambda f: _LEG_ORDER.get(f["leg"], 2))

    if not findings:
        raise RuntimeError("Combined shunt classifier returned no findings")

    primary = findings[0]
    return {
        "findings": findings,
        "shunt_type": primary.get("shunt_type"),
        "confidence": primary.get("confidence", 0.0),
        "reasoning": primary.get("reasoning", []),
        "ligation": primary.get("ligation_steps", []),  # For backward compat with old API
        "point_of_ligation": primary.get("point_of_ligation", primary.get("ligation_steps", [""])[0] if primary.get("ligation_steps") else ""),
        "summary": primary.get("summary", ""),
        "needs_elim_test": primary.get("needs_elim_test", False),
        #"ask_diameter": primary.get("ask_diameter", False),
        "ask_branching": primary.get("ask_branching", False),
        "num_clips": len(clip_list),
        "num_findings": len(findings),
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }
