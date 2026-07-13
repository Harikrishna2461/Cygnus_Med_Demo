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
    N1 = Deep venous system (common femoral vein, femoral vein, popliteal vein, calf deep veins)
    N2 = Named saphenous trunk ONLY — GSV (Great Saphenous Vein) OR SSV (Small Saphenous Vein)
         GSV: groin (SFJ) → medial malleolus, within saphenous fascial compartment ("saphenous eye")
         SSV: lateral malleolus → popliteal fossa (SPJ), posterior leg
    N3 = Tributaries, accessory veins, varicosities in subcutaneous tissue above superficial fascia
         Includes AASV (anterior accessory GSV), reticular veins, and all named tributaries
    EP = Entry Point — where blood escapes FROM the deep system INTO the superficial system (pathological)
         The EP itself is the junction or perforator that has failed; blood flows into the shunt circuit here.
    RP = Re-entry Point — where blood exits FROM the superficial system BACK INTO the deep system.
         The RP is NOT refluxing. The RP perforator carries blood superficial→deep (correct perforator direction).
         What REFLUXES is the N2 or N3 SEGMENT that delivers blood TO the RP, not the RP itself.
         NEVER say "RP reflux" or "RP is refluxing" — say "GSV reflux above the RP" or "reflux in the trunk
         segment between SFJ and RP N2→N1" or "reflux in the tributary segment feeding RP N3→N1".
    SFJ = Saphenofemoral Junction (GSV → common femoral vein)  →  posYRatio ≤ 0.098
    SPJ = Saphenopopliteal Junction (SSV → popliteal vein)     →  posYRatio ≈ 0.40–0.50 (posterior)
    Hunterian Perforator (mid-thigh, N1→N2 or N2→N2)          →  0.098 < posYRatio ≤ 0.353
    AASV = Anterior Accessory Saphenous Vein — N3, not N2; common pitfall on duplex
    NOTE: Pure Type 1 (GSV trunk loop only, zero refluxive tributaries) is rare in clinical practice.
          Most SFJ-incompetent cases with GSV trunk reflux also have refluxive tributaries → Type 1+2.

═══════════════════════════════════════════════════════════
RULE ZERO — CHECK RP COUNT BEFORE ANYTHING ELSE:
    Count every finding where flow=RP.

    IF RP count = 0:
      Check whether EP N2→N3 is present AND whether EP N1→N2 is present.

      ── EP N2→N3 present, NO EP N1→N2, zero RP → TYPE 2A (early/developing shunt).
         The GSV is escaping into a tributary with no reflux yet. This IS a shunt. Proceed to Case C.

      ── EP N2→N3 present, EP N1→N2 ALSO present, zero RP → NO SHUNT DETECTED.
         EP N1→N2 + EP N2→N3 with zero RP = NO SHUNT. This is NOT Type 3.
         Type 3 REQUIRES RP N3 clips. Without any RP, it cannot be Type 3.
         DO NOT classify as Type 3 because both EP N1→N2 and EP N2→N3 are present.

      ── NO EP N2→N3 present, zero RP → NO SHUNT DETECTED. Full stop.
         EP N1→N2 alone = NO SHUNT. EP N2→N2 alone = NO SHUNT.
         *** DO NOT classify as Type 1 because EP N1→N2 is present — Type 1 ALSO needs RP N2→N1. ***

    IF RP count ≥ 1 → proceed to the rules below.
═══════════════════════════════════════════════════════════

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
    *** FIRST: Are there ANY RP clips? ***
    If ZERO RP clips anywhere → NO SHUNT DETECTED.
      Type 1 REQUIRES RP N2→N1. EP N1→N2 alone is NOT Type 1. It is NO SHUNT.
      DO NOT classify as Type 1 if RP N2→N1 is absent.

    If RP N2→N1 present AND no RP at N3 (no RP N3→N2, no RP N3→N1) → TYPE 1
    Ligation: Ligate at SFJ (y≤0.098) or Hunterian (y≤0.353).
            If multiple RP N2→N1: ligate below each except the most distal.

─────────────────────────────────────────────────────────
Case B — EP N1→N2 EXISTS (SFJ or Hunterian) AND EP N2→N3 EXISTS
─────────────────────────────────────────────────────────
    B0: ZERO RP clips anywhere → NO SHUNT DETECTED.
        Type 3 and Type 1+2 BOTH require RP clips. EP N1→N2 + EP N2→N3 alone with zero RP = NO SHUNT.
        DO NOT classify as Type 3 just because EP N1→N2 and EP N2→N3 are present. RP is mandatory.

    B1: RP N3→N2 or RP N3→N1, NO RP N2→N1               → TYPE 3
    B2: RP N3→N2 AND RP N2→N1                             → TYPE 3
    B3: RP N3→N1 AND RP N2→N1, eliminationTest absent    → UNDETERMINED (set needs_elim_test=true)
    B4: RP N3→N1 AND RP N2→N1, eliminationTest="Reflux"  → TYPE 1+2
    B5: RP N3→N1 AND RP N2→N1, eliminationTest="No Reflux" → TYPE 3

    *** CRITICAL — WHICH CLIP TO READ eliminationTest FROM ***
    The elimination test result MUST be read from the EP N2→N3 clip OR the RP N3 clips.
    The EP N1→N2 clip (SFJ entry) may also carry an eliminationTest field — IGNORE IT for
    B3/B4/B5 purposes. It records SFJ reflux status, not the tributary compression result.
    When MULTIPLE clips carry eliminationTest values, use the value on EP N2→N3 or RP N3
    clips — that is the actual test result.
    *** If EP N2→N3 or RP N3 has eliminationTest="No Reflux" → B5 → TYPE 3,
        even if EP N1→N2 separately carries eliminationTest="Reflux". ***

    *** ABSOLUTE RULE — B2: RP N3→N2 + RP N2→N1 = TYPE 3. ALWAYS. ***
    *** RP N3→N2 (tributary→GSV) means the tributary drains BACK INTO the saphenous trunk,
        forming a CLOSED SUPERFICIAL LOOP. This is the defining feature of Type 3. ***
    *** DO NOT let background CHIVA knowledge override B2. Some Type 3 circuits DO have a
        segment of RP N2→N1 (the GSV between SFJ and the escape point carries blood downward),
        but when the tributary drains back to GSV (RP N3→N2), the circuit is closed within
        the superficial system = TYPE 3, not Type 1+2. ***
    *** TYPE 1+2 requires RP N3→N1 (tributary to DEEP) + eliminationTest="Reflux". ***
    *** TYPE 1+2 is IMPOSSIBLE when only RP N3→N2 is present (no RP N3→N1). ***
    *** TYPE 1+2 is IMPOSSIBLE without eliminationTest="Reflux" on an EP N2→N3 or RP N3 clip. ***
    *** If eliminationTest is absent from the clips, you MUST output UNDETERMINED, not Type 1+2. ***

    TYPE 3 Ligation:
        Single RP at N3: Ligate EP at N2→N3. Follow up 6–12 months; if N2 reflux develops, ligate SFJ.
        Multiple RP at N3: Ligate every refluxing tributary at N2 junction (CHIVA 2 step 1). Same follow-up.

    TYPE 1+2 Ligation:
        Procedure: CHIVA 1 (single stage, simultaneous).
        Ligate SFJ/Hunterian (EP N1→N2) AND every refluxing N2→N3 junction in the same operative session.
        If RP N2→N1 segments are present along the GSV trunk: ligate below each except the most distal.
        Do NOT use a staged CHIVA 2 approach for Type 1+2 — the SFJ and tributary escapes must be addressed together.

═══════════════════════════════════════════════════════════
TYPE 3 vs TYPE 1+2 — REASONING GUIDE
(Read this whenever you are considering Type 3 OR Type 1+2 as your answer)
═══════════════════════════════════════════════════════════

Both Type 3 and Type 1+2 share the SAME structural clip pattern:
    EP N1→N2  (SFJ or Hunterian incompetent)
  + EP N2→N3  (GSV feeds a tributary)
  + RP N3     (tributary carries retrograde flow)
  + RP N2→N1  (GSV trunk also refluxes)

They CANNOT be told apart from clips alone. The ONLY reliable differentiator is the
ELIMINATION TEST (compression test) result. Before finalising Type 3 or Type 1+2,
ask yourself: "What does the description say happens to the tributary's reflux when
the GSV or SFJ is compressed?"

─── WHAT THE ELIMINATION TEST TELLS YOU ───────────────────────────────────────

The elimination test has TWO valid methods. Read the clinician's description and
understand WHAT was compressed and WHAT was observed, then map to the correct type.

    ══ METHOD 1: Compress the TRIBUTARY — observe the SAPHENOUS VEIN ══

    If compressing the tributary causes the saphenous vein reflux to STOP:
        The entire recirculating volume was draining through the tributary alone.
        The GSV has no independent drainage path to the deep system.
        → TYPE 3. Clip value: eliminationTest="No Reflux"

    If compressing the tributary leaves saphenous vein reflux UNCHANGED/CONTINUING:
        The GSV has its own independent perforator (RP N2→N1) draining it regardless.
        Blocking the tributary does not interrupt the GSV recirculation.
        → TYPE 1+2. Clip value: eliminationTest="Reflux"

    ══ METHOD 2: Compress the EP/SFJ — observe the TRIBUTARY ══

    If compressing the SFJ/GSV causes the tributary reflux to DISAPPEAR:
        The tributary had no independent blood source — it was entirely fed by the shunt.
        Cut off the GSV inflow and the tributary has nothing left to reflux.
        → TYPE 1+2. Clip value: eliminationTest="Reflux"

    If compressing the SFJ/GSV leaves tributary reflux UNCHANGED/CONTINUING:
        The tributary has its own independent incompetent perforator feeding it.
        Even with GSV inflow cut off, the perforator keeps it refluxing.
        → TYPE 3. Clip value: eliminationTest="No Reflux"

    SCENARIO C — No compression test described at all:
        → UNDETERMINED (set needs_elim_test=true)

─── DECISION RULE ───────────────────────────────────────────────────────────────

    If you reached a conclusion of TYPE 3 or TYPE 1+2 — STOP and verify:
      1. Is there an elimination test result in the clips?
         NO  → change output to UNDETERMINED, set needs_elim_test=true
         YES → read the value:
               "Reflux"    → TYPE 1+2
               "No Reflux" → TYPE 3

    This check is MANDATORY. Do not skip it.

═══════════════════════════════════════════════════════════

─────────────────────────────────────────────────────────
Case C — NO EP N1→N2 ANYWHERE (SFJ COMPETENT)
─────────────────────────────────────────────────────────
    C-Sub-check: Check EP N2→N2 FIRST (defines 2B/2C), then EP N2→N3 (defines 2A).

    ── TYPE 2B ── EP N2→N2 present, NO EP N1→N2, RP at N3, NO RP N2→N1
        Entry is via perforator (fromType=N2, toType=N2 — NOT N1→N2).
        IMPORTANT: EP N2→N2 at ANY posYRatio (even 0.05, SFJ-Knee step) = perforator, NOT SFJ.
        Key signal: EP N2→N2 clip + RP N3→N2 or N3→N1 + NO EP N1→N2 + NO RP N2→N1.
        EP N2→N3 may also be present alongside EP N2→N2 — 2B classification still applies.
        If multiple RP at N3 → set ask_branching=true.
        Ligation: Ligate the highest EP N2→N2 (perforator entry point).

    ── TYPE 2C ── EP N2→N2 present, NO EP N1→N2, RP at N3, RP N2→N1 ALSO present
        Perforator entry (EP N2→N2) with secondary GSV reflux (RP N2→N1). SFJ still competent.
        IMPORTANT: 2C has EP N2→N2 (perforator), while Type 1+2 has EP N1→N2 (SFJ entry).
        If NO EP N1→N2 but RP N2→N1 exists with EP N2→N2 → TYPE 2C, not Type 1+2.
        Key signal: EP N2→N2 + RP N3 + RP N2→N1 + NO EP N1→N2.
        Ligation: Ligate perforator entry (highest EP N2→N2) AND all RP N2→N1 sites along GSV.

    ── TYPE 2A ── EP N2→N3 present, NO EP N2→N2, NO EP N1→N2
        The defining feature is EP N2→N3 (GSV feeding a tributary) without perforator entry.
        RP may or may not be present in early/developing cases.
        Typical pattern: EP N2→N3 + RP N3→N2 or N3→N1. No RP N2→N1. No EP N2→N2.
        Key signal: EP N2→N3 clip exists + NO EP N2→N2 + NO EP N1→N2.
        If multiple RP at N3 → set ask_branching=true (need calibre/distance/drainage info).
        Ligation: Ligate highest EP at N2→N3 junction.
                    If multiple branching at N3: ligate based on calibre, distance to perforator, drainage.

    Case C — NO SHUNT:
        If NO RP clips of any kind → NO SHUNT DETECTED.

─────────────────────────────────────────────────────────
Case D — No RP in any finding:
    If EP N2→N3 is present (without EP N1→N2) → TYPE 2A. Early shunt, no reflux yet developed.
    Otherwise (EP N1→N2 alone, EP N2→N2 alone, EP N1→N2 + EP N2→N3, or no EP at all) → NO SHUNT DETECTED.
─────────────────────────────────────────────────────────

QUICK DECISION TABLE:
    Has EP N1→N2? YES + no EP N2→N3 + RP N2→N1           → TYPE 1
    Has EP N1→N2? YES + EP N2→N3 + RP N3 only             → TYPE 3
    Has EP N1→N2? YES + EP N2→N3 + RP N3 + RP N2→N1 + eliminationTest absent → UNDETERMINED
    Has EP N1→N2? YES + EP N2→N3 + RP N3 + RP N2→N1 + elim="Reflux"          → TYPE 1+2
    Has EP N1→N2? YES + EP N2→N3 + RP N3 + RP N2→N1 + elim="No Reflux"       → TYPE 3
    Has EP N1→N2? YES + EP N2→N3 + ZERO RP clips          → NO SHUNT (not Type 3 — Type 3 requires RP)
    Has EP N1→N2? YES + no EP N2→N3 + ZERO RP clips       → NO SHUNT
    No EP N1→N2  + EP N2→N2 + RP N3 + NO RP N2→N1         → TYPE 2B  ← EP N2→N2 takes priority
    No EP N1→N2  + EP N2→N2 + RP N3 + RP N2→N1            → TYPE 2C
    No EP N1→N2  + EP N2→N2 + NO RP                        → NO SHUNT
    No EP N1→N2  + EP N2→N3 (no EP N2→N2)                  → TYPE 2A
    EP N1→N3 + RP N2→N1                                    → TYPE 4
    EP N1→N3 + RP N3→N2 + RP N2→N1                        → TYPE 4 (pelvic/perforator via tributary — RP N3→N2 is intermediate step)
    EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1              → TYPE 5 (biphasic loop: perforator→N3→N2→N3→N1)
    EP N1→N3 + RP N3→N1 (NO RP N3→N2, NO EP N2→N3)        → TYPE 6 (direct perforator circuit: N1→N3→N1)
    No EP N1→N2  + EP N2→N3 + NO RP                        → TYPE 2A (early, no reflux yet)
    No RP at all + no EP N2→N3                             → NO SHUNT

─────────────────────────────────────────────────────────
CRITICAL CONFUSION RISK — TYPE 1 vs TYPE 4:
─────────────────────────────────────────────────────────
Both Type 1 and Type 4 present with RP N2→N1 (GSV trunk reflux).
They are EASILY CONFUSED because the GSV behaves identically in both.
The single distinguishing clip is the ENTRY finding:

  TYPE 1:  EP N1→N2 is PRESENT  — SFJ or Hunterian INCOMPETENT.
           Blood enters the saphenous TRUNK directly (N2) from the deep
           system (N1). The circuit is N1 → N2 → N1 (no N3 involved in entry).
           ↳ No EP N1→N3 will be present in a pure Type 1.

  TYPE 4:  EP N1→N3 is PRESENT  — SFJ is COMPETENT. No SFJ/Hunterian failure.
           Blood enters a TRIBUTARY (N3) from the deep/pelvic system (N1),
           then drains into the GSV (RP N3→N2 may appear as intermediate step).
           The GSV trunk carries it back to deep (RP N2→N1).
           Circuit: N1/P → N3 → (N2) → N1.
           ↳ No EP N1→N2 will be present in a pure Type 4.

RULE: EP N1→N2 present → Type 4/5/6 are EXCLUDED. Must be Type 1, 3, or 1+2.
      EP N1→N3 present (no EP N1→N2) → Type 1 is EXCLUDED. Must be Type 4, 5, or 6.
─────────────────────────────────────────────────────────

CONCRETE EXAMPLES:
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
        Type 4 (perforating subtype):  [EP N1→N3 y=0.60, RP N2→N1 y=0.40]
            → EP N1→N3 (perforator enters N3 directly) + RP N2→N1 (GSV trunk return) → TYPE 4
        Type 4 (pelvic/tributary subtype):  [EP N1→N3 y=0.05, RP N3→N2 y=0.08, RP N2→N1 y=0.25]
            → EP N1→N3 (pelvic/pudendal vein enters N3 at groin) + RP N3→N2 (N3 drains into GSV)
              + RP N2→N1 (GSV returns to deep) → TYPE 4 (RP N3→N2 is intermediate, not return limb)
        Type 5:  [EP N1→N3 y=0.25, RP N3→N2 y=0.30, EP N2→N3 y=0.35, RP N3→N1 y=0.50]
            → EP N1→N3 (perforator enters N3) + RP N3→N2 (N3 drains to GSV) + EP N2→N3
              (GSV drains to 2nd tributary) + RP N3→N1 (2nd tributary re-enters deep)
              → biphasic circuit through N2, NO RP N2→N1 → TYPE 5
        Type 6:  [EP N1→N3 y=0.60, RP N3→N1 y=0.75]
            → EP N1→N3 (perforator enters N3) + RP N3→N1 (N3 re-enters deep directly)
              → pure perforator circuit, NO N2/GSV involvement → TYPE 6
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
    ("EP", "N1", "N2"): None,   # label applied dynamically by _clip_label() based on posY
    ("EP", "N1", "N3"): " [PERFORATOR/PELVIC-TO-TRIBUTARY: N1→N3, SFJ=COMPETENT — TYPE 4/5 ENTRY]",
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


def _loc_label(c: dict) -> str:
    """Return the most specific location string for a clip: step label + anatomical name."""
    y = c.get("posYRatio") or 0.0
    step = (c.get("step") or "").strip()
    anat = _posY_to_location(y)
    return f"{step} — {anat}" if step else anat


def _pick_ligation_clip(candidates: list[dict]) -> dict | None:
    """
    Return the best clip to describe the ligation point:
    prefer clips marked with ep_ligation_rect2/ep_ligation_rect (surgeon-marked in CMED),
    then fall back to the clip with the lowest posYRatio (most proximal).
    """
    marked = [c for c in candidates if c.get("ep_ligation_rect2") or c.get("ep_ligation_rect")]
    if marked:
        return min(marked, key=lambda c: c.get("posYRatio") or 0.0)
    if candidates:
        return min(candidates, key=lambda c: c.get("posYRatio") or 0.0)
    return None


def _compute_primary_step(shunt_type: str, clips: list[dict]) -> str:
    """
    Returns the first ligation step in the same terse format as the CMED clip data.
    Uses the clip's step label directly — no computed anatomical guesses.
    e.g. "Ligate the EP at N2->N3 at SFJ-Knee"
    """
    def _loc(c: dict) -> str:
        return _posY_to_location(c.get("posYRatio") or 0.0)

    if "Type 1" in shunt_type and "1+2" not in shunt_type:
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N2"]
        )
        if ep:
            entry = "SFJ" if (ep.get("posYRatio") or 0.0) <= 0.098 else "Hunterian perforator"
            loc = _loc(ep)
            return f"Ligation at the {entry}" + (f" at {loc}" if loc else "")

    elif shunt_type == "Type 2A":
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"]
        )
        if ep:
            loc = _loc(ep)
            return f"Ligate highest EP at N2->N3" + (f" at {loc}" if loc else "")

    elif shunt_type == "Type 2B":
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N2"]
        )
        if ep:
            loc = _loc(ep)
            return f"Ligate highest EP at N2->N2 (perforator)" + (f" at {loc}" if loc else "")

    elif shunt_type == "Type 2C":
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") in ("N2", "N3")]
        )
        if ep:
            ft, tt = ep.get("fromType"), ep.get("toType")
            loc = _loc(ep)
            return f"Ligate highest EP at {ft}->{tt}" + (f" at {loc}" if loc else "")

    elif shunt_type == "Type 3":
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"]
        )
        if ep:
            loc = _loc(ep)
            return f"Ligate the EP at N2->N3" + (f" at {loc}" if loc else "")

    elif shunt_type == "Type 1+2":
        ep_sfj = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N2"]
        )
        ep_trib = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"]
        )
        parts = []
        if ep_sfj:
            entry = "SFJ" if (ep_sfj.get("posYRatio") or 0.0) <= 0.098 else "Hunterian perforator"
            loc = _loc(ep_sfj)
            parts.append(f"Ligation at {entry}" + (f" at {loc}" if loc else ""))
        if ep_trib:
            loc = _loc(ep_trib)
            parts.append(f"ligate EP at N2->N3" + (f" at {loc}" if loc else ""))
        if parts:
            return " + ".join(parts)

    elif shunt_type == "Type 4":
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N3"]
        )
        if ep:
            loc = _loc(ep)
            return f"Ligate N1→N3 perforator/pelvic entry" + (f" at {loc}" if loc else "")

    elif shunt_type == "Type 5":
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N3"]
        )
        if ep:
            loc = _loc(ep)
            return f"Ligate N1→N3 perforator entry (Stage 1)" + (f" at {loc}" if loc else "")

    elif shunt_type == "Type 6":
        ep = _pick_ligation_clip(
            [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N3"]
        )
        if ep:
            loc = _loc(ep)
            return f"Ligate N1→N3 perforator entry" + (f" at {loc}" if loc else "")

    return ""


def _compute_ligation_hints(shunt_type: str, clips: list[dict]) -> str:
    """
    Pre-compute exact ligation locations from clip data.
    Prioritises surgeon-marked ligation points (ep_ligation_rect2/ep_ligation_rect),
    includes the clip's step label for precision.
    """
    hints = []

    if "Type 1" in shunt_type and "1+2" not in shunt_type:
        ep_n1n2 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N2"]
        c = _pick_ligation_clip(ep_n1n2)
        if c:
            y = c.get("posYRatio") or 0.0
            loc = _loc_label(c)
            if y <= 0.098:
                hints.append(f"LIGATION POINT 1 — Saphenofemoral Junction (SFJ) at the groin [{loc}]")
            else:
                hints.append(f"LIGATION POINT 1 — Hunterian Perforator at [{loc}] (NOT the SFJ)")
        rp_n2n1 = sorted(
            [c for c in clips if c.get("flow") == "RP" and c.get("fromType") == "N2" and c.get("toType") == "N1"],
            key=lambda c: c.get("posYRatio") or 0.0,
        )
        for i, rc in enumerate(rp_n2n1[:-1], start=2):  # all except most distal
            loc = _loc_label(rc)
            hints.append(f"LIGATION POINT {i} — GSV trunk RP segment at [{loc}]: ligate below this reflux point")

    elif shunt_type in ("Type 2A", "Type 2B", "Type 2C"):
        ep_n2n2 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N2"]
        ep_n2n3 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"]
        rp_n3 = [c for c in clips if c.get("flow") == "RP" and c.get("fromType") == "N3"]
        primary = _pick_ligation_clip(ep_n2n2 or ep_n2n3)
        if primary:
            loc = _loc_label(primary)
            ft, tt = primary.get("fromType"), primary.get("toType")
            calibre_note = f" [calibre: {primary['calibre']}]" if primary.get("calibre") else ""
            notes_note = f" [{primary['notes']}]" if primary.get("notes") else ""
            if ft == "N2" and tt == "N2":
                hints.append(f"LIGATION POINT 1 — Perforator entry into GSV at [{loc}]{calibre_note}{notes_note}: divide flush at this level")
            elif ft == "N2" and tt == "N3":
                hints.append(f"LIGATION POINT 1 — GSV-to-tributary junction at [{loc}]{calibre_note}{notes_note}: flush tie at this level")
        # Additional N3 branches with calibre/notes info for ligation sequence guidance
        extra_ep_n2n3 = [c for c in ep_n2n3 if c is not primary]
        for i, bc in enumerate(extra_ep_n2n3, start=2):
            loc = _loc_label(bc)
            calibre_note = f" [calibre: {bc['calibre']}]" if bc.get("calibre") else ""
            notes_note = f" [{bc['notes']}]" if bc.get("notes") else ""
            hints.append(f"BRANCH {i} — Additional GSV-to-tributary junction at [{loc}]{calibre_note}{notes_note}: ligate after primary branch if indicated")
        # Surface calibre/notes on RP N3 clips for context
        for rc in rp_n3:
            if rc.get("calibre") or rc.get("notes"):
                loc = _loc_label(rc)
                calibre_note = f" [calibre: {rc['calibre']}]" if rc.get("calibre") else ""
                notes_note = f" [{rc['notes']}]" if rc.get("notes") else ""
                hints.append(f"TRIBUTARY INFO — RP {rc.get('fromType')}→{rc.get('toType')} at [{loc}]{calibre_note}{notes_note}")
        if shunt_type == "Type 2C":
            rp_n2n1 = sorted(
                [c for c in clips if c.get("flow") == "RP" and c.get("fromType") == "N2" and c.get("toType") == "N1"],
                key=lambda c: c.get("posYRatio") or 0.0,
            )
            for i, rc in enumerate(rp_n2n1[:-1], start=2):
                loc = _loc_label(rc)
                hints.append(f"LIGATION POINT {i} — GSV reflux segment at [{loc}]: ligate below this RP N2→N1")

    elif shunt_type == "Type 3":
        ep_n2n3 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"]
        c = _pick_ligation_clip(ep_n2n3)
        if c:
            loc = _loc_label(c)
            hints.append(
                f"LIGATION POINT 1 (Stage 1) — GSV-to-tributary junction at [{loc}]: "
                f"flush tie at this exact level. SFJ NOT touched in Stage 1."
            )
        hints.append(
            "LIGATION POINT 2 (Stage 2, only if N2 reflux confirmed at follow-up) — "
            "Saphenofemoral Junction (SFJ) at the groin: high tie / flush SFJ ligation."
        )

    elif shunt_type == "Type 1+2":
        ep_n1n2 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N2"]
        c1 = _pick_ligation_clip(ep_n1n2)
        if c1:
            y = c1.get("posYRatio") or 0.0
            loc = _loc_label(c1)
            if y <= 0.098:
                hints.append(f"LIGATION POINT 1 — SFJ at the groin [{loc}]: high tie / flush SFJ ligation")
            else:
                hints.append(f"LIGATION POINT 1 — Hunterian Perforator at [{loc}]: flush ligation")
        ep_n2n3 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"]
        c2 = _pick_ligation_clip(ep_n2n3)
        if c2:
            loc = _loc_label(c2)
            hints.append(f"LIGATION POINT 2 — GSV-to-tributary junction at [{loc}]: flush tie at this level")

    elif shunt_type == "Type 4":
        ep_n1n3 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N3"]
        c = _pick_ligation_clip(ep_n1n3)
        if c:
            loc = _loc_label(c)
            hints.append(f"LIGATION POINT 1 — N1→N3 perforator/pelvic entry at [{loc}]: divide flush at this level")
        # RP N3→N2 = tributary-to-GSV junction (intermediate step in pelvic and perforating subtypes)
        rp_n3n2 = sorted(
            [c for c in clips if c.get("flow") == "RP" and c.get("fromType") == "N3" and c.get("toType") == "N2"],
            key=lambda c: c.get("posYRatio") or 0.0,
        )
        for i, rc in enumerate(rp_n3n2, start=2):
            loc = _loc_label(rc)
            hints.append(
                f"LIGATION POINT {i} — Tributary-to-GSV junction (RP N3→N2) at [{loc}]: "
                f"flush tie at this tributary-GSV connection (intermediate escape point)"
            )
        # RP N2→N1 = GSV trunk return segments — ligate all except most distal
        rp_n2n1 = sorted(
            [c for c in clips if c.get("flow") == "RP" and c.get("fromType") == "N2" and c.get("toType") == "N1"],
            key=lambda c: c.get("posYRatio") or 0.0,
        )
        base_i = 2 + len(rp_n3n2)
        for i, rc in enumerate(rp_n2n1[:-1], start=base_i):
            loc = _loc_label(rc)
            hints.append(f"LIGATION POINT {i} — GSV trunk RP N2→N1 segment at [{loc}]: ligate below this reflux point")

    elif shunt_type == "Type 5":
        ep_n1n3 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N3"]
        c = _pick_ligation_clip(ep_n1n3)
        if c:
            loc = _loc_label(c)
            hints.append(
                f"LIGATION POINT 1 (Stage 1) — N1→N3 perforator entry at [{loc}]: "
                f"divide flush at deep-to-superficial entry. Stage 1 only."
            )
        # EP N2→N3 = GSV-to-tributary junction (the exit from GSV into the 2nd tributary)
        ep_n2n3 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N2" and c.get("toType") == "N3"]
        c2 = _pick_ligation_clip(ep_n2n3)
        if c2:
            loc = _loc_label(c2)
            hints.append(
                f"LIGATION POINT 2 (Stage 2 if EP N2→N3 persists at follow-up) — "
                f"GSV-to-tributary junction at [{loc}]: flush tie to eliminate residual Type 2 shunt."
            )

    elif shunt_type == "Type 6":
        ep_n1n3 = [c for c in clips if c.get("flow") == "EP" and c.get("fromType") == "N1" and c.get("toType") == "N3"]
        c = _pick_ligation_clip(ep_n1n3)
        if c:
            loc = _loc_label(c)
            hints.append(
                f"LIGATION POINT 1 — N1→N3 perforator entry at [{loc}]: "
                f"divide flush at deep-to-superficial entry. RP N3→N1 re-entry will resolve spontaneously."
            )

    if not hints:
        return ""
    return (
        "\n=== LIGATION POINTS FROM THIS PATIENT'S CLIPS ===\n"
        "(Use these specific locations when writing ligation_steps — do not substitute generic body region names)\n"
        + "\n".join(f"  {h}" for h in hints)
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
        if c.get("calibre"):
            parts.append(f'calibre="{c["calibre"]}"')
        if c.get("source"):
            parts.append(f'source="{c["source"]}"')
        if c.get("notes"):
            parts.append(f'notes="{c["notes"]}"')
        lines.append("  ".join(parts))
    return "\n".join(lines)


def build_shunt_classification_prompt(clips: list[dict], leg_label: str) -> str:
    """Build prompt for shunt classification — NO RAG context."""
    clips_str = _summarise_clips(clips)

    rp_count = sum(1 for c in clips if c.get("flow") == "RP")

    return f"""[Findings summary: {len(clips)} total finding(s), {rp_count} RP finding(s).]
{CHIVA_RULES}

=== ASSESSMENT: {leg_label} ({len(clips)} clips) ===
{clips_str}

═══════════════════════════════════════════════════════════════
HEMODYNAMIC REASONING — Reason like a clinician, not a flowchart
═══════════════════════════════════════════════════════════════

You are a CHIVA-trained vascular clinician. Do not follow a mechanical checklist.
Reason about what these findings mean physiologically, then arrive at a classification
by understanding the hemodynamic circuit — not by matching patterns to a lookup table.

WHAT EACH FINDING TELLS YOU ABOUT THIS PATIENT'S VENOUS PHYSIOLOGY:

  EP N1→N2  — Blood is escaping from the deep venous system into the GSV (at SFJ/Hunterian)
               or into the SSV (at SPJ). The terminal or preterminal valve at that junction
               has failed. This is the hallmark of SFJ, SPJ, or Hunterian incompetence.
               Without accompanying reflux (RP), this entry alone does not constitute an
               active shunt — the valve is open but no closed recirculation circuit exists yet.
               The deep-to-saphenous pressure gradient drives blood into the superficial system
               during the hydrostatic phase when the calf pump is at rest.

  EP N2→N2  — A perforator is injecting blood laterally into the mid-GSV trunk.
               The SFJ valve is intact and competent. This is fundamentally different
               from EP N1→N2: it represents mid-segment perforator pathology, not
               junction failure. EP N2→N2 and EP N1→N2 can look similar in notation
               but describe completely different anatomical events.

  EP N1→N3  — Blood escapes directly from the deep system into a tributary, entirely
               bypassing the GSV trunk. A pelvic point or deep perforator connects
               directly to superficial tributaries. When present, this finding defines
               the shunt architecture as Type 4 or 5 before any other consideration —
               the GSV trunk is not the primary conduit.

  EP N2→N3  — The GSV trunk pressure is high enough to push blood forward (antegrade) into
               a tributary. This is overflow, not reflux. Alone with no RP, it indicates an
               early developing shunt (Type 2A) — the hydrostatic load has exceeded the
               tributary outflow threshold but sustained retrograde flow has not yet established.
               The oscillatory flow this creates within the tributary drives progressive vein wall
               dilation over time. Combined with SFJ entry (EP N1→N2), it means the shunt
               circuit extends into the tributary network.

  RP N2→N1  — The GSV or SSV trunk is carrying blood backward (downward, away from the heart)
               under hydrostatic load. Blood that entered at the SFJ or Hunterian perforator now
               flows retrograde down the trunk. At the bottom of this refluxing segment, blood exits
               the superficial system via an incompetent re-entry perforator back into the deep vein.
               The RP N2→N1 marks that exit point — the perforator itself is NOT refluxing, it carries
               blood superficial→deep. The TRUNK SEGMENT above the RP is what refluxes.
               Sustained retrograde trunk flow >500 ms is pathological. Without this finding,
               Type 1 cannot be present — SFJ entry alone with no trunk reflux is haemodynamically
               incomplete and does not constitute an active shunt.

  RP N3→N2  — A tributary is draining retrograde back into the GSV trunk. The shunt
               loop closes through the saphenous trunk — overflow from the GSV reaches
               a tributary and then refluxes back into the GSV rather than draining
               toward the foot.

  RP N3→N1  — Tributary reflux drains directly back to the deep system without using
               the GSV trunk as a return highway. The reflux loop closes entirely within
               the tributary network.

  Zero RP   — No retrograde flow exists anywhere in this leg. Without reflux there is
               no closed shunt circuit — blood that enters the superficial system has
               somewhere to go physiologically. The sole exception: EP N2→N3 alone
               (early Type 2A) where antegrade overflow is the presenting finding
               before reflux develops. All other entries without RP = no active shunt.

THE HEMODYNAMIC CIRCUIT EACH SHUNT TYPE REPRESENTS:

  Type 1   — Full SFJ trunk loop: deep blood enters GSV at the SFJ (EP N1→N2)
              and the GSV trunk itself refluxes backward (RP N2→N1). A closed
              saphenous trunk circuit using the GSV as both the inflow and the
              reflux highway. No tributary escape needed.

  Type 2A  — Early antegrade tributary escape: GSV overflows forward into a tributary
              (EP N2→N3) without established retrograde flow yet. The shunt is present
              but immature — reflux has not developed. No SFJ failure required.

  Type 2B  — Perforator-fed tributary circuit: a perforator pushes blood into the
              GSV mid-segment (EP N2→N2), and the loop closes via tributary reflux
              (RP N3) back to the deep system. The GSV trunk does not itself reflux
              backward — the circuit bypasses SFJ entirely.

  Type 2C  — Perforator-fed with secondary GSV trunk involvement: same perforator
              entry (EP N2→N2) but the shunt circuit has expanded to also drive GSV
              trunk reflux (RP N2→N1) alongside tributary reflux (RP N3). Greater
              hemodynamic load than Type 2B.

  Type 3   — SFJ-entry with tributary extension, trunk-sparing return: SFJ is
              incompetent (EP N1→N2) and the overflow extends into tributaries
              (EP N2→N3), with the loop closing via tributary reflux (RP N3 back
              to deep). Critically, the GSV trunk does NOT reflux backward — the
              shunt circuit routes through tributaries rather than the trunk.

  Type 4   — Direct deep/pelvic-to-tributary escape, GSV trunk return: blood bypasses
              the SFJ entirely WITHOUT SFJ incompetence. Two anatomical subtypes share
              the same hemodynamic circuit (entry → N3 → N2 → N1 re-entry):
              • Perforating subtype: an incompetent perforator delivers deep blood
                directly into a tributary (EP N1→N3). That tributary may connect into
                the GSV (RP N3→N2), and the GSV returns blood to deep (RP N2→N1).
                Circuit: N1 → N3 → (N2) → N1.
              • Pelvic subtype: a pelvic, pudendal, gluteal, or labial vein delivers
                blood into a groin tributary (EP N1→N3), which drains into the GSV
                (RP N3→N2), and the GSV trunk refluxes back to deep (RP N2→N1).
                Circuit: P → N3 → N2 → N1.
              In both subtypes, RP N2→N1 is the return limb (same anatomical site as
              Type 1) but the SFJ is COMPETENT — the entry is NOT at the SFJ.
              RP N3→N2 may appear as an intermediate step (tributary draining into GSV)
              and does not reclassify to Type 5. Presence of RP N2→N1 always confirms
              Type 4 over Type 5 when EP N1→N3 is also present.

  Type 5   — Biphasic perforator escape with N2 as intermediate, tributary return:
              blood enters a tributary via a perforator (EP N1→N3), that tributary
              drains into the GSV (RP N3→N2), the GSV then delivers blood to a SECOND
              tributary (EP N2→N3), and that second tributary re-enters the deep system
              (RP N3→N1). Circuit: N1 → N3 → N2 → N3 → N1. The GSV is an intermediate
              conduit — not the return limb. NO RP N2→N1.
              Differentiator from Type 4: both have RP N3→N2 as an intermediate, but
              Type 4 then has RP N2→N1 (GSV to deep), while Type 5 has EP N2→N3 (GSV
              to a second tributary) followed by RP N3→N1.
              Differentiator from Type 6: Type 5 routes through N2 (GSV present as
              intermediate); Type 6 returns to deep WITHOUT any N2 involvement.

  Type 6   — Pure perforator-to-perforator circuit, no saphenous trunk involvement:
              blood enters a tributary directly via an incompetent perforator (EP N1→N3)
              and re-enters the deep system via a second incompetent perforator from the
              same or an adjacent tributary (RP N3→N1). Circuit: N1 → N3 → N1.
              The GSV trunk (N2) is NOT involved — no RP N3→N2, no EP N2→N3, no RP N2→N1.
              Occurs most commonly in varicose recurrences after stripping (neo-perforators)
              and in venous malformations. Hemodynamic load is contained within tributaries.
              Differentiator from Type 5: no N2 intermediate step.
              Differentiator from Type 4: no RP N2→N1 and no N2 involvement at all.

  Type 1+2 — Two concurrent entry points: SFJ incompetence (EP N1→N2) AND tributary
              escape (EP N2→N3), with BOTH trunk reflux (RP N2→N1) AND tributary
              reflux (RP N3). Requires the elimination test to confirm, because this
              pattern overlaps with Type 3 until that test is performed.

  Undetermined — The hemodynamic picture raises Type 1+2 vs Type 3 but the
                 elimination test result is absent from the findings, making it
                 impossible to resolve the circuit definitively.

PHYSIOLOGICAL TRUTHS THAT MUST HOLD IN YOUR REASONING:
  • A closed shunt circuit requires retrograde flow. Zero RP = no active circuit
    in all cases except Type 2A (antegrade overflow without established reflux).
  • Type 1 physically requires GSV trunk reflux (RP N2→N1). A failed SFJ valve
    (EP N1→N2) alone means the valve is open, not that a circuit is flowing.
  • EP N2→N2 and EP N1→N2 are anatomically distinct events. Confusing them
    means confusing perforator pathology with SFJ failure — entirely different
    clinical significance and treatment implications.
  • EP N1→N3, when present, defines a Type 4/5/6 architecture. The return path
    determines which of the three it is.
  • TYPE 4 vs TYPE 5 vs TYPE 6 — all share EP N1→N3; differentiate by return path:
    - Type 4: RP N2→N1 present (blood returns via GSV trunk to deep). N2 is RETURN LIMB.
    - Type 5: RP N3→N2 + EP N2→N3 + RP N3→N1 (N2 is INTERMEDIATE only; blood exits N2
              via EP N2→N3 into a second tributary, then returns to deep via RP N3→N1).
              NO RP N2→N1. The biphasic N2 loop (N3→N2→N3) is the signature of Type 5.
    - Type 6: RP N3→N1 only — N2 is ABSENT entirely. Pure perforator-to-perforator.
              No RP N3→N2, no EP N2→N3, no RP N2→N1.
  • In Type 4, RP N3→N2 may appear as an intermediate step (tributary draining
    into GSV before GSV returns to deep). This does NOT reclassify to Type 5.
    If RP N2→N1 is present alongside EP N1→N3, it is always Type 4.
  • Type 4 SFJ rule: The SFJ is COMPETENT in Type 4. EP N1→N2 (SFJ/Hunterian
    incompetence) must NOT be present for a pure Type 4 diagnosis. If EP N1→N2
    is also present, classify using the Type 1, 3, or 1+2 rules instead.
  • TYPE 1 vs TYPE 4 — CRITICAL DISTINCTION (both have RP N2→N1):
    Both types produce GSV trunk reflux (RP N2→N1), making them similar on duplex.
    The ONLY definitive distinguishing finding is the ENTRY clip:
      - Type 1 has EP N1→N2 (SFJ or Hunterian INCOMPETENT). No EP N1→N3 in pure Type 1.
        Circuit is a direct trunk loop: N1 → N2 → N1.
      - Type 4 has EP N1→N3 (perforator or pelvic vein enters TRIBUTARY). No EP N1→N2.
        SFJ is COMPETENT. Circuit routes through a tributary before reaching the trunk:
        N1/P → N3 → (N2) → N1.
    If EP N1→N2 is in the findings → Type 4 is EXCLUDED. Choose Type 1, 3, or 1+2.
    If EP N1→N3 is in the findings and EP N1→N2 is ABSENT → Type 1 is EXCLUDED. Choose Type 4 or 5.
    Never assign both EP N1→N2 and EP N1→N3 to a pure Type 4 or pure Type 1 case.
  • Type 3 vs Type 1+2: both share EP N1→N2 + EP N2→N3 + RP N3. The difference
    is whether RP N2→N1 is also present AND what the elimination test shows.
    Without both, the case is Undetermined.
  • Never infer, assume, or hypothesize findings not explicitly listed above.

CONFIDENCE CALIBRATION:
  Complete, unambiguous hemodynamic picture → 0.90–0.97
  Pattern present but minor uncertainty → 0.80–0.89
  Ambiguous or elimination test needed → 0.50–0.65
  Insufficient findings to form a full picture → 0.40–0.55
═══════════════════════════════════════════════════════════════

=== TASK ===
Reason about this patient's venous hemodynamics as a clinician and classify the shunt for: {leg_label}.

STRICT OUTPUT RULES:
- chain_of_thought: Reason through the hemodynamics in plain clinical language. Write EACH distinct reasoning point as its OWN SEPARATE LINE — do NOT write a continuous paragraph. Use this structure, one line per point:
    Line 1: What the entry finding(s) tell you physiologically (where blood enters, what valve/junction has failed)
    Line 2: What the reflux finding(s) tell you (where blood travels backward and what circuit that forms)
    Line 3: The overall hemodynamic circuit these findings describe together
    Line 4: Why this matches the chosen CHIVA type
    Line 5: Why it does NOT match the one or two types it most closely resembles
  Separate each line with a literal newline character (\n) inside the JSON string. Never write all reasoning as one long paragraph.
- summary: 1 sentence clinical summary. Do NOT mention "left leg" or "right leg" unless {leg_label} is explicitly Left or Right (i.e. not "Unspecified").
- reasoning: describe each decision step in plain clinical language (e.g. "EP N1→N2 present, indicating SFJ incompetence"). Do NOT reference internal clip indices ("Clip 00", "Clip 01", etc.), y-coordinates, or posYRatio values in any reasoning step.
- STRICT NO-INFERENCE RULE: classify ONLY based on flow findings listed in the assessment above. Do NOT write "RP might be present", "could have reflux", or any similar inference. If no RP finding is listed, no RP exists.
- NEVER use the word "clip" or "clips" anywhere in summary, reasoning, or chain_of_thought. Say "finding", "flow finding", "entry point", "reflux finding", or "EP/RP finding" instead.
- needs_elim_test: Set to true when EP N2→N3 is present AND both RP N3→N1 and RP N2→N1 are present AND no eliminationTest result has been provided on any EP N2→N3 or RP N3 finding (Case B3). When needs_elim_test is true, shunt_type MUST be "Undetermined" — do NOT output Type 1+2 or Type 3. This is MANDATORY: reaching B3 conditions without an eliminationTest result means you CANNOT classify further.

Output ONLY the JSON below — no other text, no markdown.

{{
    "chain_of_thought": "<what entry finding(s) mean physiologically>\\n<what reflux finding(s) mean physiologically>\\n<the overall hemodynamic circuit these form>\\n<why this matches the chosen type>\\n<why it does not match the closest alternative type(s)>",
    "shunt_type": "<Type 1 / Type 2A / Type 2B / Type 2C / Type 3 / Type 4 / Type 5 / Type 6 / Type 1+2 / No shunt detected / Undetermined>",
    "confidence": <0.0-1.0>,
    "reasoning": ["<decision step 1>", "<decision step 2>", "..."],
    "needs_elim_test": <true/false>,
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
    "Type 1+2": "Complex dual entry shunt SFJ incompetent N1->N2 plus tributary escape N2->N3. CHIVA 1 single stage simultaneous high tie SFJ and flush tie every N2->N3 junction.",
    "No shunt detected": "No significant shunt detected. Standard compression therapy. No surgical intervention required.",
    "Undetermined": "Unclear shunt classification. Elimination test required to determine type. Defer ligation planning until classification confirmed.",
}

LIGATION_QUERIES = {
    "Type 1": (
        "Type 1 CHIVA shunt SFJ incompetent closed trunk circuit N1→N2→N1. "
        "High tie flush ligation at saphenofemoral junction groin. "
        "Preserve saphenous vein drainage. Ligate below each GSV reflux segment except most distal."
    ),
    "Type 2A": (
        "Type 2A CHIVA shunt GSV antegrade overflow into tributary N2→N3 without SFJ failure. "
        "Selective tributary flush tie at GSV junction. GSV trunk preserved. "
        "Multiple branching tributaries calibre distance perforator drainage."
    ),
    "Type 2B": (
        "Type 2B CHIVA shunt incompetent Hunterian perforator N2→N2 entry into saphenous trunk. "
        "SFJ competent. Tributary reflux N3→N1 open deviating shunt. "
        "Sub-fascial perforator ligation flush at entry point. GSV trunk preserved."
    ),
    "Type 2C": (
        "Type 2C CHIVA shunt perforator N2→N2 entry secondary GSV trunk reflux N2→N1. "
        "Combined perforator ligation plus selective GSV segment ligation. "
        "SFJ competent. Greater haemodynamic load than 2B."
    ),
    "Type 3": (
        "Type 3 CHIVA shunt SFJ incompetent GSV escape into tributary N2→N3 tributary reflux N3 closes loop. "
        "GSV trunk does not reflux. Staged CHIVA 2 approach: flush tie tributary at N2→N3 junction first. "
        "SFJ ligation deferred to Stage 2 only if N2 reflux develops at 6-12 month follow-up duplex."
    ),
    "Type 4": (
        "Type 4 CHIVA shunt two subtypes pelvic and perforating. SFJ competent in both. "
        "Perforating subtype: incompetent perforator N1→N3 deep blood directly into tributary. "
        "Pelvic subtype: pelvic pudendal gluteal labial vein enters groin tributary N3 bypassing SFJ. "
        "Both subtypes: tributary may drain into GSV via RP N3→N2 junction, then GSV returns to deep via RP N2→N1. "
        "Circuit N1/P → N3 → N2 → N1. SFJ NOT ligated. "
        "Ligation targets: N1→N3 entry flush divide, N3→N2 tributary-GSV junction flush tie if present, "
        "GSV trunk segments below each RP N2→N1 except most distal. "
        "Pelvic origin: groin incision for pudendal or labial vein ligation; consider coil embolisation if residual reflux."
    ),
    "Type 5": (
        "Type 5 CHIVA shunt biphasic perforator circuit N1→N3→N2→N3→N1. "
        "Blood enters tributary via perforator N1→N3, tributary drains into GSV RP N3→N2, "
        "GSV delivers to second tributary EP N2→N3, second tributary re-enters deep RP N3→N1. "
        "GSV is intermediate conduit not return limb. No RP N2→N1. "
        "Staged CHIVA 2: Stage 1 ligate N1→N3 perforator entry. "
        "Stage 2 at 4-6 weeks: if EP N2→N3 GSV-to-tributary junction persists, ligate it to prevent residual Type 2 shunt. "
        "SEPS approach or subfascial mini-open. Compression 10 weeks."
    ),
    "Type 6": (
        "Type 6 CHIVA shunt pure perforator-to-perforator circuit N1→N3→N1. "
        "Blood enters tributary via incompetent perforator N1→N3, re-enters deep system directly via RP N3→N1. "
        "No GSV involvement whatsoever — no RP N3→N2, no EP N2→N3, no RP N2→N1. "
        "Occurs in varicose recurrences post-stripping and venous malformations. "
        "CHIVA 1 single-stage: ligate N1→N3 perforator entry only. RP N3→N1 re-entry resolves spontaneously. "
        "Subfascial perforator ligation or SEPS. Compression 8 weeks."
    ),
    "Type 1+2": (
        "Type 1+2 CHIVA shunt dual entry SFJ incompetent N1→N2 plus tributary escape N2→N3. "
        "Both GSV trunk reflux N2→N1 and tributary reflux N3 present. "
        "Elimination test confirmed GSV as sole feeder of tributary. "
        "CHIVA 1 single stage simultaneous: high tie at SFJ plus flush tie at every N2→N3 junction in same session. "
        "Ligate GSV below each RP N2→N1 segment except the most distal."
    ),
    "No shunt detected": "No pathological venous shunt identified. No surgical intervention required. Conservative management compression therapy.",
    "Undetermined": (
        "Undetermined shunt classification Type 3 versus Type 1+2 cannot be distinguished without elimination test. "
        "Defer ligation planning. Perform SFJ or GSV compression test on duplex to assess tributary reflux independence."
    ),
}


def build_ligation_prompt(shunt_type: str, clips: list[dict], rag_context: str, leg_label: str) -> str:
    """Build prompt for ligation planning — WITH RAG context from ligation database."""
    clips_str = _summarise_clips(clips)
    location_hints = _compute_ligation_hints(shunt_type, clips)
    primary_step = _compute_primary_step(shunt_type, clips)
    primary_step_block = (
        f"\n=== STEP 1 OF LIGATION PLAN (use as ligation_steps[0]) ===\n  {primary_step}\n"
        if primary_step else ""
    )

    return f"""You are an expert CHIVA vascular surgeon writing a detailed operative plan for a colleague. Be specific about technique names, procedure stages, and follow-up.

=== LIGATION KNOWLEDGE BASE ===
{rag_context}

=== SHUNT TYPE: {shunt_type} | {leg_label} ===
{clips_str}
{location_hints}{primary_step_block}

=== CHIVA PROCEDURE FRAMEWORK ===

CHIVA 1 (single-stage simultaneous):
  All escape/entry points ligated in one operative session.
  Indicated when: Type 1, Type 2A/2B/2C with single tributary, Type 1+2, Type 6.
  Technique: under local or tumescent anaesthesia, expose target junction, divide and double-ligate with flush tie.

CHIVA 2 (two-stage sequential):
  Stage 1: Ligate primary escape point only. Wait 6–12 months for venous haemodynamic adaptation.
  Stage 2: Reassess duplex. Ligate remaining refluxing points if still present.
  Indicated when: Type 3, Type 5, complex multi-tributary cases requiring staged decompression.

=== LIGATION TECHNIQUE VOCABULARY ===

Flush ligation: divide the vessel at its junction with no residual stump — mandatory at SFJ/SPJ to prevent recurrence from reflux into a residual stump.
High tie (Trendelenburg ligation): ligate the GSV at the SFJ level with all inguinal tributaries divided flush, leaving no cribriform stump. Standard for SFJ incompetence.
SPJ ligation (posterior approach): patient prone or lateral; incision in popliteal fossa; expose SSV at its junction with popliteal vein; flush-divide with no stump. Sural nerve at risk — identify and preserve.
Selective tributary ligation: isolate and double-ligate only the refluxing tributary at its junction with the GSV/SSV; saphenous trunk preserved.
Perforator ligation: expose and divide the incompetent perforator (EP N2→N2) through a targeted sub-fascial or open mini-incision. For Hunterian perforators: medial thigh approach through a 2 cm incision.
Flush tie at N2→N3: divide and double-ligate the EP N2→N3 branch at the point it exits the GSV trunk.
CHIVA rationale: preserve the saphenous trunk as a draining conduit — a draining GSV/SSV reduces recurrence rates and preserves the vein for future cardiac or peripheral bypass use.

=== PER-TYPE PLAN REQUIREMENTS ===

TYPE 1 (GSV/SFJ system):
  Procedure: CHIVA 1 (single stage).
  Technique: High tie / flush ligation at the SFJ (groin) or Hunterian perforator (mid-thigh).
  Steps: (1) Groin incision (SFJ) or medial thigh incision (Hunterian). (2) Expose the GSV at SFJ or Hunterian level. (3) At SFJ: divide all inguinal tributaries (superficial epigastric, superficial circumflex iliac, pudendal) flush — no cribriform stump. (4) Flush ligation of the EP N1→N2 entry point. (5) If RP N2→N1 segments present along GSV trunk: ligate GSV below each refluxing segment except the most distal (preserve distal drainage outflow).
  Follow-up: Duplex at 6 weeks (check for short-term diastolic retrograde flow — this is normal post-SFJ ligation and not recurrence), then 6 months.
  Complications: SFJ recurrence from retained cribriform stump, lymphocele (groin dissection), saphenous nerve injury (Hunterian approach), haematoma.

TYPE 1 (SSV/SPJ system):
  Procedure: CHIVA 1 (single stage).
  Technique: SPJ flush ligation via posterior popliteal fossa approach.
  Steps: (1) Patient prone or lateral decubitus. (2) Popliteal fossa incision over SPJ. (3) Identify and preserve the sural nerve (runs adjacent to SSV). (4) Expose SSV at its junction with popliteal vein. (5) Flush ligation of SPJ — no residual stump. (6) If RP N2→N1 segments in SSV trunk: ligate below each except most distal.
  Follow-up: Duplex at 6 weeks and 6 months.
  Complications: Sural nerve injury (lateral foot numbness), SPJ stump recurrence, popliteal vein injury, deep vein thrombosis.

TYPE 2A:
  Procedure: CHIVA 1 (single stage).
  Technique: Selective tributary ligation / flush tie at N2→N3 junction.
  Steps: (1) Identify highest EP N2→N3 junction under duplex guidance. (2) Targeted incision over junction. (3) Flush tie at N2→N3 — divide branch flush with GSV trunk, double-ligate. (4) GSV trunk NOT ligated or stripped. (5) If multiple branches: ligate each at GSV junction; for branching anatomy, choose branch with larger calibre, longer perforator distance, or independent drainage.
  Follow-up: Duplex at 6 weeks and 6–12 months to check for late N2 reflux development.
  Complications: Incomplete ligation if branching missed, haematoma, skin numbness at incision.

TYPE 2B:
  Procedure: CHIVA 1 (single stage).
  Technique: Perforator ligation (EP N2→N2 entry point).
  Steps: (1) Identify highest EP N2→N2 perforator entry on duplex. (2) Sub-fascial or mini-open incision at perforator level. (3) Expose and divide perforator flush — double-ligate. (4) Do NOT ligate SFJ — it is competent. (5) GSV trunk preserved; aim is to remove the perforator-driven inflow only.
  Follow-up: Duplex at 6 weeks and 6 months; if GSV fails to normalise, reassess for secondary ligation.
  Complications: Perforator injury, deep venous thrombosis risk (sub-fascial approach), wound infection.

TYPE 2C:
  Procedure: CHIVA 1 (single stage) — combined perforator + GSV segment ligation.
  Technique: Perforator ligation + selective GSV trunk ligation at each RP N2→N1 site.
  Steps: (1) Ligate highest EP N2→N2 perforator (as per 2B). (2) For each RP N2→N1 segment: ligate the GSV below the refluxing segment except the most distal. (3) SFJ NOT ligated. (4) GSV competent segments preserved.
  Follow-up: Duplex at 6 weeks and 6–12 months.
  Complications: Multiple incision sites, saphenous nerve risk, incomplete decompression if GSV trunk segments missed.

TYPE 3:
  Procedure: CHIVA 2 (staged).
  Stage 1 technique: Selective tributary ligation (flush tie at N2→N3). Do NOT ligate SFJ at this stage.
  Stage 2 technique: If N2 (GSV trunk) reflux develops after follow-up → high tie / flush ligation at SFJ.
  Steps (Stage 1): (1) Expose and flush-ligate each refluxing tributary at its N2→N3 junction. (2) Avoid SFJ. (3) Document tributary positions for follow-up duplex.
  Steps (Stage 2 — only if N2 reflux confirmed): (1) Groin incision. (2) High tie with flush SFJ ligation. (3) Divide all SFJ tributaries.
  Follow-up: Duplex at 6 weeks (Stage 1 outcome), then 6–12 months (Stage 2 trigger).
  Complications: Stage 1 — tributary recurrence. Stage 2 — lymphocele, groin wound, nerve injury.

TYPE 1+2:
  Procedure: CHIVA 1 (single stage, simultaneous). Do NOT use a staged CHIVA 2 approach for Type 1+2.
  Rationale: Both the SFJ entry (EP N1→N2) and the tributary escape (EP N2→N3) are driven by the same GSV pressure head. Eliminating only one in isolation leaves the circuit partially open. CHIVA 1 simultaneously addresses all escape points, collapsing the shunt in a single session.
  Technique: Simultaneous high-tie SFJ ligation + flush tie at every refluxing N2→N3 junction.
  Steps: (1) Groin incision — high tie with flush SFJ ligation; divide all inguinal tributaries flush — no cribriform stump. (2) Expose each refluxing N2→N3 tributary junction; flush tie at each junction — double-ligate. (3) For any GSV trunk RP N2→N1 segments: ligate the GSV below each refluxing segment except the most distal (preserve distal drainage outflow). (4) All steps performed in the same operative session under local or tumescent anaesthesia.
  Follow-up: Duplex at 6 weeks and 6 months to confirm shunt resolution.
  Complications: Lymphocele (groin dissection), groin wound infection, saphenous nerve injury, SFJ recurrence from retained cribriform stump, incomplete ligation if a refluxing N2→N3 junction is missed at pre-op mapping.

TYPE 4:
  Procedure: CHIVA 1 (single stage). Two anatomical subtypes — identify which applies from the clinical history and duplex findings.
  NOTE: SFJ is COMPETENT in both Type 4 subtypes. Do NOT ligate the SFJ.

  TYPE 4 — Perforating subtype (N1 → N3 → [N2] → N1):
  Source: Incompetent perforator (Hunterian, calf, or posterior tibial perforator) delivers deep blood directly into a tributary.
  Technique: Sub-fascial perforator ligation at N1→N3 entry + flush tie at RP N3→N2 tributary-GSV junction if present + selective GSV ligation at RP N2→N1 sites.
  Steps: (1) Mark the N1→N3 perforator on pre-op duplex (standing, with Valsalva). (2) Sub-fascial or mini-open incision over the perforator; expose and divide flush at the deep-to-superficial entry point. (3) If RP N3→N2 is present (tributary draining into GSV): expose and flush-tie at the tributary-to-GSV junction. (4) For each RP N2→N1 reflux segment along the GSV: ligate the GSV below that segment; preserve the most distal RP N2→N1 for distal venous drainage. (5) Do NOT ligate the SFJ — it is competent.
  Complications: Perforator not identifiable without pre-op duplex marking; deep vein thrombosis risk from sub-fascial dissection; incomplete decompression if RP N3→N2 tributary-GSV junction is missed; GSV trunk injury.

  TYPE 4 — Pelvic subtype (P → N3 → N2 → N1  or  P → N2 → N1):
  Source: Pelvic vein, pudendal vein, labial vein, or gluteal vein delivers blood into a groin tributary (N3) that connects to the GSV (N2), or directly into the GSV at groin level — bypassing the SFJ.
  Technique: Ligation of pelvic/pudendal vein at groin tributary-GSV entry + selective GSV ligation at RP N2→N1 sites.
  Steps: (1) Confirm pelvic origin on duplex in standing position with Valsalva: flow reversal from pelvis into a groin tributary or directly into the GSV proximal segment. (2) Groin incision over the entry tributary at the inguinal level. (3) Identify and flush-ligate the pelvic/pudendal/labial vein as it connects to the groin tributary (EP N1→N3 entry) — this is the primary ligation target. (4) If RP N3→N2 is present: flush tie at the tributary-to-GSV junction. (5) Ligate GSV below each RP N2→N1 segment except the most distal. (6) SFJ NOT ligated — SFJ is competent; ligating it would be an error. (7) If reflux persists at 6-month follow-up duplex: consider coil embolisation or laparoscopic ligation of the pelvic vein source (ovarian, internal iliac, or pudendal vein).
  Complications: Pelvic vein not visible on standard supine duplex — must scan standing; multiple pelvic tributaries may be present requiring bilateral assessment; residual reflux if pelvic source not fully ablated; pudendal nerve proximity in groin dissection; lymphocele; second-look procedure (laparoscopy or embolisation) required in refractory cases.

  Follow-up (both subtypes): Duplex at 6 weeks post-op. Full reassessment at 6 months — if pelvic origin confirmed and reflux persists, refer for pelvic venous imaging (MR venography or catheter venography) and consider ovarian/iliac vein coil embolisation.

TYPE 5:
  Procedure: CHIVA 2 (staged). Circuit: N1→N3→N2→N3→N1. The GSV acts as an intermediate conduit between two tributary segments.
  Technique: Stage 1 — ligate N1→N3 perforator entry. Stage 2 — ligate EP N2→N3 GSV-to-tributary junction if it persists.
  Stage 1 Steps: (1) Mark the N1→N3 perforator entry on pre-op duplex (standing, Valsalva). (2) Sub-fascial or endoscopic (SEPS) approach; divide flush at deep-to-superficial entry. (3) Do NOT ligate the GSV or SFJ at this stage. (4) Post-op compression 10 weeks at 20–30 mmHg.
  Stage 2 (triggered if EP N2→N3 persists at 4–6 week duplex): (1) Identify the GSV-to-tributary junction (EP N2→N3). (2) Flush tie at this junction to eliminate the residual tributary escape and prevent a secondary Type 2 shunt developing. (3) Do NOT ligate the SFJ.
  Rationale: Cutting only the N1→N3 perforator removes the driving pressure from the circuit. If the GSV-to-tributary escape persists after the perforator is ligated, the shunt has converted to a Type 2 pattern and requires the EP N2→N3 ligation as Stage 2.
  Follow-up: Duplex at 4–6 weeks (Stage 2 trigger assessment), then 6 months.
  Complications: Residual Type 2 shunt if EP N2→N3 junction not ligated at Stage 2; sural or superficial peroneal nerve injury (SEPS approach); DVT risk post-subfascial dissection; wound infection.

TYPE 6:
  Procedure: CHIVA 1 (single stage). Circuit: N1→N3→N1. Pure perforator-to-perforator circuit — the GSV trunk (N2) is not involved.
  Technique: Ligation of N1→N3 incompetent perforator entry point only. The RP N3→N1 re-entry perforator resolves spontaneously once inflow is eliminated.
  Steps: (1) Mark N1→N3 entry perforator on pre-op duplex. (2) Sub-fascial mini-open incision or SEPS at the perforator entry level. (3) Divide flush at the deep-to-superficial entry point. (4) Do NOT ligate the GSV, SFJ, or tributaries — the circuit is entirely contained within the perforator-tributary loop. (5) Post-op compression 8 weeks at 20–30 mmHg.
  Rationale: Eliminating the inflow perforator (N1→N3) collapses the entire circuit. The RP N3→N1 re-entry is passive and will normalise without direct ligation.
  Follow-up: Duplex at 6 weeks and 3 months to confirm RP N3→N1 has resolved. If it persists, consider additional perforator ligation.
  Complications: Incomplete ligation if multiple entry perforators not mapped; neo-angiogenesis in recurrent post-stripping cases; sural or peroneal nerve proximity; DVT risk; residual RP N3→N1 if secondary perforators missed.

NO SHUNT DETECTED:
  No pathological shunt identified. No surgical intervention required.
  ligation_steps: [] (empty — no ligation indicated)
  clinical_rationale: Venous system haemodynamically normal. No entry point, reflux, or re-entry circuit present.
  chiva_approach: None — no shunt, no procedure.
  followup_schedule: Routine clinical review if symptoms persist. Repeat duplex in 12 months if clinically indicated.
  complications_contraindications: Not applicable — no intervention planned.
  confidence: 0.95

UNDETERMINED:
  Shunt type cannot be classified without the elimination (compression) test result.
  Type 3 and Type 1+2 share identical clip patterns — only the elimination test distinguishes them.
  ligation_steps: ["Perform elimination test before ligation planning can proceed — compress the EP at the SFJ or Hunterian level and observe whether tributary reflux is abolished (Type 1+2) or persists (Type 3)."]
  clinical_rationale: Cannot safely plan ligation without knowing whether the tributary is a dependent loop (Type 1+2) or independently fed (Type 3) — the procedures are different.
  chiva_approach: Deferred pending elimination test result.
  followup_schedule: Resubmit with elimination test result to proceed with classification and ligation planning.
  complications_contraindications: Not applicable — no procedure planned until classification is confirmed.
  additional_info_needed: ["Elimination test result required: compress GSV/SFJ and report whether tributary reflux is abolished or persists."]
  confidence: 0.0

=== OUTPUT REQUIREMENTS ===
- ligation_steps: REQUIRED. Short, direct surgical statements — one action per step.
  ligation_steps[0] MUST be exactly the string from "STEP 1 OF LIGATION PLAN" above.
  Steps 2 onwards: add procedural detail — technique, further ligation points, follow-up trigger for staged plans.
- chiva_approach: REQUIRED. State "CHIVA 1" or "CHIVA 2". For CHIVA 2: describe what happens in Stage 1 and Stage 2 and the trigger for Stage 2.
- followup_schedule: REQUIRED. Always specify timing (e.g. "Duplex at 6 weeks post-op, then 6–12 months to assess for N2 reflux development.").
- complications_contraindications: REQUIRED. At least 2 type-specific items. Do not list generic surgical risks.
- clinical_rationale: 1-2 sentences explaining WHY this plan suits this specific anatomy.
- additional_info_needed: leave [] unless something genuinely prevents completing the plan.
- Do NOT mention "left leg" / "right leg" if {leg_label} is "Unspecified".
- NEVER include y-values, posYRatio, or coordinate numbers.
- NEVER use the word "clip" — say "finding", "entry point", "reflux segment", etc.
- NEVER say "RP reflux" or "RP is refluxing" — the RP is a Re-entry Point where blood exits the superficial
  system back to deep. It is NOT refluxing. The REFLUX is in the GSV/tributary SEGMENT that delivers blood
  TO the RP. Say "GSV reflux above RP N2→N1" or "reflux in the trunk segment between SFJ and RP".

Output ONLY valid JSON — no markdown, no extra text:

{{
    "shunt_type": "{shunt_type}",
    "ligation_steps": ["<step 1 — include technique name>", "<step 2>", "<step 3>", "<step 4 if needed>"],
    "clinical_rationale": "<1-2 sentences — why this plan suits this anatomy>",
    "additional_info_needed": [],
    "complications_contraindications": ["<complication 1>", "<complication 2>"],
    "followup_schedule": "<timing and what to assess>",
    "chiva_approach": "<CHIVA 1 or CHIVA 2 — describe stages and decision triggers>",
    "confidence": <0.7-1.0>
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
    "chain_of_thought": "",
    "reasoning": ["The LLM did not return a parseable classification response. Please retry."],
    "needs_elim_test": False,
    "ask_branching": False,
    "summary": "Classification unavailable.",
    "_llm_error": True,
}

_LIGATION_ERROR_RESULT: dict = {
    "shunt_type": "Unknown",
    "ligation_steps": ["Unable to generate ligation plan — please retry"],
    "clinical_rationale": "LLM response could not be parsed.",
    "additional_info_needed": [],
    "complications_contraindications": [],
    "followup_schedule": "",
    "chiva_approach": "",
    "confidence": 0.0,
    "_llm_error": True,
}

_LEG_ORDER = {"Left": 0, "Right": 1}


def _call_llm_for_shunt_classification(group: list[dict], leg_label: str, call_llm_fn: Callable) -> dict:
    """Task 1: Classify shunt type — NO RAG."""
    prompt = build_shunt_classification_prompt(group, leg_label)
    logger.info(f"Shunt classification LLM prompt for {leg_label}: {len(prompt)} chars")
    try:
        raw, usage = call_llm_fn(prompt, max_tokens=2048, temperature=0, return_usage=True)
        logger.info(f"Shunt classification LLM response ({leg_label}): {raw[:300]!r}")
        logger.info(f"Shunt classification tokens ({leg_label}): prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}")
        result = _repair_and_parse(raw)
        if result and "shunt_type" in result:
            result['_llm_usage'] = usage
            return result
        logger.error(f"Shunt classification returned unparseable response for {leg_label}: {raw[:200]!r}")
        raise RuntimeError(f"The model returned an unreadable response for {leg_label}. Please retry.")
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Shunt classification LLM call failed for {leg_label}: {e}")
        raise RuntimeError(str(e)) from e




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
    # Group by leg — if no clips provided, run with an empty "Unspecified" group
    groups: dict[str, list[dict]] = {}
    if not clip_list:
        groups["Unspecified"] = []
    else:
        for c in clip_list:
            side = (c.get("legSide") or c.get("leg_side") or "Assessment").strip().capitalize()
            groups.setdefault(side, []).append(c)

    findings = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for leg_label, group in groups.items():
        # Step 1: Shunt Classification (NO RAG)
        classification = _call_llm_for_shunt_classification(group, leg_label, call_llm_fn)
        classification_usage = classification.pop("_llm_usage", {})
        shunt_type = classification.get("shunt_type", "Unknown")

        # Step 2: Ligation Planning — LLM handles all types including No shunt / Undetermined
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
            "chain_of_thought": classification.get("chain_of_thought", ""),
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
        "chain_of_thought": primary.get("chain_of_thought", ""),
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