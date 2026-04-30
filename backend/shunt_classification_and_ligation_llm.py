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
    No EP N1→N2  + EP N2→N3                                → TYPE 2A
    No EP N1→N2  + EP N2→N2 + RP N3 + NO RP N2→N1         → TYPE 2B
    No EP N1→N2  + EP N2→N2 + RP N3 + RP N2→N1            → TYPE 2C
    No EP N1→N2  + EP N2→N2 + NO RP                        → NO SHUNT
    EP N1→N3 + RP N2→N1                                    → TYPE 4
    EP N1→N3 + RP N3→N2 or RP N3→N1                         → TYPE 5
    No RP at all                                            → NO SHUNT

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
        elim = (c.get("eliminationTest") or "").strip()
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

STEP 1: CHECK FOR EP N1→N2 (SFJ or Hunterian ENTRY)
    Look for: "EP N1→N2" with y≤0.098 (SFJ) or y≤0.353 (Hunterian)
    If YES with SFJ-ENTRY/Hunterian-ENTRY label → SFJ INCOMPETENT
    If NO  → SFJ COMPETENT (go to Case C)
    ✓ Found EP N1→N2? YES/NO

    STEP 2: IF YES to EP N1→N2, CHECK FOR REFLUX PATTERNS
    2a) ANY RP N3→N2 or RP N3→N1? (tributary reflux)
    2b) ANY RP N2→N1? (GSV reflux)
    2c) ANY RP anywhere else?
    2d) ANY EP N2→N3? (extra antegrade to tributary)

STEP 3: MATCH PATTERN TO TYPE

    ┌─ SFJ INCOMPETENT PATH (has EP N1→N2):
    │
    ├─ NO EP N2→N3:
    │  └─ Has RP N2→N1, no RP at N3 → TYPE 1 (confidence 0.90)
    │
    └─ YES EP N2→N3 EXISTS:
        ├─ Has RP N3 (at N2 or N1), NO RP N2→N1 → TYPE 3 (confidence 0.88)
        ├─ Has RP N3 AND RP N2→N1:
        │  ├─ eliminationTest absent → UNDETERMINED (confidence 0.55) [needs_elim_test=true]
        │  ├─ eliminationTest="Reflux" → TYPE 1+2 (confidence 0.80) 
        │  └─ eliminationTest="No Reflux" → TYPE 3 (confidence 0.75)

    ┌─ SFJ COMPETENT PATH (NO EP N1→N2):
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
    • EP N1→N2 is THE KEY decision point — check this FIRST
    • EP N2→N2 means perforator (SFJ COMPETENT), never confuse with N1→N2
    • Type 2A has EP N2→N3; Type 2B/2C have EP N2→N2 (NOT N2→N3)
    • Type 2C differs from Type 1+2: 2C has EP N2→N2, Type 1+2 has EP N1→N2
    • Type 4/5 are N1→N3 path shunts and should be classified explicitly when present
    • RP only at N3 (not N2→N1) + EP N1→N2 = TYPE 3 (not 1+2)
═══════════════════════════════════════════════════════════════

=== TASK ===
Follow the Step-by-Step Decision Guide above. Classify the {leg_label} leg.
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

=== TASK ===
Based on the shunt type "{shunt_type}", the clinical findings above, and the medical knowledge base provided:

1. Generate a detailed ligation plan with specific steps
2. Identify any additional clinical information needed
3. Consider complications and contraindications
4. Provide follow-up and monitoring recommendations
5. Consider CHIVA principles (hemodynamic, saphenous-vein-sparing when appropriate)

Important formatting rules:
1. ligation_steps must be a JSON array with one clear action per item.
2. Each ligation step must name the ligation point or vessel segment explicitly.
3. clinical_rationale must explain why that plan fits the shunt anatomy.
4. additional_info_needed must be [] when there is no meaningful extra information to request.
5. chiva_approach must describe the hemodynamic CHIVA reasoning, even if brief.

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


def _call_llm_for_shunt_classification(group: list[dict], leg_label: str, call_llm_fn: Callable) -> dict:
    """Task 1: Classify shunt type — NO RAG."""
    prompt = build_shunt_classification_prompt(group, leg_label)
    logger.info(f"Shunt classification LLM prompt for {leg_label}: {len(prompt)} chars")
    try:
        raw, usage = call_llm_fn(prompt, stream=False, return_usage=True)
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
        raw, usage = call_llm_fn(prompt, stream=False, return_usage=True)
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

    findings = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for leg_label, group in groups.items():
        # Step 1: Shunt Classification (NO RAG)
        classification = _call_llm_for_shunt_classification(group, leg_label, call_llm_fn)
        classification_usage = classification.pop("_llm_usage", {})
        shunt_type = classification.get("shunt_type", "Unknown")

        # Step 2: Ligation Planning (WITH RAG from ligation database)
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
