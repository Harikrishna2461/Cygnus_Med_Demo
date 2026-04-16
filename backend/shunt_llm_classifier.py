"""
Shunt Classifier — Fully LLM-based with RAG context.

The LLM is given:
  1. The complete CHIVA classification rules (anatomy + decision tree)
  2. Retrieved chunks from the FAISS medical knowledge base (RAG)
  3. All clip data from the assessment

Outputs a structured classification per leg with reasoning and ligation plan.
"""

import json
import re
import logging
from contextlib import suppress
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE CHIVA RULES  (embedded knowledge — injected into every prompt)
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

  TYPE 1+2 Ligation — depends on RP N2→N1 calibre (set ask_diameter=true):
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
  ask_diameter    : true for Type 1+2 (need RP N2→N1 calibre to choose ligation strategy)
  ask_branching   : true for Type 2A/2B/2C with multiple RP at N3

CONFIDENCE GUIDE:
  Clear single pattern, no ambiguity         → 0.90–0.97
  Pattern present but some noise clips       → 0.80–0.89
  Ambiguous (needs elimination test)         → 0.50–0.65
  No pattern / insufficient clips            → 0.40–0.55
"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

_CLIP_LABELS: dict[tuple, str] = {
    ("EP", "N1", "N2"): None,           # handled separately (posYRatio-dependent)
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


def build_prompt(clips: list[dict], rag_context: str, leg_label: str) -> str:
    clips_str = _summarise_clips(clips)

    return f"""{CHIVA_RULES}

=== MEDICAL KNOWLEDGE BASE (retrieved via RAG) ===
{rag_context}

=== ASSESSMENT: {leg_label} ({len(clips)} clips) ===
{clips_str}

=== TASK ===
Using ONLY the Classification Rules above and the Medical Knowledge Base, classify the shunt type
for this {leg_label} leg assessment. Output ONLY the JSON below — no other text, no markdown.

{{
  "shunt_type": "<e.g. Type 1 / Type 2A / Type 2B / Type 2C / Type 3 / Type 1+2 / No shunt detected / Undetermined>",
  "confidence": <0.0-1.0>,
  "reasoning": ["<clinical bullet 1>", "<clinical bullet 2>", "..."],
  "ligation": ["<ligation step 1>", "<ligation step 2>", "..."],
  "needs_elim_test": <true/false>,
  "ask_diameter": <true/false>,
  "ask_branching": <true/false>,
  "summary": "<1 sentence plain-English clinical summary>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPAIR  (handles truncated output from small models)
# ─────────────────────────────────────────────────────────────────────────────

def _repair_and_parse(text: str) -> dict | None:
    if not text:
        return None
    # strip markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.rstrip())

    # try clean
    with suppress(Exception):
        return json.loads(text)

    # find first {
    start = text.find("{")
    if start == -1:
        return None
    raw = text[start:]

    # close unclosed strings / arrays / objects
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

    # regex fallback — extract what we can
    result: dict = {}
    for k in ("shunt_type", "summary"):
        if m := re.search(rf'"{k}"\s*:\s*"([^"]*)"', raw):
            result[k] = m[1]
    if cm := re.search(r'"confidence"\s*:\s*([\d.]+)', raw):
        result["confidence"] = float(cm[1])

    def ex_list(key):
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', raw)
        return re.findall(r'"([^"]+)"', m[1]) if m else []

    result["reasoning"] = ex_list("reasoning")
    result["ligation"]  = ex_list("ligation")
    if "shunt_type" not in result:
        return None
    for k in ("needs_elim_test", "ask_diameter", "ask_branching"):
        result.setdefault(k, False)
    result["_repaired"] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

_LLM_ERROR_RESULT: dict = {
    "shunt_type": "LLM classification failed",
    "confidence": 0.0,
    "reasoning": ["The LLM did not return a parseable response. Please retry."],
    "ligation": [],
    "needs_elim_test": False,
    "ask_diameter": False,
    "ask_branching": False,
    "summary": "Classification unavailable — LLM response could not be parsed.",
    "_llm_error": True,
}

_LEG_ORDER = {"Left": 0, "Right": 1}


def _retrieve_rag_context(group: list[dict], leg_label: str, retrieve_fn: Callable) -> str:
    try:
        rp_flows = [f"{c['fromType']}→{c['toType']}" for c in group if c.get("flow") == "RP"]
        ep_flows = [f"{c['fromType']}→{c['toType']}" for c in group if c.get("flow") == "EP"]
        query = (
            f"CHIVA shunt classification venous reflux "
            f"RP flows: {', '.join(rp_flows) or 'none'}. "
            f"EP flows: {', '.join(ep_flows[:4]) or 'none'}. "
            f"saphenofemoral junction GSV tributary ligation treatment"
        )
        if chunks := retrieve_fn(query, k=3):
            return "\n\n---\n\n".join(str(ch)[:600] for ch in chunks)
    except Exception as e:
        logger.warning(f"RAG retrieval failed for {leg_label}: {e}")
    return "No RAG context available."


def _call_llm_for_leg(group: list[dict], rag_context: str, leg_label: str, call_llm_fn: Callable) -> dict:
    prompt = build_prompt(group, rag_context, leg_label)
    logger.info(f"Shunt LLM prompt for {leg_label}: {len(prompt)} chars")
    try:
        raw = call_llm_fn(prompt, stream=False)
        logger.info(f"LLM raw response ({leg_label}): {raw[:300]!r}")
        result = _repair_and_parse(raw)
        if result and "shunt_type" in result:
            return result
    except Exception as e:
        logger.error(f"LLM call failed for {leg_label}: {e}")
    logger.error(f"LLM classification failed for {leg_label} — returning error result")
    return dict(_LLM_ERROR_RESULT)


def classify_shunt_with_llm(
    clip_list: list[dict[str, Any]],
    call_llm_fn: Callable,
    retrieve_context_fn: Callable | None = None,
) -> dict:
    """
    Classify shunts from a post-assessment clip_list using LLM + RAG.
    Splits clips by legSide, retrieves RAG context, calls LLM per leg.
    Returns top-level keys for backward compat + 'findings' list.
    """
    groups: dict[str, list[dict]] = {}
    for c in clip_list:
        side = (c.get("legSide") or c.get("leg_side") or "Assessment").strip().capitalize()
        groups.setdefault(side, []).append(c)

    findings = []
    for leg_label, group in groups.items():
        rag_context = (
            _retrieve_rag_context(group, leg_label, retrieve_context_fn)
            if retrieve_context_fn else "No RAG context available."
        )
        llm_result = _call_llm_for_leg(group, rag_context, leg_label, call_llm_fn)
        llm_result["leg"]       = leg_label
        llm_result["num_clips"] = len(group)
        findings.append(llm_result)

    findings.sort(key=lambda f: _LEG_ORDER.get(f["leg"], 2))

    primary = findings[0] if findings else {}
    return {
        "findings":        findings,
        "shunt_type":      primary.get("shunt_type", "No shunt detected"),
        "confidence":      primary.get("confidence", 0.0),
        "reasoning":       primary.get("reasoning", []),
        "ligation":        primary.get("ligation", []),
        "needs_elim_test": primary.get("needs_elim_test", False),
        "ask_diameter":    primary.get("ask_diameter", False),
        "ask_branching":   primary.get("ask_branching", False),
        "summary":         primary.get("summary", ""),
        "num_clips":       len(clip_list),
        "num_findings":    len(findings),
    }
