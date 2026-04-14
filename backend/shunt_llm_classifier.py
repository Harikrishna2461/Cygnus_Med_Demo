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
  N2 = Great Saphenous Vein (GSV) trunk
  N3 = Tributaries / superficial branches
  EP = Physiological (forward, antegrade) flow — NORMAL
  RP = Retrograde (pathological, reflux) flow — ABNORMAL
  SFJ = Saphenofemoral Junction  →  posYRatio ≤ 0.098
  Hunterian Perforator            →  0.098 < posYRatio ≤ 0.353

CLASSIFICATION DECISION TREE:

Case A — EP N1→N2 exists (at SFJ or Hunterian), NO EP N2→N3:
  • If RP N2→N1 present, NO RP at N3  →  TYPE 1
    Ligation: Ligate at SFJ (if SFJ entry) or Hunterian Perforator (if Hunterian entry).
              If multiple RP N2→N1, ligate below each RP along N2 except the most distal.

Case B — EP N1→N2 exists AND EP N2→N3 exists:
  B1: RP N3→N2 or N3→N1, but NO RP N2→N1          →  TYPE 3
  B2: RP N3→N2 AND RP N2→N1                         →  TYPE 3
  B3: RP N3→N1 AND RP N2→N1, eliminationTest=""    →  UNDETERMINED — elimination test required
  B4: RP N3→N1 AND RP N2→N1, eliminationTest="Reflux"   →  TYPE 1+2
  B5: RP N3→N1 AND RP N2→N1, eliminationTest="No Reflux" →  TYPE 3

  TYPE 3 Ligation:
    Single RP at tributary: Ligate EP at N2→N3. Follow up 6–12 months for proximal N2 reflux. If reflux, ligate SFJ.
    Multiple RP at tributary: Ligate every refluxing tributary at N2 junction (CHIVA 2 step 1). Follow up same.

  TYPE 1+2 Ligation:
    Small RP diameter: Apply CHIVA 2 (ligate EP N2→N3 first, then SFJ/Hunterian). OR ligate SFJ first + all tributaries except one; once N2 normalises ligate last tributary.
    Large RP / multiple RP N2→N1: Ligate SFJ/Hunterian + every refluxing tributary. Ligate below each RP N2→N1 except most distal. Check if any RP near N2 is dilated.

Case C — EP N2→N3 or EP N2→N2 exists, NO EP N1→N2:
  C1: RP N3 only, NO RP N2→N1   →  TYPE 2A (if EP N2→N3) or TYPE 2B (if EP N2→N2)
  C2: RP N3 AND RP N2→N1        →  TYPE 2C
  Ligation: Ligate highest EP at N2→N3. If multiple N3 branches: choose based on calibre, distance to perforator, drainage.

Case D — No RP found in any clip  →  NO SHUNT DETECTED. No ligation needed.

COORDINATE HINTS:
  posYRatio ≤ 0.098  = SFJ region (upper thigh)
  0.099–0.353        = Hunterian / mid-thigh
  0.354–0.60         = Knee / popliteal
  > 0.60             = Calf / ankle

CONFIDENCE GUIDE:
  Clear single pattern, no ambiguity         → 0.90–0.97
  Pattern present but some noise clips       → 0.80–0.89
  Ambiguous (needs elimination test)         → 0.50–0.65
  No pattern / insufficient clips            → 0.40–0.55
"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_clips(clips: list[dict]) -> str:
    lines = []
    for i, c in enumerate(clips):
        flow = c.get("flow", "?")
        ft   = c.get("fromType", "?")
        tt   = c.get("toType", "?")
        y    = c.get("posYRatio") or 0.0
        elim = (c.get("eliminationTest") or "").strip()

        loc = ""
        if flow == "EP" and ft == "N1" and tt == "N2":
            loc = " [SFJ]" if y <= 0.098 else " [Hunterian]" if y <= 0.353 else ""

        elim_str = f'  eliminationTest="{elim}"' if elim else ""
        lines.append(f"  Clip {i:02d}: {flow} {ft}→{tt}  y={y:.3f}{loc}{elim_str}")
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

    try:
        return json.loads(raw)
    except Exception:
        pass

    # regex fallback — extract what we can
    result: dict = {}
    for k in ("shunt_type", "summary"):
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', raw)
        if m: result[k] = m.group(1)
    cm = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
    if cm: result["confidence"] = float(cm.group(1))

    def ex_list(key):
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', raw)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []

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

def classify_shunt_with_llm(
    clip_list: list[dict[str, Any]],
    call_llm_fn: Callable,
    retrieve_context_fn: Callable | None = None,
) -> dict:
    """
    Classify shunts from a post-assessment clip_list using LLM + RAG.

    - Splits clips by legSide
    - Retrieves RAG context (FAISS) for each leg
    - Asks LLM to classify using full CHIVA rules + RAG context
    - Returns an explicit error result if LLM fails (no rule-based fallback)

    Returns top-level keys for backward compat + 'findings' list.
    """
    # Split by leg
    groups: dict[str, list[dict]] = {}
    for c in clip_list:
        side = (c.get("legSide") or c.get("leg_side") or "Assessment").strip().capitalize()
        groups.setdefault(side, []).append(c)

    if list(groups.keys()) == ["Assessment"]:
        # no legSide info — single group
        pass

    findings = []
    for leg_label, group in groups.items():
        # ── RAG retrieval ──────────────────────────────────────
        rag_context = "No RAG context available."
        if retrieve_context_fn:
            try:
                # Build a query that describes this leg's clip pattern
                rp_flows = [f"{c['fromType']}→{c['toType']}" for c in group
                            if c.get("flow") == "RP"]
                ep_flows = [f"{c['fromType']}→{c['toType']}" for c in group
                            if c.get("flow") == "EP"]
                query = (
                    f"CHIVA shunt classification venous reflux "
                    f"RP flows: {', '.join(rp_flows) or 'none'}. "
                    f"EP flows: {', '.join(ep_flows[:4]) or 'none'}. "
                    f"saphenofemoral junction GSV tributary ligation treatment"
                )
                chunks = retrieve_context_fn(query, k=3)
                if chunks:
                    # Each chunk is a raw string from the FAISS metadata
                    rag_context = "\n\n---\n\n".join(
                        str(ch)[:600] for ch in chunks
                    )
            except Exception as e:
                logger.warning(f"RAG retrieval failed for {leg_label}: {e}")

        # ── LLM call ──────────────────────────────────────────
        prompt = build_prompt(group, rag_context, leg_label)
        logger.info(f"Shunt LLM prompt for {leg_label}: {len(prompt)} chars")

        llm_result = None
        try:
            raw = call_llm_fn(prompt, stream=False)
            logger.info(f"LLM raw response ({leg_label}): {raw[:300]!r}")
            llm_result = _repair_and_parse(raw)
        except Exception as e:
            logger.error(f"LLM call failed for {leg_label}: {e}")

        # ── If LLM fails return an explicit error — no rule-based fallback ──────
        if not llm_result or "shunt_type" not in llm_result:
            logger.error(f"LLM classification failed for {leg_label} — no fallback, returning error")
            llm_result = {
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

        llm_result["leg"]       = leg_label
        llm_result["num_clips"] = len(group)
        findings.append(llm_result)

    # Sort: Left first, Right second
    findings.sort(key=lambda f: {"Left": 0, "Right": 1}.get(f["leg"], 2))

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
