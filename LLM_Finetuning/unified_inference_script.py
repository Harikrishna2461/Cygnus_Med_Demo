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
import torch
from peft import PeftModel
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE CHIVA RULES (embedded knowledge — used in shunt classification only)
# ─────────────────────────────────────────────────────────────────────────────
CHIVA_RULES = """
=== CHIVA VENOUS SHUNT CLASSIFICATION RULES ===

ANATOMY:
    N1 = Deep venous system (femoral / popliteal vein)
    N2 = Great Saphenous Vein (GSV) or Small Saphenous Vein (SSV) trunk
    N3 = Tributaries / superficial branches
    EP = Physiological (forward, antegrade) flow — NORMAL
    RP = Retrograde (pathological, reflux) flow — ABNORMAL
    SFJ = Saphenofemoral Junction  →  posYRatio ≤ 0.098
    Hunterian Perforator            →  0.098 < posYRatio ≤ 0.353

═══════════════════════════════════════════════════════════
CRITICAL RULE — SFJ COMPETENCE:
    SFJ is INCOMPETENT iff clip has EP N1→N2 (fromType=N1, toType=N2).
    EP N2→N2 means perforator entry (SFJ COMPETENT).
═══════════════════════════════════════════════════════════

STEP 1: CHECK FOR EP N1→N2
    YES → SFJ INCOMPETENT → Case A or B
    NO  → SFJ COMPETENT → Case C

Case A: EP N1→N2 exists, NO EP N2→N3
    RP N2→N1 only, no RP at N3 → TYPE 1

Case B: EP N1→N2 exists AND EP N2→N3 exists
    RP N3 only (no RP N2→N1) → TYPE 3
    RP N3 AND RP N2→N1 + elim test absent → UNDETERMINED
    RP N3 AND RP N2→N1 + elim="Reflux" → TYPE 1+2
    RP N3 AND RP N2→N1 + elim="No Reflux" → TYPE 3

Case C: NO EP N1→N2 (SFJ COMPETENT)
    EP N2→N3 present → TYPE 2A
    EP N2→N2 (perforator) + RP N3 only → TYPE 2B
    EP N2→N2 (perforator) + RP N3 + RP N2→N1 → TYPE 2C
    EP N2→N2 + no RP → NO SHUNT

QUICK TABLE:
    EP N1→N2 + no EP N2→N3 + RP N2→N1           → TYPE 1
    EP N1→N2 + EP N2→N3 + RP N3 only             → TYPE 3
    EP N1→N2 + EP N2→N3 + RP N3+N2→N1 + elim absent → UNDETERMINED
    EP N1→N2 + EP N2→N3 + RP N3+N2→N1 + elim="Reflux"  → TYPE 1+2
    EP N1→N2 + EP N2→N3 + RP N3+N2→N1 + elim="No Reflux" → TYPE 3
    No EP N1→N2 + EP N2→N3                      → TYPE 2A
    No EP N1→N2 + EP N2→N2 + RP N3 only         → TYPE 2B
    No EP N1→N2 + EP N2→N2 + RP N3 + RP N2→N1  → TYPE 2C
    No RP anywhere                               → NO SHUNT
"""

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: SHUNT CLASSIFICATION (No RAG)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_label(flow: str, ft: str, tt: str, y: float) -> str:
    if flow == "EP" and ft == "N1" and tt == "N2":
        if y <= 0.098:
            return " [SFJ-ENTRY=INCOMPETENT]"
        return " [Hunterian-ENTRY=INCOMPETENT]" if y <= 0.353 else " [Deep-to-GSV-ENTRY]"
    if flow == "RP" and ft == "N3":
        return f" [TRIBUTARY-REFLUX: N3→{tt}]"
    return ""

def _summarise_clips(clips: list[dict]) -> str:
    lines = []
    for i, c in enumerate(clips):
        flow = c.get("flow", "?")
        ft   = c.get("fromType", "?")
        tt   = c.get("toType", "?")
        y    = c.get("posYRatio") or 0.0
        elim = (c.get("eliminationTest") or "").strip()

        loc = _clip_label(flow, ft, tt, y)
        parts = [f"  Clip {i:02d}: {flow} {ft}→{tt}  y={y:.3f}{loc}"]
        if elim:
            parts.append(f'eliminationTest="{elim}"')
        lines.append(" ".join(parts))
    return "\n".join(lines)

def build_shunt_classification_prompt(clips: list[dict], leg_label: str) -> str:
    """Build prompt for shunt classification — NO RAG context."""
    clips_str = _summarise_clips(clips)

    return f"""{CHIVA_RULES}

=== ASSESSMENT: {leg_label} ({len(clips)} clips) ===
{clips_str}

═══════════════════════════════════════════════════════════════
CLASSIFICATION TASK
═══════════════════════════════════════════════════════════════

Follow the CHIVA rules above. Classify the {leg_label} leg shunt type.

Output in EXACTLY this format (no JSON, just plain text):

TYPE: [shunt type here]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]
"""

def _repair_and_parse(text: str) -> dict | None:
    """Parse plain text response format."""
    if not text:
        return None

    result = {}

    # Extract TYPE
    type_match = re.search(r"TYPE:\s*([^\n]+)", text, re.IGNORECASE)
    if type_match:
        result["shunt_type"] = type_match.group(1).strip()
    else:
        return None

    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text, re.IGNORECASE)
    if conf_match:
        result["confidence"] = float(conf_match.group(1))
    else:
        result["confidence"] = 0.5

    # Extract REASONING
    reason_match = re.search(r"REASONING:\s*(.+?)(?=\n[A-Z]|$)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        result["reasoning"] = [reason_match.group(1).strip()]
    else:
        result["reasoning"] = ["Unable to extract reasoning"]

    result["summary"] = f"{result.get('shunt_type', 'Unknown')} - {result.get('confidence', 0):.0%} confidence"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — UNIFIED INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def classify_shunt_with_lora_model(
    clip_list: list[dict],
    model,
    tokenizer,
    use_lora: bool = True,
    lora_path: str = "./lora_chiva_classifier",
) -> dict:
    """
    Classify shunts using base model (optionally with LoRA).

    Args:
        clip_list: List of clip dictionaries with flow, fromType, toType, posYRatio
        model: Loaded base model
        tokenizer: Loaded tokenizer
        use_lora: If True, load and apply LoRA adapter
        lora_path: Path to LoRA adapter directory

    Returns:
        Classification result with shunt type, confidence, reasoning
    """

    # Load LoRA once if requested
    inference_model = model
    if use_lora:
        inference_model = PeftModel.from_pretrained(model, lora_path)

    inference_model.gradient_checkpointing_disable()
    inference_model.eval()

    # Group by leg
    groups = {}
    for c in clip_list:
        leg = (c.get("legSide") or c.get("leg_side") or "Assessment").strip().capitalize()
        groups.setdefault(leg, []).append(c)

    findings = []

    for leg_label, group in groups.items():
        # Build classification prompt
        prompt = build_shunt_classification_prompt(group, leg_label)
        full_prompt = f"[INST] {prompt} [/INST]"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(inference_model.device)

        # Generate response
        with torch.no_grad():
            outputs = inference_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()

        # Parse response
        classification = _repair_and_parse(response)

        if not classification:
            classification = {
                "shunt_type": "Classification failed",
                "confidence": 0.0,
                "reasoning": ["LLM response could not be parsed"],
                "summary": "Unable to classify"
            }

        finding = {
            "leg": leg_label,
            "shunt_type": classification.get("shunt_type", "Unknown"),
            "confidence": classification.get("confidence", 0.0),
            "reasoning": classification.get("reasoning", []),
            "summary": classification.get("summary", ""),
            "num_clips": len(group),
            "llm_response": response[:500]  # For debugging
        }
        findings.append(finding)

    # Return primary finding
    if not findings:
        return {"shunt_type": "No findings", "confidence": 0.0}

    primary = findings[0]
    return {
        "findings": findings,
        "shunt_type": primary.get("shunt_type"),
        "confidence": primary.get("confidence", 0.0),
        "reasoning": primary.get("reasoning", []),
        "summary": primary.get("summary", ""),
        "num_clips": len(clip_list),
        "num_findings": len(findings),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example test cases
    test_clips = {
        "Type 1": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080, "legSide": "Left"},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300, "legSide": "Left"},
        ],
        "Type 2A": [
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.200, "legSide": "Left"},
            {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.470, "legSide": "Left"},
        ],
        "Type 3": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.050, "legSide": "Left"},
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132, "legSide": "Left"},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212, "legSide": "Left"},
        ],
    }

    print("Testing unified classification script...")
    print("Note: Model and tokenizer must be loaded before calling classify_shunt_with_lora_model()")
