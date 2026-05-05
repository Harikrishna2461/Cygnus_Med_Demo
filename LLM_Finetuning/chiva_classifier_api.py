"""
CHIVA SHUNT CLASSIFIER API
Production-ready inference module for clinical integration
"""

import torch
import re
from typing import Dict, List, Optional
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class CHIVAShuntClassifier:
    """
    CHIVA venous shunt classifier using embedded decision rules and optional LoRA fine-tuning.
    """

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
"""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "auto",
        use_lora: bool = False,
        lora_path: str = "./lora_chiva_classifier",
        torch_dtype=None,
    ):
        """
        Initialize the classifier.

        Args:
            model_name: HuggingFace model ID
            device: Device to load model on ('cuda', 'cpu', 'auto')
            use_lora: Whether to load LoRA adapter
            lora_path: Path to LoRA adapter
            torch_dtype: Data type (default: bfloat16 for RTX 5090)
        """
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        print(f"Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_lora:
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.gradient_checkpointing_disable()
        self.model.eval()
        print("✓ Model ready for inference\n")

    def _format_clips(self, clips: List[Dict]) -> str:
        """Format clip data for prompt."""
        lines = []
        for i, c in enumerate(clips):
            flow = c.get("flow", "?")
            ft = c.get("fromType", "?")
            tt = c.get("toType", "?")
            y = c.get("posYRatio") or 0.0

            # Add anatomical annotations
            annotation = ""
            if flow == "EP" and ft == "N1" and tt == "N2":
                if y <= 0.098:
                    annotation = " [SFJ-ENTRY=INCOMPETENT]"
                elif y <= 0.353:
                    annotation = " [Hunterian-ENTRY=INCOMPETENT]"

            elim = c.get("eliminationTest", "").strip()
            elim_str = f' eliminationTest="{elim}"' if elim else ""

            lines.append(f"  Clip {i:02d}: {flow} {ft}→{tt}  y={y:.3f}{annotation}{elim_str}")

        return "\n".join(lines)

    def _build_prompt(self, clips: List[Dict], leg_label: str = "Assessment") -> str:
        """Build classification prompt with embedded CHIVA rules."""
        clips_str = self._format_clips(clips)

        return f"""{self.CHIVA_RULES}

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

    def _parse_response(self, text: str) -> Optional[Dict]:
        """Parse model response into structured format."""
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
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                result["confidence"] = 0.5
        else:
            result["confidence"] = 0.5

        # Extract REASONING
        reason_match = re.search(
            r"REASONING:\s*(.+?)(?=\n[A-Z]|$)", text, re.IGNORECASE | re.DOTALL
        )
        if reason_match:
            result["reasoning"] = reason_match.group(1).strip()
        else:
            result["reasoning"] = ""

        return result

    def classify(self, clips: List[Dict], leg_label: str = "Assessment") -> Dict:
        """
        Classify a venous shunt based on ultrasound clips.

        Args:
            clips: List of clip dictionaries with:
                - flow: 'EP' (antegrade) or 'RP' (retrograde)
                - fromType: 'N1', 'N2', 'N3'
                - toType: 'N1', 'N2', 'N3'
                - posYRatio: position along vein (0.0-1.0)
                - legSide: (optional) 'Left', 'Right'
                - eliminationTest: (optional) 'Reflux', 'No Reflux'
            leg_label: Label for this assessment (e.g., 'Left', 'Right')

        Returns:
            Dict with shunt_type, confidence, reasoning, raw_response
        """
        if not clips:
            return {
                "shunt_type": "No clips",
                "confidence": 0.0,
                "reasoning": "No ultrasound clips provided",
                "raw_response": "",
                "status": "error"
            }

        # Build prompt
        prompt = self._build_prompt(clips, leg_label)
        full_prompt = f"[INST] {prompt} [/INST]"

        # Generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()

        # Parse
        parsed = self._parse_response(response)

        if not parsed:
            return {
                "shunt_type": "Classification failed",
                "confidence": 0.0,
                "reasoning": "Could not parse model response",
                "raw_response": response[:500],
                "status": "error"
            }

        return {
            "shunt_type": parsed.get("shunt_type"),
            "confidence": parsed.get("confidence", 0.0),
            "reasoning": parsed.get("reasoning", ""),
            "raw_response": response[:500],
            "status": "success"
        }

    def batch_classify(self, batch: Dict[str, List[Dict]]) -> Dict:
        """
        Classify multiple legs at once.

        Args:
            batch: Dict with leg labels as keys, clip lists as values
                Example: {"Left": [clip1, clip2], "Right": [clip3, clip4]}

        Returns:
            Dict with results for each leg
        """
        results = {}
        for leg_label, clips in batch.items():
            results[leg_label] = self.classify(clips, leg_label)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize classifier
    classifier = CHIVAShuntClassifier(use_lora=False)  # Set to True if LoRA model exists

    # Example cases
    test_cases = {
        "Type 1": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300},
        ],
        "Type 2A": [
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.200},
            {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.470},
        ],
        "Type 3": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.050},
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212},
        ],
    }

    print("=" * 80)
    print("TESTING CHIVA CLASSIFIER API")
    print("=" * 80)

    for expected, clips in test_cases.items():
        print(f"\nTest case: {expected}")
        result = classifier.classify(clips, "Left")
        print(f"  Detected: {result['shunt_type']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Status: {result['status']}")
