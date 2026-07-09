"""
VLM-based ultrasound frame reader using Groq (Llama-4-Scout).
Job: convert annotated frame visual into textual anatomy context for the guidance LLM.
Reports only which N1/N2/N3 vessels are annotated and their position relative to fascia.
No flow, no Doppler, no EP/RP — those are the guidance LLM's job.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

REGION_TARGET_VESSELS: dict[str, str] = {
    "SFJ":        "Saphenofemoral Junction — GSV (N2) meets femoral vein (N1) at groin",
    "Upper Thigh":"GSV trunk (N2) in upper thigh, within fascial compartment",
    "Hunterian":  "GSV trunk (N2) in proximal thigh (Hunter's canal), within fascial compartment",
    "Dodd":       "GSV trunk (N2) in distal thigh, within fascial compartment",
    "GSV-THI":    "GSV trunk (N2) in thigh, within fascial compartment",
    "GSV-CAL":    "GSV trunk (N2) in calf, within fascial compartment",
    "SPJ":        "Saphenopopliteal Junction — SSV (N2) meets popliteal vein (N1) behind knee",
    "SSV":        "SSV trunk (N2) in posterior calf",
    "Giacomini":  "Giacomini vein / posterior thigh — SSV tributary running up posterior thigh, may show SSV or Tributary annotation",
    "Calf":       "GSV in medial calf, within fascial compartment",
    "UNKNOWN":    "superficial venous structure",
}

_SYSTEM_PROMPT = (
    "You read annotated venous duplex ultrasound frames. Respond ONLY with valid compact JSON — no markdown, no explanation.\n\n"
    "THESE FRAMES USE ONE OF TWO ANNOTATION STYLES:\n\n"
    "STYLE A — Guidance frames (have yellow fascia lines):\n"
    "  Two bright YELLOW annotation lines (horizontal or gently curved, spanning the full image width) = saphenous fascia boundary.\n"
    "  fascial_layer_visible=true ONLY if you see these two artificial yellow lines.\n"
    "  IMPORTANT: Yellow lines alone do NOT mean a vein is present. You must ALSO find:\n"
    "    - An anechoic (dark oval) vessel visible in the image, AND\n"
    "    - A printed text label near that vessel.\n"
    "  Label format in Style A: 'N2 [n]' or 'N2 [n] HL' = saphenous trunk between the fascia lines → label_n2_visible=true, n2_in_fascial_compartment=true.\n"
    "  'N3 [n]' or 'N3 [n] SL' = tributary above fascia lines → label_n3_visible=true, n3_superficial_to_fascia=true.\n"
    "  'N1 [n]' = deep vein below fascia lines → label_n1_visible=true, n1_deep_to_fascia=true.\n"
    "  If you see yellow fascia lines but NO dark vessel oval and NO N-label text: fascial_layer_visible=true, all others false.\n\n"
    "STYLE B — Segmented vein frames (NO yellow fascia lines):\n"
    "  Oval contour drawn around each vessel + VEIN NAME text label (colour varies — ignore outline colour).\n"
    "  Read the TEXT LABEL near each oval:\n"
    "    'GSV', 'GSV_PROX', 'GSV_DISTAL' → label_n2_visible=true, n2_in_fascial_compartment=true, add 'GSV' to vein_types\n"
    "    'TRIBUTARY' → label_n3_visible=true, n3_superficial_to_fascia=true, add 'Tributary' to vein_types\n"
    "    'SSV' → label_n2_visible=true, n2_in_fascial_compartment=true, add 'SSV' to vein_types\n"
    "    'CFV', 'FV', 'PV', 'DFV' → label_n1_visible=true, n1_deep_to_fascia=true, add that name to vein_types\n"
    "    'AASV', 'PASV' → add to vein_types\n"
    "  fascial_layer_visible=false (no yellow fascia lines in Style B).\n\n"
    "CRITICAL RULES:\n"
    "  1. Seeing yellow fascia lines is NOT sufficient to set any label_n* or n*_in_* field true.\n"
    "  2. Only set label_n2_visible=true if you can read 'N2' text OR an oval labeled GSV/SSV.\n"
    "  3. NEVER guess or infer a vein from anatomy alone — only from visible text labels.\n"
    "  4. Do not report SSV unless the text 'SSV' is literally printed in the image.\n\n"
    "frame_note: 1 natural sentence describing what you actually see in this image "
    "(e.g. 'Yellow fascia lines visible with no labeled vessel between them.' or "
    "'GSV oval labeled N2 visible between yellow fascia lines.' or "
    "'Oval labeled GSV_DISTAL and another labeled TRIBUTARY, no fascia lines.')"
)

_ANALYSIS_SCHEMA = """{
  "image_quality": "good | fair | poor | unusable",
  "fascial_layer_visible": true or false,
  "n2_in_fascial_compartment": true or false,
  "n3_superficial_to_fascia": true or false,
  "n1_deep_to_fascia": true or false,
  "label_n1_visible": true or false,
  "label_n2_visible": true or false,
  "label_n3_visible": true or false,
  "vein_types": ["GSV", "FV"],
  "frame_note": "1 natural sentence: what structures are actually visible (e.g. 'Yellow fascia lines present, no labeled vessel visible.' or 'Dark oval labeled N2 between fascia lines.')"
}"""


@dataclass
class UltrasoundAssessment:
    image_quality: str = "unknown"
    fascial_layer_visible: bool = False
    n2_in_fascial_compartment: bool = False
    n3_superficial_to_fascia: bool = False
    n1_deep_to_fascia: bool = False
    label_n1_visible: bool = False
    label_n2_visible: bool = False
    label_n3_visible: bool = False
    vein_types: list = None  # type: ignore
    frame_note: str = ""
    raw_text: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        if self.vein_types is None:
            self.vein_types = []

    def to_dict(self) -> dict:
        return {
            "image_quality": self.image_quality,
            "fascial_layer_visible": self.fascial_layer_visible,
            "n2_in_fascial_compartment": self.n2_in_fascial_compartment,
            "n3_superficial_to_fascia": self.n3_superficial_to_fascia,
            "n1_deep_to_fascia": self.n1_deep_to_fascia,
            "label_n1_visible": self.label_n1_visible,
            "label_n2_visible": self.label_n2_visible,
            "label_n3_visible": self.label_n3_visible,
            "vein_types": self.vein_types,
            "frame_note": self.frame_note,
            "error": self.error,
            "raw_text": self.raw_text,
        }

    def summary(self) -> str:
        """Textual anatomy context passed to the guidance LLM."""
        if self.error:
            return f"Frame read failed: {self.error}"
        parts = []
        if self.fascial_layer_visible:
            parts.append("Fascial layer visible.")
        else:
            parts.append("Fascial layer not visible.")
        if self.n2_in_fascial_compartment:
            parts.append("N2 (saphenous trunk/GSV) annotated within fascial compartment.")
        else:
            parts.append("N2 (saphenous trunk) not annotated in this frame.")
        if self.n3_superficial_to_fascia:
            parts.append("N3 (tributary) annotated above fascia, superficial.")
        if self.n1_deep_to_fascia:
            parts.append("N1 (deep vein) annotated below fascia.")
        if self.vein_types:
            parts.append(f"Veins identified: {', '.join(self.vein_types)}.")
        if self.frame_note:
            parts.append(self.frame_note)
        return " ".join(parts)


_N2_VEIN_FOR_REGION = {
    "SFJ": "GSV", "Upper Thigh": "GSV", "Hunterian": "GSV",
    "Dodd": "GSV", "GSV-THI": "GSV", "GSV-CAL": "GSV", "Calf": "GSV",
    "SPJ": "SSV", "SSV": "SSV", "Giacomini": "SSV",
}
_N1_VEIN_FOR_REGION = {
    "SFJ": "CFV", "Upper Thigh": "FV", "Hunterian": "FV",
    "Dodd": "FV", "GSV-THI": "FV", "GSV-CAL": "FV",
    "SPJ": "PV", "SSV": "PV", "Calf": "FV",
}

def _build_user_prompt(region: str, leg: str) -> str:
    target = REGION_TARGET_VESSELS.get(region, "superficial venous structure")
    n2_vein = _N2_VEIN_FOR_REGION.get(region, "GSV")
    n1_vein = _N1_VEIN_FOR_REGION.get(region, "FV")
    return (
        f"Region: {region} ({leg} leg). Context: {target}.\n"
        f"Step 1 — scan the image and READ EVERY TEXT LABEL printed near an oval/circle outline.\n"
        f"Step 2 — fill in the JSON below based ONLY on what labels you literally read:\n"
        f"{_ANALYSIS_SCHEMA}\n"
        f"STRICT RULES for vein_types:\n"
        f"  Style B (no yellow fascia lines): read the exact text labels on ovals — "
        f"'TRIBUTARY'→add 'Tributary'; 'GSV','GSV_PROX','GSV_DISTAL'→add 'GSV'; "
        f"'SSV'→add 'SSV'; 'CFV'→add 'CFV'; 'FV'→add 'FV'; 'AASV'→add 'AASV'; 'PASV'→add 'PASV'. "
        f"Report EVERY labeled vessel — if two ovals with different labels, add both.\n"
        f"  Style A (yellow fascia lines present): if N2 label visible → add '{n2_vein}'; "
        f"if N3 label visible → add 'Tributary'; if N1 label visible → add '{n1_vein}'.\n"
        f"NEVER add SSV unless the text 'SSV' or 'N2' in SPJ/SSV region is literally in the image. "
        f"Output JSON only."
    )


def _parse_vlm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.startswith("```"))
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _assessment_from_dict(d: dict, raw: str, region: str = "") -> UltrasoundAssessment:
    raw_vt = d.get("vein_types", [])
    vein_types: list = list(raw_vt) if isinstance(raw_vt, list) else []

    label_n1 = bool(d.get("label_n1_visible", False))
    label_n2 = bool(d.get("label_n2_visible", False))
    label_n3 = bool(d.get("label_n3_visible", False))
    n2 = bool(d.get("n2_in_fascial_compartment", False))
    n3 = bool(d.get("n3_superficial_to_fascia", False))
    n1 = bool(d.get("n1_deep_to_fascia", False))

    # Style A guidance frames: N-class labels are detected → rebuild vein_types from
    # region + N-class only. Discard VLM's vein_types text (model hallucinates wrong
    # vein names, e.g. SSV for GSV regions). Region tells us exactly what each N-class is.
    if label_n2 or n2 or label_n3 or n3 or label_n1 or n1:
        vein_types = []
        if label_n2 or n2:
            vein_types.append(_N2_VEIN_FOR_REGION.get(region, "GSV"))
            n2 = True
        if label_n3 or n3:
            vein_types.append("Tributary")
            n3 = True
        if label_n1 or n1:
            vein_types.append(_N1_VEIN_FOR_REGION.get(region, "FV"))
            n1 = True
    # else: Style B or stream — use VLM-reported vein_types as-is
    # (vein_frames are bypassed before reaching here; stream is best-effort)

    return UltrasoundAssessment(
        image_quality=d.get("image_quality", "unknown"),
        fascial_layer_visible=bool(d.get("fascial_layer_visible", False)),
        n2_in_fascial_compartment=n2,
        n3_superficial_to_fascia=n3,
        n1_deep_to_fascia=n1,
        label_n1_visible=label_n1,
        label_n2_visible=label_n2,
        label_n3_visible=label_n3,
        vein_types=vein_types,
        frame_note=d.get("frame_note", ""),
        raw_text=raw,
    )


def analyze_frame(
    base64_image: str,
    region: str,
    leg: str,
    media_type: str = "image/jpeg",
) -> UltrasoundAssessment:
    from config import GROQ_API_KEY, GROQ_VISION_MODEL

    if not GROQ_API_KEY:
        return UltrasoundAssessment(error="GROQ_API_KEY not configured.")

    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=GROQ_API_KEY)
        prompt = _build_user_prompt(region, leg)
        data_url = f"data:{media_type};base64,{base64_image}"
        resp = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            max_tokens=512,
            temperature=0.0,
            timeout=15,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        raw = resp.choices[0].message.content or ""
        d = _parse_vlm_json(raw)
        return _assessment_from_dict(d, raw, region)

    except Exception as exc:
        logger.error("Groq VLM error: %s", exc)
        return UltrasoundAssessment(error=str(exc))
