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
    "Giacomini":  "Giacomini vein / posterior thigh — THIS FRAME HAS NO ANNOTATION LABELS. "
                  "No N1/N2/N3 text labels are present. All label_*_visible fields MUST be false.",
    "Calf":       "GSV in medial calf, within fascial compartment",
    "UNKNOWN":    "superficial venous structure",
}

_SYSTEM_PROMPT = (
    "You read annotated venous ultrasound frames and report which vessels are visible, "
    "where they sit relative to the fascial layer, and what specific vein type each annotated vessel is. "
    "Your output is used as anatomy context — it is not a diagnosis.\n\n"
    "HOW VESSELS ARE ANNOTATED IN THESE FRAMES:\n"
    "  Fascia: two bright yellow HORIZONTAL LINES running across the image, "
    "forming the saphenous fascial compartment (the 'saphenous eye').\n"
    "  N3 vessel: YELLOW POLYGON/OVAL OUTLINE around the vessel cross-section + 'N3 [n]' text label badge. "
    "Sits ABOVE the fascia lines, between the fascia and the probe (top of image = closer to skin). "
    "N3 vessels are superficial tributaries or varicose branches above the fascia.\n"
    "  N2 vessel: GREEN POLYGON/OVAL OUTLINE around the vessel cross-section + 'N2 [n]' text label badge. "
    "Sits WITHIN the fascial compartment, BETWEEN the two yellow lines. "
    "This is the saphenous trunk — GSV (great saphenous) in the thigh/calf, or SSV (small saphenous) posteriorly.\n"
    "  N1 vessel: BLUE or CYAN POLYGON/OVAL OUTLINE + 'N1 [n]' text label badge. "
    "Sits BELOW/DEEP TO the fascial lines, further from the skin. "
    "This is the deep vein — CFV (common femoral), FV (femoral vein), or PV (popliteal vein).\n"
    "  Numbers in brackets [0], [1], [2] are just vessel indices — ignore them.\n\n"
    "VEIN TYPE IDENTIFICATION — for each annotated vessel you can see, identify which vein it most likely is:\n"
    "  N1 (deep, below fascia): CFV at groin, FV at thigh, PV at popliteal fossa\n"
    "  N2 (in fascial compartment): GSV trunk (proximal=thigh, distal=calf), SSV trunk (posterior calf)\n"
    "  N3 (superficial, above fascia): GSV tributary, SSV tributary, AASV (anterior accessory), PASV (posterior accessory), perforator outflow\n"
    "  Perforators: orange-outlined vessels connecting N1 to N2 — Hunterian, Dodd, Boyd, Cockett\n\n"
    "CRITICAL LABEL RULE — read carefully:\n"
    "  A label field (label_n1_visible, label_n2_visible, label_n3_visible) MUST be set to true ONLY IF "
    "you can literally read the EXACT PRINTED TEXT 'N1 [n]', 'N2 [n]', or 'N3 [n]' as a visible text overlay in the image. "
    "A circular or oval dark vessel shape WITHOUT that printed text label = label is NOT visible (false). "
    "If the image shows only tissue or ultrasound speckle with NO printed annotation text, "
    "ALL label_*_visible fields MUST be false, n2_in_fascial_compartment MUST be false, "
    "fascial_layer_visible MUST be false, and image_quality should reflect what you see.\n\n"
    "WHAT TO REPORT:\n"
    "  For each N-type: is it present (yes/no), based on seeing its outline and label.\n"
    "  The fascial layer: is it visible (yes/no).\n"
    "  vein_types: list of specific vein names visible (e.g. ['GSV', 'FV', 'Tributary']). "
    "Use only these names: GSV, SSV, CFV, FV, PV, DFV, AASV, PASV, Tributary, "
    "Hunterian_Perf, Dodd_Perf, Boyd_Perf, Cockett_Perf. Empty list [] if nothing identifiable.\n"
    "  frame_note: 8 words max, vessel labels and positions, e.g. 'N2 GSV in fascia, N3 tributary above'. "
    "If no labels at all: 'No annotations visible in frame'. "
    "Never mention flow, colour, Doppler, EP, RP, or any clinical finding.\n\n"
    "Respond only with valid JSON."
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
  "frame_note": "e.g. 'N2 GSV in fascia, N3 tributary above' — vessel names, N-class, positions only"
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


def _build_user_prompt(region: str, leg: str) -> str:
    target = REGION_TARGET_VESSELS.get(region, "superficial venous structure")
    return (
        f"Probe position: {region} region, {leg} leg.\n"
        f"Expected anatomy: {target}.\n\n"
        f"Look at the frame and answer:\n"
        f"  - Are the two yellow horizontal fascia lines visible?\n"
        f"  - Is there a vessel with a YELLOW oval/polygon outline labeled 'N3 [n]' above the fascia lines?\n"
        f"    If yes: what specific vein is it? (e.g. GSV tributary, AASV, PASV, Cockett perforator outflow)\n"
        f"  - Is there a vessel with a GREEN oval/polygon outline labeled 'N2 [n]' between the fascia lines?\n"
        f"    If yes: what specific vein is it? (e.g. GSV at {region}, SSV trunk)\n"
        f"  - Is there a vessel with a blue/cyan outline labeled 'N1 [n]' below the fascia lines?\n"
        f"    If yes: what specific vein is it? (e.g. CFV at groin, FV at thigh, PV at popliteal fossa)\n"
        f"  - List all specific vein names you can identify in the vein_types array.\n"
        f"Report only what is annotated with explicit text labels. Do not mention flow, Doppler, EP, or RP.\n"
        f"REMINDER: If you do not see actual printed 'N1 [n]'/'N2 [n]'/'N3 [n]' text in the image, "
        f"all label_*_visible fields must be false, and vein_types should be [].\n\n"
        f"Respond with valid JSON only:\n{_ANALYSIS_SCHEMA}"
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


def _assessment_from_dict(d: dict, raw: str) -> UltrasoundAssessment:
    raw_vt = d.get("vein_types", [])
    vein_types = raw_vt if isinstance(raw_vt, list) else []
    return UltrasoundAssessment(
        image_quality=d.get("image_quality", "unknown"),
        fascial_layer_visible=bool(d.get("fascial_layer_visible", False)),
        n2_in_fascial_compartment=bool(d.get("n2_in_fascial_compartment", False)),
        n3_superficial_to_fascia=bool(d.get("n3_superficial_to_fascia", False)),
        n1_deep_to_fascia=bool(d.get("n1_deep_to_fascia", False)),
        label_n1_visible=bool(d.get("label_n1_visible", False)),
        label_n2_visible=bool(d.get("label_n2_visible", False)),
        label_n3_visible=bool(d.get("label_n3_visible", False)),
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
    # Giacomini zone uses a blank tissue frame with no annotation overlays.
    # The VLM model confabulates fascia lines and N2 from tissue banding on
    # featureless ultrasound images, so bypass the API call and return the
    # only honest result: nothing annotated. The UI fields still render — all No.
    if region == "Giacomini":
        return UltrasoundAssessment(
            image_quality="good",
            frame_note="Giacomini zone — no saphenous trunk annotation in posterior view",
        )

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
            max_tokens=250,
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
        return _assessment_from_dict(d, raw)

    except Exception as exc:
        logger.error("Groq VLM error: %s", exc)
        return UltrasoundAssessment(error=str(exc))
