"""Stage 3a: look at one webcam frame and describe where the probe is on the leg."""
import base64

import cv2

import anatomy_knowledge
import groq_client

SYSTEM_PROMPT = (
    "You look at a single frame from a webcam video of a clinician performing a leg "
    "venous ultrasound exam. Your job is ONLY to describe where the ultrasound probe is "
    "touching the patient's leg — you are not diagnosing anything.\n\n"
    "Use these leg-level names when you can tell: "
    + ", ".join(l for l in anatomy_knowledge.LEG_LEVELS if l != "uncertain") + ".\n"
    "Landmarks: the groin crease (top of thigh), the knee, the medial and lateral "
    "malleoli (ankle bones), the popliteal fossa (back of the knee).\n\n"
    "It is expected and fine to answer 'uncertain' for any field you cannot actually "
    "determine from the image — a wrong specific guess is worse than an honest "
    "'uncertain'. Respond with ONLY a compact JSON object, no markdown, no prose outside "
    "the JSON, in exactly this shape:\n"
    '{"leg_side": "left"|"right"|"uncertain", '
    '"leg_level": "<one of the leg-level names above>"|"uncertain", '
    '"surface": "anterior"|"medial"|"posterior"|"lateral"|"uncertain", '
    '"confidence": "high"|"medium"|"low", '
    '"probe_visible": true|false, '
    '"visual_evidence": "<one sentence: what landmarks/cues you actually used>"}'
)

USER_PROMPT = (
    "This frame is from the webcam video, at the timestamp matching an ultrasound frame "
    "we need to interpret. Identify the probe location on the leg."
)


def build_prompt() -> tuple[str, str]:
    """Pure — no image/model/network. Kept as a function (not just module constants) so
    prompt construction stays consistent with the other two stages."""
    return SYSTEM_PROMPT, USER_PROMPT


def normalize(parsed: dict) -> dict:
    """Fills safe 'uncertain'/'low' defaults so downstream code never KeyErrors on a
    malformed or empty VLM response — this is the system-boundary validation point."""
    return {
        "leg_side": parsed.get("leg_side") or "uncertain",
        "leg_level": parsed.get("leg_level") or "uncertain",
        "surface": parsed.get("surface") or "uncertain",
        "confidence": parsed.get("confidence") or "low",
        "probe_visible": bool(parsed.get("probe_visible", False)),
        "visual_evidence": parsed.get("visual_evidence") or "",
    }


def read_location(frame_bgr) -> dict:
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    system, user = build_prompt()
    parsed, _raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/jpeg")
    return normalize(parsed)
