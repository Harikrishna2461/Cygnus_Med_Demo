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
    "The thigh has three levels in that list (upper_thigh, proximal_thigh_hunterian, "
    "distal_thigh_dodd) and they have no distinct visible landmark between them — judge "
    "by roughly what FRACTION of the way down the thigh (groin to knee) the probe is: "
    "top third (nearest the groin) = upper_thigh; middle third = proximal_thigh_hunterian; "
    "bottom third (nearest the knee) = distal_thigh_dodd. Do not skip straight from "
    "groin_sfj or upper_thigh to distal_thigh_dodd without considering whether the probe "
    "position actually looks closer to the middle third first.\n\n"
    "HOW TO DETERMINE leg_side (left/right) — this is easy to get backwards, reason "
    "through it explicitly every time, do not default to a naive screen-left=patient's-"
    "left mapping:\n"
    "1. First decide the patient's orientation relative to the camera: is their FRONT "
    "facing the camera, their BACK facing the camera (cue: you see their back/shoulder "
    "blades/back of the head, not their face), or a side profile?\n"
    "2. If the patient's FRONT faces the camera: this is the same as facing another "
    "person — their body is mirrored relative to the image. The leg on the LEFT side "
    "of the image is the PATIENT'S OWN RIGHT leg; the leg on the RIGHT side of the "
    "image is the PATIENT'S OWN LEFT leg.\n"
    "3. If the patient's BACK faces the camera: no mirroring. The leg on the LEFT side "
    "of the image is the patient's own LEFT leg; the leg on the RIGHT side of the "
    "image is the patient's own RIGHT leg.\n"
    "4. If you cannot tell which way the patient is facing, answer leg_side='uncertain' "
    "rather than guessing — never default to assuming the image is unmirrored.\n\n"
    "HOW TO DETERMINE surface (anterior/medial/posterior/lateral) of the leg being "
    "scanned:\n"
    "- posterior: the patient's back faces the camera (or you can otherwise see the "
    "back of the leg/back of the knee) and the probe is on the back surface of the leg.\n"
    "- anterior: the patient's front faces the camera and the probe is on the "
    "front-facing surface of the leg, toward the camera.\n"
    "- medial: the probe is on the INNER side of the leg, toward the midline / the "
    "other leg (often visible as the probe angled toward or between both legs).\n"
    "- lateral: the probe is on the OUTER side of the leg, away from the other leg.\n"
    "medial/lateral do NOT flip with facing direction (medial always means toward the "
    "body's midline); anterior/posterior DO depend on whether you're seeing the front "
    "or back of the patient's body.\n\n"
    "Work through the facing-direction and leg_side reasoning ONCE, reach a conclusion, "
    "and move on — do not re-litigate the same left/right judgment back and forth "
    "repeatedly. If after one careful pass you are genuinely torn, answer 'uncertain' "
    "rather than continuing to deliberate.\n\n"
    "It is expected and fine to answer 'uncertain' for any field you cannot actually "
    "determine from the image — a wrong specific guess is worse than an honest "
    "'uncertain'. Respond with ONLY a compact JSON object, no markdown, no prose outside "
    "the JSON, in exactly this shape:\n"
    '{"leg_side": "left"|"right"|"uncertain", '
    '"leg_level": "<one of the leg-level names above>"|"uncertain", '
    '"surface": "anterior"|"medial"|"posterior"|"lateral"|"uncertain", '
    '"confidence": "high"|"medium"|"low", '
    '"probe_visible": true|false, '
    '"visual_evidence": "<one sentence: what landmarks/cues you actually used, '
    'INCLUDING which way the patient is facing and how that determined leg_side>"}'
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
    # reasoning_effort="default" (full chain-of-thought, not the fast/no-think mode used
    # elsewhere) — confirmed necessary on real footage: the facing-direction mirroring
    # this prompt asks for is genuinely multi-step spatial reasoning, and with reasoning
    # off the same near-identical frame got mirrored correctly on some ticks and backwards
    # on others. This call is less frequent than segmentation/naming (gated on
    # WEBCAM_LOCATION_MIN_INTERVAL_SEC) so the added latency per call matters less here.
    #
    # max_tokens raised well above config.GROQ_MAX_TOKENS (3072) — confirmed necessary:
    # a real call hit 3072 tokens mid-<think>, before ever reaching the JSON answer, and
    # got silently truncated to {} (parsed as a total failure, defaulted to all-uncertain
    # via normalize() below). Full reasoning on an image genuinely needs more headroom
    # than the shorter Stage 2/3b calls.
    parsed, _raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/jpeg",
                                              reasoning_effort="default", max_tokens=10000)
    return normalize(parsed)
