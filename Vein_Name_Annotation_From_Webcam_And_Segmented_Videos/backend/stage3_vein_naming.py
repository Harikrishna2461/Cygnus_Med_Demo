"""
Stage 3b: combine the N-classed blobs (Stage 2) and the probe location (Stage 3a) with
the anatomy reference text to assign real medical vein names. VLM-only decision — Python
never branches on leg_level/n_class to pick a name.
"""
import base64

import cv2

import anatomy_knowledge
import groq_client


def _system_prompt(anatomy_text: str) -> str:
    return (
        "You assign real medical vein names to already-depth-classified vein "
        "cross-sections in a leg ultrasound frame, using where the probe is on the "
        "patient's leg (from a paired webcam frame) and the anatomy reference below. "
        "There is no lookup table for this — reason from the location and the anatomy "
        "text each time.\n\n"
        + anatomy_text +
        "\n\nEach blob already has an N-class (N1=deep, N2=saphenous trunk, "
        "N3=superficial tributary) from a separate depth analysis — trust it, do not "
        "re-derive it. Your job is only to pick the specific vein name consistent with "
        "that N-class AND the probe location.\n\n"
        "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
        "JSON: "
        '{"<blob_id>": {"vein_name": "<e.g. GSV, SSV, CFV, FV, PV, AASV, PASV, '
        'Tributary, Perforator>", "reasoning": "<one sentence>"}, ...}'
    )


def build_naming_prompt(blobs: list[dict], location: dict,
                         anatomy_text: str = anatomy_knowledge.ANATOMY_REFERENCE_TEXT) -> tuple[str, str]:
    """Pure function — testable with hand-built blob/location dicts, no image/model/network.
    blobs: [{"blob_id": int, "n_class": "N1"|"N2"|"N3", "centroid": [x, y]}, ...]
    location: Stage-3a output dict (see stage3_webcam_location.normalize)."""
    system = _system_prompt(anatomy_text)
    loc_desc = (
        f"Probe location (from webcam, confidence={location.get('confidence', 'uncertain')}): "
        f"leg_side={location.get('leg_side', 'uncertain')}, "
        f"leg_level={location.get('leg_level', 'uncertain')}, "
        f"surface={location.get('surface', 'uncertain')}. "
        f"Visual evidence: {location.get('visual_evidence', 'none')}"
    )
    blob_lines = [
        f"Blob {b['blob_id']}: n_class={b.get('n_class', '?')}, centroid={b.get('centroid')}"
        for b in blobs
    ]
    user = loc_desc + "\n\n" + "\n".join(blob_lines) + "\n\nName each blob."
    return system, user


def name_veins(blobs: list[dict], location: dict, annotated_ultrasound_frame_bgr=None) -> dict:
    """Returns {blob_id(int): {"vein_name": str, "reasoning": str}} — only for blobs the
    VLM actually answered with a non-empty vein_name."""
    system, user = build_naming_prompt(blobs, location)
    img_b64 = None
    if annotated_ultrasound_frame_bgr is not None:
        _, buf = cv2.imencode(".png", annotated_ultrasound_frame_bgr)
        img_b64 = base64.b64encode(buf).decode()
    parsed, _raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/png")

    out = {}
    for key, val in parsed.items():
        try:
            bid = int(key)
        except ValueError:
            continue
        if isinstance(val, dict) and val.get("vein_name"):
            out[bid] = {"vein_name": val["vein_name"], "reasoning": val.get("reasoning", "")}
    return out
