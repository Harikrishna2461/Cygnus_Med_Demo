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


MAX_NAMING_RETRIES = 2  # extra attempts beyond the first, asking again only for blobs
                         # the model skipped — see name_veins()


def _name_veins_once(blobs: list[dict], location: dict, annotated_ultrasound_frame_bgr,
                      extra_instruction: str = "") -> dict:
    system, user = build_naming_prompt(blobs, location)
    if extra_instruction:
        user = user + "\n\n" + extra_instruction
    img_b64 = None
    if annotated_ultrasound_frame_bgr is not None:
        _, buf = cv2.imencode(".png", annotated_ultrasound_frame_bgr)
        img_b64 = base64.b64encode(buf).decode()
    parsed, _raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/png",
                                              label="stage3b_naming")

    out = {}
    for key, val in parsed.items():
        try:
            bid = int(key)
        except ValueError:
            continue
        if isinstance(val, dict) and val.get("vein_name"):
            out[bid] = {"vein_name": val["vein_name"], "reasoning": val.get("reasoning", "")}
    return out


def name_veins(blobs: list[dict], location: dict, annotated_ultrasound_frame_bgr=None) -> dict:
    """Returns {blob_id(int): {"vein_name": str, "reasoning": str}}.

    Every blob passed in gets a real, VLM-decided name if at all possible — the caller
    (pipeline.py) requires this so the final video never shows a bare N-class instead of
    a name. A single call sometimes skips a blob_id in its JSON response even though
    build_naming_prompt() asked about all of them; rather than filling the gap with a
    Python-decided default (which would reintroduce exactly the hardcoded-lookup pattern
    this project exists to avoid), this retries up to MAX_NAMING_RETRIES times, each time
    asking ONLY about the still-missing blob(s) with an explicit "you must answer for all
    of these" instruction. Genuinely unnamed blobs after all retries are rare (network/
    parsing failure, not the model choosing to skip) and are left absent from the
    returned dict — pipeline.py's hold logic will retry again on the next naming tick.
    """
    remaining = {b["blob_id"]: b for b in blobs}
    out: dict = {}
    attempt = 0
    while remaining and attempt <= MAX_NAMING_RETRIES:
        extra = ""
        if attempt > 0:
            extra = (
                f"Your previous answer did not include a vein_name for blob(s) "
                f"{sorted(remaining.keys())}. You MUST provide a vein_name for every one "
                f"of those blob_ids now, even if uncertain — do not omit any."
            )
        result = _name_veins_once(list(remaining.values()), location,
                                   annotated_ultrasound_frame_bgr, extra_instruction=extra)
        out.update(result)
        for bid in list(remaining.keys()):
            if bid in result:
                del remaining[bid]
        attempt += 1
    return out