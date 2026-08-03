"""
Stage 2: classify each vein blob as N1 (deep) / N2 (saphenous trunk) / N3 (superficial
tributary). The VLM makes the call; Python only computes and hands over the geometric
measurement (blob position relative to the two fascia lines) as supporting text — nothing
here branches on the answer. See project plan for the "hybrid" design rationale.
"""
import base64

import cv2
import numpy as np

import anatomy_knowledge
import groq_client
import renderer

_FASCIAL_DEPTH_TEXT = anatomy_knowledge.ANATOMY_REFERENCE_TEXT.split("LEG LEVELS")[0].strip()

SYSTEM_PROMPT = (
    "You read annotated leg-ultrasound frames. A YELLOW line marks the superficial edge of "
    "the saphenous fascia; an ORANGE line marks the deep edge (at the muscle fascia). "
    "Numbered contours mark candidate vein lumens detected by an automated segmentation "
    "model — the model sometimes fires on things that are NOT real veins, for example "
    "letters/words from an on-screen watermark or logo (a closed letter shape like 'e', "
    "'o', 'g', or 'P' can look like a small dark oval), a UI icon, or other non-tissue "
    "graphics. Real ultrasound tissue has a grainy speckle texture; text/logos/watermarks "
    "have flat colour and sharp typographic edges with no speckle around them, and often "
    "sit in a visually distinct strip or overlay rather than embedded in the grayscale scan "
    "image.\n\n"
    "For EACH numbered blob, first judge is_valid_vein: does this actually sit inside real "
    "speckled ultrasound tissue, or is it part of text/a watermark/a logo/UI graphics? "
    "If invalid, you do not need to classify its depth — set n_class to null.\n\n"
    "If valid, classify its depth relative to the fascial compartment:\n"
    + _FASCIAL_DEPTH_TEXT +
    "\n\nFor each blob you are also given a precomputed geometric measurement (from the "
    "segmentation model itself, not a guess) describing its position relative to both "
    "fascia lines at its own column — use this alongside the image for the depth call, but "
    "it says nothing about whether the blob is a real vein in the first place, judge that "
    "from the image.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON, in "
    "exactly this shape:\n"
    '{"<blob_id>": {"is_valid_vein": true|false, "n_class": "N1"|"N2"|"N3"|null, '
    '"reasoning": "<one sentence>"}, ...}'
)


def _geometry_hint(blob, fascia) -> str:
    cx, cy = blob.centroid
    col = int(round(cx))
    col = max(0, min(col, len(fascia.sup_row_at_col) - 1))
    sup, deep = fascia.sup_row_at_col[col], fascia.deep_row_at_col[col]
    if np.isnan(sup) or np.isnan(deep):
        return f"Blob {blob.blob_id}: fascia lines not reliably detected at this column — judge from the image alone."
    d_sup = cy - sup     # >0 => centroid below the superficial line
    d_deep = deep - cy   # >0 => centroid above the deep line
    sup_desc = f"{abs(d_sup):.0f}px {'below' if d_sup >= 0 else 'above'} the superficial (yellow) line"
    deep_desc = f"{abs(d_deep):.0f}px {'above' if d_deep >= 0 else 'below'} the deep (orange) line"
    return f"Blob {blob.blob_id}: centroid is {sup_desc}, and {deep_desc}."


def build_prompt(blobs: list, fascia) -> str:
    """Pure function — testable with hand-built VeinBlob/FasciaBoundary instances,
    no image/model/network needed."""
    header = f"There are {len(blobs)} numbered vein blob(s) in this frame. Classify each one.\n\n"
    return header + "\n".join(_geometry_hint(b, fascia) for b in blobs)


def classify_blobs(frame_bgr: np.ndarray, blobs: list, fascia) -> None:
    """Mutates blobs in place, filling n_class/n_class_reasoning. No-op if blobs is empty."""
    if not blobs:
        return
    annotated = renderer.draw_intermediate_frame(frame_bgr, blobs, fascia)  # numbers only, n_class unset
    _, buf = cv2.imencode(".png", annotated)
    img_b64 = base64.b64encode(buf).decode()
    user_text = build_prompt(blobs, fascia)
    parsed, _raw = groq_client.call_vlm_json(SYSTEM_PROMPT, user_text, image_b64=img_b64)

    by_id = {b.blob_id: b for b in blobs}
    for key, val in parsed.items():
        try:
            bid = int(key)
        except ValueError:
            continue
        blob = by_id.get(bid)
        if blob is None or not isinstance(val, dict):
            continue
        blob.n_class_reasoning = val.get("reasoning")
        if val.get("is_valid_vein") is False:
            blob.is_valid = False
            blob.n_class = None
            continue
        n_class = val.get("n_class")
        if n_class in ("N1", "N2", "N3"):
            blob.n_class = n_class
