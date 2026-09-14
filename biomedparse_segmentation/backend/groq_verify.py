"""
Per-blob vein verification via a Groq vision LLM.

BiomedParse's single-best-query selection (`engine._grounding_prob`) picks
whichever of 101 object-query channels has the single highest peak
confidence, then thresholds that whole channel's map. When the wrong query
wins, its thresholded map can span several unrelated blobs at once (muscle
texture, bright irregular tissue) that classical shape/darkness filters
(`engine.prob_to_vein_mask`) aren't able to tell apart from a real vein.

This module ports the exact fix already proven in
Task_4_VLM_Fascia_Vein_Detection/app.py (`evaluate_vein_mask`/`_evaluate_blob`):
ask a vision LLM, per connected-component blob, "does this actually look
like a vein?" and drop any blob it rejects. Same Groq key/model already used
elsewhere in this project (see memory note `project_api_key_locations`).
"""
import base64
import concurrent.futures
import io
import re

import cv2
import numpy as np
from PIL import Image
from groq import Groq

_GROQ_API_KEY = ""
_GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
_client = Groq(api_key=_GROQ_API_KEY)

_EVALUATOR_SYSTEM = (
    "You are a peripheral vascular sonographer reviewing a full B-mode ultrasound frame. "
    "One candidate structure is marked with a GREEN outline. "
    "Decide whether it is a real vein — which can appear anywhere: above the fascia (superficial veins), "
    "at the fascia, or deep inside the muscle compartment (deep veins like femoral or popliteal).\n\n"
    "A real vein has ALL of these:\n"
    "  - DISCRETE BOUNDARY — a clear, well-defined wall separating it from surrounding tissue\n"
    "  - DARK INTERIOR — anechoic or hypoechoic lumen (clearly darker than surrounding muscle speckle)\n"
    "  - COMPACT SHAPE — oval or round, not sprawling or irregular\n\n"
    "Say NO only if the structure is CLEARLY:\n"
    "  - Muscle tissue with internal speckle (not truly dark inside)\n"
    "  - A diffuse irregular blob with no clean boundary\n"
    "  - A horizontal band or sheet (fascia or artifact, not a vessel)\n\n"
    "When uncertain, say YES. Reply with exactly one word: YES or NO."
)


def _full_img_b64_highlighted(image_rgb: np.ndarray, blob_mask: np.ndarray) -> str:
    annotated = image_rgb.copy()
    contours, _ = cv2.findContours(blob_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 3)
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _evaluate_blob(image_rgb: np.ndarray, blob_mask: np.ndarray) -> bool:
    """Returns True if the Groq vision model confirms this blob is a real vein."""
    try:
        b64 = _full_img_b64_highlighted(image_rgb, blob_mask)
        resp = _client.chat.completions.create(
            model=_GROQ_VISION_MODEL,
            messages=[
                {"role": "system", "content": _EVALUATOR_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "The GREEN outlined structure is the candidate. Is it a vein? YES or NO. /no_think"},
                ]},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ''
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
        answer = (clean.split()[0] if clean.split() else raw).upper()
        return answer.startswith('YES')
    except Exception as e:
        print(f"[groq_verify] error: {e} — keeping blob (fail-open)")
        return True


def verify_vein_mask(vein_mask: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """
    Split the raw vein mask into individual connected-component blobs,
    verify each in parallel with the Groq vision LLM, and return a mask
    containing only confirmed veins. Fails open (keeps a blob) on any
    per-blob API error, so a transient Groq outage degrades to the
    pre-verifier behaviour rather than dropping every detection.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(vein_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return vein_mask

    h, w = vein_mask.shape
    min_px = max(50, int(0.0003 * h * w))
    blob_ids = [i for i in range(1, n) if int((labels == i).sum()) >= min_px]
    if not blob_ids:
        return np.zeros_like(vein_mask)

    def _check(i):
        return i, _evaluate_blob(image_rgb, (labels == i))

    verified = np.zeros_like(vein_mask)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(blob_ids))) as ex:
        for blob_id, is_vein in ex.map(_check, blob_ids):
            if is_vein:
                verified[labels == blob_id] = 1
    return verified
