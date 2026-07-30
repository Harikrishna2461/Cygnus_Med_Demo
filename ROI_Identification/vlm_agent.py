"""
VLM-based ROI detector using Groq vision API.
Sends sample frames to the model and asks it to identify the ultrasound
scan area bounding box in pixel coordinates.

Coordinates are always normalised to original frame dimensions:
  - frames are downscaled before sending (to control token cost)
  - returned coordinates are scaled back to original size
"""

import base64
import json
import re
import cv2
import numpy as np
from groq import Groq

GROQ_API_KEY = ""
# Vision-capable model (qwen3.6-27b accepts image_url inputs on Groq).
GROQ_VISION_MODELS = [
    "qwen/qwen3.6-27b",
]
GROQ_MODEL = GROQ_VISION_MODELS[0]
MAX_SEND_DIM   = 720   # longest side when downscaling for the API call

_client: Groq | None = None


def _client_get() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Frame prep ────────────────────────────────────────────────────────────────

def _encode_frame(frame: np.ndarray, max_dim: int = MAX_SEND_DIM) -> tuple[str, float]:
    """
    Resize frame so longest side <= max_dim.
    Returns (base64_jpeg_string, scale_factor) where scale_factor maps
    sent-image coords -> original coords (multiply VLM output by 1/scale).
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode(), scale


# ── Prompts ───────────────────────────────────────────────────────────────────

def _prompt_view_type(img_w: int, img_h: int) -> str:
    return (
        f"This is a cropped ultrasound scan frame ({img_w}x{img_h} pixels, "
        "no UI chrome — only scan content). "
        "Classify the probe orientation:\n"
        "- TRANSVERSE: vessels appear as circular or roughly round dark cross-sections "
        "(probe cut perpendicular to the vessel axis).\n"
        "- LONGITUDINAL: vessels appear as elongated, tube-like dark channels "
        "running horizontally across most of the frame width "
        "(probe parallel to the vessel axis).\n"
        'Reply with ONLY this JSON and nothing else: {"view_type": "TRANSVERSE"} '
        'or {"view_type": "LONGITUDINAL"}'
    )


def _prompt_detect(img_w: int, img_h: int) -> str:
    return (
        f"This is a frame from a medical ultrasound machine screen recording "
        f"({img_w}x{img_h} pixels, origin at top-left corner). "
        "Your task: identify the bounding box of ONLY the ultrasound scan area — "
        "the region showing actual tissue anatomy as a grayscale speckle pattern. "
        "Exclude everything that is UI chrome: parameter panels, depth/gain rulers, "
        "machine logos, icons, menus, coloured overlay bars. "
        "Reply with ONLY this JSON and nothing else: "
        '{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}'
    )


def _prompt_verify(img_w: int, img_h: int, hint: tuple) -> str:
    x1, y1, x2, y2 = hint
    return (
        f"This is a frame from a medical ultrasound machine screen recording "
        f"({img_w}x{img_h} pixels, origin at top-left). "
        f"A proposed crop box is x1={x1}, y1={y1}, x2={x2}, y2={y2}. "
        "Does this box correctly isolate ONLY the ultrasound scan content "
        "(grayscale speckle pattern) while excluding all UI chrome? "
        "If yes, return the same box. If not, return the corrected box. "
        "Reply with ONLY this JSON and nothing else: "
        '{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}'
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse(text: str, max_w: int, max_h: int) -> tuple | None:
    """Extract (x1, y1, x2, y2) from VLM text, validate against frame bounds."""
    # Attempt JSON extraction via several patterns
    candidates = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
    for c in reversed(candidates):          # prefer last / most complete
        try:
            d = json.loads(c)
            x1 = int(d["x1"]); y1 = int(d["y1"])
            x2 = int(d["x2"]); y2 = int(d["y2"])
            x1 = max(0, min(x1, max_w)); x2 = max(0, min(x2, max_w))
            y1 = max(0, min(y1, max_h)); y2 = max(0, min(y2, max_h))
            if x2 > x1 + 10 and y2 > y1 + 10:
                return (x1, y1, x2, y2)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    # Fallback: grab first 4 integers from text
    nums = re.findall(r'\b(\d+)\b', text)
    if len(nums) >= 4:
        vals = [int(n) for n in nums[:4]]
        x1, y1, x2, y2 = vals
        if 0 <= x1 < x2 <= max_w and 0 <= y1 < y2 <= max_h:
            return (x1, y1, x2, y2)
    return None


# ── VLM call with model fallback ─────────────────────────────────────────────

_SYSTEM = "You are a medical image analysis AI. Output ONLY raw JSON — no markdown, no explanation."


def _call_vlm_text(
    client: Groq,
    b64: str,
    prompt: str,
    max_tokens: int = 150,
) -> str | None:
    """Try each model in GROQ_VISION_MODELS until one succeeds. Returns raw text."""
    global GROQ_MODEL
    for model in GROQ_VISION_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort="none",
            )
            GROQ_MODEL = model
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[VLM] Model {model} failed: {e}")
    return None


def _call_vlm(
    client: Groq,
    b64: str,
    prompt: str,
    max_w: int,
    max_h: int,
) -> tuple | None:
    """Try each model in GROQ_VISION_MODELS until one succeeds."""
    text = _call_vlm_text(client, b64, prompt, max_tokens=150)
    if text is None:
        return None
    return _parse(text, max_w, max_h)


# ── Public API ────────────────────────────────────────────────────────────────

def query_roi(
    frames: list[np.ndarray],
    hint_box: tuple | None = None,
) -> tuple | None:
    """
    Query the VLM on up to 3 frames to detect the scan-area ROI.
    If hint_box is provided, the VLM is asked to verify/correct it.
    Returns consensus (x1, y1, x2, y2) in original frame coordinates, or None.
    """
    client = _client_get()
    results: list[tuple] = []

    for frame in frames[:3]:
        orig_h, orig_w = frame.shape[:2]
        b64, scale = _encode_frame(frame)
        sent_w = int(orig_w * scale)
        sent_h = int(orig_h * scale)

        # Scale hint to sent-image coordinates for the prompt
        scaled_hint = None
        if hint_box:
            scaled_hint = tuple(int(c * scale) for c in hint_box)

        prompt = _prompt_verify(sent_w, sent_h, scaled_hint) if scaled_hint \
            else _prompt_detect(sent_w, sent_h)

        bbox_sent = _call_vlm(client, b64, prompt, sent_w, sent_h)
        if bbox_sent:
            inv = 1.0 / scale
            results.append((
                int(bbox_sent[0] * inv),
                int(bbox_sent[1] * inv),
                int(bbox_sent[2] * inv),
                int(bbox_sent[3] * inv),
            ))

    if not results:
        return None

    # Consensus across frames: coordinate-wise median
    return (
        int(np.median([r[0] for r in results])),
        int(np.median([r[1] for r in results])),
        int(np.median([r[2] for r in results])),
        int(np.median([r[3] for r in results])),
    )


def classify_view(frames: list[np.ndarray]) -> str:
    """
    Classify ultrasound probe orientation using VLM on already-cropped scan frames.
    Returns 'TRANSVERSE', 'LONGITUDINAL', or 'UNKNOWN'.
    """
    client = _client_get()
    votes: list[str] = []

    for frame in frames[:3]:
        orig_h, orig_w = frame.shape[:2]
        b64, scale = _encode_frame(frame)
        sent_w = int(orig_w * scale)
        sent_h = int(orig_h * scale)
        prompt = _prompt_view_type(sent_w, sent_h)

        text = _call_vlm_text(client, b64, prompt, max_tokens=60)
        if not text:
            continue

        label = None
        for c in reversed(re.findall(r'\{[^{}]*\}', text, re.DOTALL)):
            try:
                d = json.loads(c)
                vt = d.get("view_type", "").upper()
                if vt in ("TRANSVERSE", "LONGITUDINAL"):
                    label = vt
                    break
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

        if label is None:
            tu = text.upper()
            if "TRANSVERSE" in tu:
                label = "TRANSVERSE"
            elif "LONGITUDINAL" in tu:
                label = "LONGITUDINAL"

        if label:
            votes.append(label)
            print(f"[VLM view] frame vote: {label}")

    if not votes:
        return "UNKNOWN"

    t = votes.count("TRANSVERSE")
    l = votes.count("LONGITUDINAL")
    result = "TRANSVERSE" if t >= l else "LONGITUDINAL"
    print(f"[VLM view] final: {result} ({t}T / {l}L)")
    return result
