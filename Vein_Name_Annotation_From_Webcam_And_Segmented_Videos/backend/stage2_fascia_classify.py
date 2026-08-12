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
    "\n\nDECISION RULE — apply this explicitly, in order, for every valid blob, using the "
    "geometric measurement below alongside the image:\n"
    "- Is the blob BELOW the ORANGE (deep) line? → N1. This includes a blob sitting ON or "
    "straddling the orange line itself — do not call anything N3 just because it touches a "
    "fascia line; touching/straddling the DEEP line still means N1, not N3.\n"
    "- Is the blob BETWEEN the two lines — below/on the yellow line AND above/on the orange "
    "line? → N2. This is the most commonly under-used class: a vein sitting close to, "
    "touching, or even flattened against EITHER fascia line, as long as it is still within "
    "the space bounded by the two lines, is N2 (the saphenous compartment), never N3. Do "
    "NOT default to N3 just because a blob looks small, faint, or close to a boundary line — "
    "closeness to a line is not superficiality; only genuinely being ABOVE both lines is.\n"
    "- Is the blob ABOVE the YELLOW (superficial) line, with no part of it inside the "
    "compartment? → N3. Reserve N3 strictly for blobs clearly sitting in the subcutaneous "
    "fat above the yellow line — a blob merely near the yellow line but still on/below it is "
    "N2, not N3.\n"
    "A confirmed, repeated real-world failure this system exists to avoid: blobs sitting "
    "exactly at or just inside a fascia line getting defaulted to N3 (superficial) when the "
    "geometry clearly places them within or below the compartment. Read the precomputed "
    "pixel-distance measurement carefully — a small \"Npx above/below\" distance still has a "
    "definite sign (above vs. below); use that sign, do not round it away to 'basically on "
    "the line, so probably superficial'.\n\n"
    "For each blob you are also given a precomputed geometric measurement (from the "
    "segmentation model itself, not a guess) describing its position relative to both "
    "fascia lines at its own column — use this alongside the image for the depth call, but "
    "it says nothing about whether the blob is a real vein in the first place, judge that "
    "from the image.\n\n"
    "STRICT reasoning budget when there are multiple blobs: work through each blob in 2-3 "
    "short sentences (validity, then which side of which line, therefore which class), "
    "reach a conclusion, and move on to the next blob immediately — do not re-litigate a "
    "blob you've already decided, and do not re-read the whole frame from scratch for "
    "each one. The moment you have an answer for every blob, stop reasoning and output "
    "the JSON — spending the whole budget circling back on the same 1-2 blobs repeatedly "
    "is a failure mode that leaves every blob unclassified, not a sign of thoroughness.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON, in "
    "exactly this shape:\n"
    '{"<blob_id>": {"is_valid_vein": true|false, "n_class": "N1"|"N2"|"N3"|null, '
    '"reasoning": "<one sentence, cite the actual pixel measurement and which side of which '
    'line it puts the blob on>"}, ...}'
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


MAX_TOKENS = 16384  # Groq's hard ceiling for this model -- confirmed elsewhere in this
# project (stage3_webcam_location.py) that requesting above this raises a 400. Used as the
# retry ceiling (see classify_blobs) -- the FIRST attempt requests a smaller, blob-count-
# scaled max_tokens instead (see _first_attempt_max_tokens), since groq_client.py's
# _OtpmLimiter reserves budget equal to whatever max_tokens a call actually requests (the
# only way to GUARANTEE it can never exceed that reservation, since Groq enforces on real
# generated tokens). Always requesting the full 16384 ceiling would reserve far more
# budget than almost any single-frame call actually needs (confirmed on real footage:
# 1-2 blob frames use a few thousand tokens, not 16k) and would serialize Stage 2 far more
# than necessary against the 28000-token rolling budget. Scaling down the common case
# while keeping the full ceiling available for the rare truncation retry gets both real
# concurrency AND a mathematical no-429 guarantee at the same time.


def _first_attempt_max_tokens(n_blobs: int) -> int:
    """Scales with blob count -- more blobs means more reasoning + JSON, confirmed on real
    footage (a 4-blob frame needed close to the full ceiling before the retry-budget
    prompt fix; 1-2 blob frames comfortably finished in a few thousand tokens). Generous
    margin still included per blob since this number IS the OTPM reservation -- an
    under-estimate here doesn't cause a 429 (the call is simply capped, worst case
    triggering the truncation retry, not a rate-limit error), so erring larger is cheap;
    erring smaller only costs a rare extra retry, never a real rate-limit violation."""
    return min(MAX_TOKENS, 2500 + 3000 * max(n_blobs, 1))

_RETRY_SUFFIX = (
    "\n\nYour previous attempt at this exact frame ran out of space mid-reasoning without "
    "ever reaching a JSON answer for every blob -- you were spending too much reasoning "
    "per blob. This time: for EACH blob, reason in at most 2 short sentences (which side "
    "of which line, therefore which class), then move to the next blob. Do not re-litigate "
    "a blob once you've decided it. Output the JSON for ALL blobs the instant you have "
    "every answer."
)


def _looks_truncated(parsed: dict, raw: str) -> bool:
    """True when the call produced no usable JSON at all and the raw text is long -- i.e.
    the model spent its whole budget mid-reasoning instead of a genuinely short/empty
    response. Confirmed real failure mode on a busy 4-blob frame: 28k+ chars of reasoning,
    zero blobs ever got a JSON answer, all silently left unclassified with no signal that
    anything had gone wrong."""
    return not parsed and len(raw) > 2000


def _missing_blob_ids(parsed: dict, blobs: list) -> list:
    """Confirmed a SEPARATE real failure mode from _looks_truncated: the model can return
    syntactically valid, non-empty JSON that simply never mentions one or more blob_ids at
    all (as opposed to the whole response being empty/truncated) -- e.g. a busy multi-blob
    frame where the model's own JSON is well-formed but incomplete. _looks_truncated's
    `not parsed` check is blind to this exact case (parsed is non-empty, so it looks
    "fine"), which is exactly what silently left every blob on a real frame unclassified
    with no error and no retry ever firing. This checks actual per-blob coverage instead
    of just "did we get JSON at all"."""
    present = set()
    for key in parsed.keys():
        try:
            present.add(int(key))
        except ValueError:
            continue
    return [b.blob_id for b in blobs if b.blob_id not in present]


_MISSING_RETRY_SUFFIX = (
    "\n\nYour previous attempt returned JSON but left out one or more blob numbers "
    "entirely -- EVERY numbered blob shown in the image MUST appear as a key in your "
    "JSON output, with no exceptions. If you are genuinely unsure about a blob's class, "
    "still include it with your best judgment (or is_valid_vein:false if it's not a real "
    "vein) rather than omitting it -- an omitted blob_id is treated as a total failure for "
    "that blob, which is worse than an uncertain-but-present answer."
)


def classify_blobs(frame_bgr: np.ndarray, blobs: list, fascia) -> None:
    """Mutates blobs in place, filling n_class/n_class_reasoning. No-op if blobs is empty.

    reasoning_effort='default' (full chain-of-thought) -- this call previously ran on the
    project-wide config default of 'none', which was never deliberately chosen for this
    stage, just inherited. Confirmed real-world complaint: veins sitting within or very
    close to the fascial compartment were being defaulted to N3 (superficial) instead of
    N2/N1. This is exactly the class of judgment ('none' mode makes seen fail on this
    project every other time it was tried: leg_side mirroring, knee-boundary distance
    calls) -- comparing a blob's position against two geometric lines and correctly
    reading a small signed pixel distance needs real reasoning, not a single-shot
    pattern-match.

    max_tokens raised to the model's hard ceiling (16384), and a single truncation retry
    added -- confirmed necessary on real footage: a busy 4-blob frame produced 28k+ chars
    of reasoning and STILL never reached JSON for any blob at the previous 8192 cap,
    leaving every blob silently unclassified (worse than the original N3-default bug,
    since that at least produced an answer). This retry is scoped narrowly (truncation
    only, not a general quality re-check/vote -- that pattern was removed elsewhere this
    session for cost reasons) and only fires on the specific failure it targets."""
    if not blobs:
        return
    annotated = renderer.draw_intermediate_frame(frame_bgr, blobs, fascia)  # numbers only, n_class unset
    _, buf = cv2.imencode(".png", annotated)
    img_b64 = base64.b64encode(buf).decode()
    user_text = build_prompt(blobs, fascia)
    parsed, raw = groq_client.call_vlm_json(
        SYSTEM_PROMPT, user_text, image_b64=img_b64,
        reasoning_effort="default", max_tokens=_first_attempt_max_tokens(len(blobs)),
        label="stage2_nclass",
    )
    truncated = _looks_truncated(parsed, raw)
    missing = _missing_blob_ids(parsed, blobs)
    if truncated or missing:
        # Same retry call covers both failure modes (see _looks_truncated vs.
        # _missing_blob_ids docstrings) -- pick the more specific suffix when the JSON
        # was well-formed but incomplete, since that's a more actionable correction than
        # the generic "you ran out of space" framing.
        suffix = _RETRY_SUFFIX if truncated else _MISSING_RETRY_SUFFIX
        print(f"[stage2] retrying: truncated={truncated}, missing_blob_ids={missing}")
        parsed, raw = groq_client.call_vlm_json(
            SYSTEM_PROMPT, user_text + suffix, image_b64=img_b64,
            reasoning_effort="default", max_tokens=MAX_TOKENS,
            label="stage2_nclass_retry",
        )
        still_missing = _missing_blob_ids(parsed, blobs)
        if still_missing:
            # Surface this loudly rather than silently rendering a blank label -- this is
            # exactly the class of failure ("looked fine, just quietly missing answers")
            # that went unnoticed before _missing_blob_ids existed.
            print(f"[stage2] WARNING: blob_id(s) {still_missing} still missing after "
                  f"retry -- will render/name as unclassified for this tick.")

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
