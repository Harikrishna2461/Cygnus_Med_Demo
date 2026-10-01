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
import vlm_client
import renderer

_FASCIAL_DEPTH_TEXT = anatomy_knowledge.ANATOMY_REFERENCE_TEXT.split("LEG LEVELS")[0].strip()

SYSTEM_PROMPT = (
    "You read annotated leg-ultrasound frames. Two lines mark the fascial compartment that "
    "contains the saphenous trunk: a YELLOW line is its superficial edge, an ORANGE line is "
    "its deep edge (the muscle fascia). Together they form one band -- the fascial layer. "
    "Numbered contours mark candidate vein lumens found by an automated segmentation model, "
    "which sometimes fires on things that are NOT real veins: letters/words from an "
    "on-screen watermark or logo (a closed letter shape like 'e', 'o', 'g', or 'P' can look "
    "like a small dark oval), a UI icon, or other non-tissue graphics. Real ultrasound "
    "tissue has a grainy speckle texture; text/logos/watermarks have flat colour and sharp "
    "typographic edges with no speckle around them.\n\n"
    "For EACH numbered blob, first judge is_valid_vein: does this actually sit inside real "
    "speckled ultrasound tissue, or is it text/a watermark/a logo/UI graphics? If invalid, "
    "set n_class to null.\n\n"
    "If valid, classify its depth using EXACTLY this rule, nothing else:\n"
    "- N2 = the blob is WITHIN the fascial compartment (between the yellow and orange "
    "line), INCLUDING a blob that overlaps or touches either line. This is the default "
    "for any blob that is not clearly and entirely on one side of the band.\n"
    "- N3 = the blob is CLEARLY above the yellow line, entirely outside and above the "
    "fascial compartment.\n"
    "- N1 = the blob is CLEARLY below the orange line, entirely outside and below the "
    "fascial compartment.\n\n"
    "You are given a precomputed SIGNED pixel distance from each blob's centre to each "
    "line -- use its sign, not a visual impression of where the contour's edge appears to "
    "touch a line:\n"
    "- Centre below the orange line (deep, negative d_deep) -> N1.\n"
    "- Centre above the yellow line (superficial, negative d_sup) -> N3.\n"
    "- Anything else -- centre sits between the two lines, on either side of the exact "
    "midline, however close to either line, even 1px from it -- is N2. A blob whose "
    "contour visually grazes or slightly crosses a line while its centre sign still says "
    "between the lines is N2, not N1 or N3: the segmentation contour is never "
    "pixel-perfect, the centre sign is the real position.\n\n"
    "Thinking mode is OFF for this call: do not reason at length. For each blob, look at "
    "its two signed distances, apply the rule above, and answer -- one line of internal "
    "reasoning per blob at most, then move straight to the next blob. Do not re-check a "
    "blob you already decided.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON, in "
    "exactly this shape:\n"
    '{"<blob_id>": {"is_valid_vein": true|false, "n_class": "N1"|"N2"|"N3"|null, '
    "\"reasoning\": \"<one short phrase citing the sign, e.g. 'd_deep +14px -> N2'>\"}, ...}"
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


MAX_TOKENS = 16384  # ceiling for the truncation retry (see classify_blobs); equals
# config.VLM_MAX_TOKENS_CAP. The FIRST attempt requests a smaller, blob-count-scaled
# max_tokens instead (see _first_attempt_max_tokens): 1-2 blob frames finish in a few
# thousand tokens, and a lower ceiling stops a runaway reasoning trace from occupying a
# llama-server decode slot for minutes. The full ceiling stays available for the rare
# truncation retry.


def _first_attempt_max_tokens(n_blobs: int) -> int:
    """Scales with blob count -- more blobs means more reasoning + JSON, confirmed on real
    footage (a 4-blob frame needed close to the full ceiling before the retry-budget
    prompt fix; 1-2 blob frames comfortably finished in a few thousand tokens). Generous
    margin still included per blob -- an under-estimate only caps the call (worst case
    triggering the truncation retry), so erring larger is cheap; erring smaller only costs
    a rare extra retry."""
    return min(MAX_TOKENS, 2500 + 3000 * max(n_blobs, 1))


def _retry_max_tokens(n_blobs: int) -> int:
    """Confirmed real cost waste this fixes: the retry used to always jump straight to
    the full 16384-token ceiling regardless of blob count -- fine for a genuinely busy
    4-blob frame that needs it, but wasteful for a 1-blob truncation (real observed case:
    first attempt at 5500 tokens truncated, retry then reserved/paid for the FULL 16384
    ceiling, an ~11000-token jump for a single blob). The retry also carries
    _RETRY_SUFFIX, which explicitly asks for a terser answer, so it rarely needs MORE
    room than a generous multiple of the first attempt's own budget -- real evidence
    backs this too (retries have consistently finished in far fewer tokens than the
    truncated first attempt, e.g. a 32s/56k-char truncated attempt recovering in a 4.7s/
    short retry). 2x the first-attempt budget, capped at the true ceiling, keeps genuinely
    complex multi-blob frames at (or near) the full ceiling while cutting the wasted
    headroom for the common low-blob-count case."""
    return min(MAX_TOKENS, _first_attempt_max_tokens(n_blobs) * 2)

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

    Runs with thinking OFF (config.VLM_FORCE_NO_THINKING, project-wide -- see
    Qwen_Local_VLM_Evaluation_Report.docx: thinking off scored 100% on this exact task in
    1.5s/call, thinking on scored the same 100% but took ~17s/call and no better accuracy).
    SYSTEM_PROMPT is written for that mode: the N1/N2/N3 rule is a strict sign check with a
    single explicit default (N2, since "within or touching the compartment" covers most real
    ambiguity), not a multi-step judgment call -- the earlier long, repetitive prompt in this
    file's history was written to correct a reasoning model talking itself out of the sign
    rule; a non-reasoning model just needs the rule stated once, clearly.

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
    parsed, raw = vlm_client.call_vlm_json(
        SYSTEM_PROMPT, user_text, image_b64=img_b64,
        reasoning_effort="none", max_tokens=_first_attempt_max_tokens(len(blobs)),  # thinking forced off project-wide anyway (config.VLM_FORCE_NO_THINKING); "none" here just makes the call site honest
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
        parsed, raw = vlm_client.call_vlm_json(
            SYSTEM_PROMPT, user_text + suffix, image_b64=img_b64,
            reasoning_effort="none", max_tokens=_retry_max_tokens(len(blobs)),
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
