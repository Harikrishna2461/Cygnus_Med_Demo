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
    "\n\nDECISION RULE — apply this explicitly, in order, for every valid blob. The "
    "precomputed geometric measurement gives you a SIGNED distance to each line — that "
    "sign is the primary signal, always defer to it over a vague visual impression of "
    "'this looks close to a line so it's probably that class':\n"
    "- d_deep (distance to the deep/orange line) POSITIVE means the blob's centroid is "
    "still ABOVE the deep line, i.e. still inside the compartment. d_deep NEGATIVE means "
    "the centroid has crossed BELOW the deep line, into deep tissue.\n"
    "- d_sup (distance to the superficial/yellow line) POSITIVE means the centroid is "
    "BELOW the yellow line, i.e. still inside or below the compartment. d_sup NEGATIVE "
    "means the centroid has crossed ABOVE the yellow line, into subcutaneous fat.\n\n"
    "Using those signs — THE SIGN IS THE ONLY RULE, there is no visual override or "
    "exception to it. Do not classify by how the contour LOOKS relative to a line "
    "(touching, brushing, dipping across it, etc.) — a real segmentation contour is "
    "never pixel-perfect, so a genuinely-N2 vein's lumen boundary will often visually "
    "graze or slightly overlap the orange line even though its actual position is still "
    "within the compartment. The centroid sign already accounts for this — trust it "
    "completely instead of re-judging from the picture:\n"
    "- d_deep NEGATIVE (centroid below the deep line) → N1.\n"
    "- d_deep POSITIVE (centroid above the deep line), no matter how small the number, "
    "AND d_sup POSITIVE → N2. THIS IS THE DEFAULT for any blob that is solidly between "
    "the two lines, including one whose contour visually touches, brushes, or appears to "
    "slightly cross either line — being NEAR a line, or LOOKING like it crosses one, is "
    "NOT the same as its centroid actually crossing it. A d_deep of +2px is still N2, "
    "exactly the same rule as +80px — do not treat a small positive number as 'close "
    "enough to N1'. Do not downgrade a blob to N1 just because it is the lowest/"
    "deepest-looking blob in the frame, or because part of its visible boundary appears "
    "to reach the orange line — check the actual sign, only the sign.\n"
    "- d_sup NEGATIVE (centroid above the yellow line) → N3, using the same sign-only "
    "rule (no visual override) as above.\n\n"
    "TWO CONFIRMED, OPPOSITE, REAL FAILURE MODES this system exists to avoid — you must "
    "guard against BOTH, not overcorrect from one into the other. Both have been "
    "confirmed to recur MULTIPLE times on real footage, including after earlier attempts "
    "to fix them, which is exactly why the rule above allows NO visual judgment call at "
    "all anymore -- only the sign:\n"
    "(a) Blobs sitting exactly at or just inside a fascia line getting defaulted to N3 "
    "when the geometry clearly places them within or below the compartment — closeness "
    "to the YELLOW line from below/inside is not superficiality.\n"
    "(b) Blobs that are genuinely still N2 (d_deep POSITIVE, even barely) getting pushed "
    "down to N1 because they sit in the lower part of the compartment and visually LOOK "
    "close to, or like they touch, the ORANGE line — this has recurred even after "
    "explicit prior instruction not to do it, so the fix now is structural: there is no "
    "situation in which a POSITIVE d_deep should produce N1. None. If you find yourself "
    "wanting to call a blob N1 because of how its contour looks against the orange line, "
    "re-check the sign — if it says positive, the answer is N2, full stop, regardless of "
    "what the picture seems to show.\n"
    "Read the precomputed pixel-distance measurement carefully — a small \"Npx above/"
    "below\" distance still has a definite sign; use ONLY that sign, never a visual "
    "impression, no matter how confident that impression feels.\n\n"
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
            reasoning_effort="default", max_tokens=_retry_max_tokens(len(blobs)),
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
