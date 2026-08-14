"""Stage 3a: look at one webcam frame and describe where the probe is on the leg.

Two-call design (see read_location):
  Stage A (read_probe_position): FAST, reasoning_effort="none" -- a simple binary
    judgment, is the probe above the knee or at/below it? Cheap and frequent.
  Stage B (read_level_and_surface): full reasoning_effort="default" -- picks the
    SPECIFIC leg_level from a NARROWED vocabulary (only the 3-4 levels on Stage A's
    side of the knee) plus leg_side and surface.

Why split it this way: real user testing found the model systematically mislabeling
calf/knee frames as dodd/hunterian/upper_thigh -- a directional anchoring bias, not
random noise (confirmed: the bias direction matched the reference image's own depicted
probe position, and a "don't anchor on the reference" text instruction did not fix it
across a full retest). A 1-of-7 choice is also just a harder discrimination task than a
1-of-3-or-4 choice on its own. Narrowing the vocabulary via a cheap upstream binary call
directly targets the actual failure rather than adding more corrective text to a prompt
that's already carrying a lot of instruction.

leg_side stays on Stage B (full reasoning), NOT the fast Stage A call, despite that
being how the user originally proposed splitting it -- confirmed twice this session that
leg_side mirroring specifically breaks under reasoning_effort="none" regardless of what
else is in the prompt (a different failure mode than the level-anchoring bug above, and
narrowing the level vocabulary doesn't make mirroring logic any easier). Flagged this
deviation explicitly rather than silently changing scope.
"""
import base64
import os

import cv2

import anatomy_knowledge
import groq_client
import knee_cv

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def _encode_jpeg_b64(frame_bgr) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr)
    return base64.b64encode(buf).decode()

# Appended to the relevant system prompts when knee_cv.occlusion_score flags a frame --
# confirmed directly on real footage (not assumed): telling the model to actively
# ignore the blurry near-lens obstruction and judge from whatever IS visible recovers
# real, confident, often-correct answers instead of a bare "uncertain" -- tested on 3
# frames that previously came back uncertain/empty, 1 became fully correct, 1 became a
# close near-miss (adjacent level), 1 still didn't converge (same rare truncation risk
# any call has, not specific to occlusion). Costs nothing extra structurally -- same
# single call, just more prompt text -- so this replaced an earlier version of this fix
# that skipped straight to "uncertain" without even trying, per explicit user direction
# not to give up on occluded frames without attempting a real answer first.
OCCLUSION_NOTE = (
    "\n\nNOTE: part of this frame may be blocked by a blurry hand/arm very close to "
    "the camera lens in the foreground -- this is NOT the clinician's hand or the "
    "probe, it is irrelevant to your task. IGNORE that blurry region entirely. Look "
    "at whatever part of the leg and probe/hand ARE actually visible and in focus "
    "(even if only part of the leg is visible) and make your best real judgment from "
    "that -- do NOT default to 'uncertain' just because part of the frame is "
    "obstructed. Only answer 'uncertain' if the probe/leg contact point itself is "
    "the part that's blocked, not merely because something else in the frame is blurry."
)


def _min_confidence(*confs: str) -> str:
    """Combine confidences from multiple agents conservatively -- the combined result is
    only as confident as its weakest contributing agent."""
    ranked = [c for c in confs if c in _CONF_RANK]
    if not ranked:
        return "low"
    return min(ranked, key=lambda c: _CONF_RANK[c])

_REFLUX_REF_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "reference_images", "reflux_probe_vs_compression_hand.jpg")
_reflux_ref_image_b64 = None


def _get_reflux_reference_image_b64() -> str | None:
    """Loaded once, reused on every Stage A call ONLY — a real annotated frame from this
    project's own footage: green circle = the hand actually holding the cabled probe
    device (the real scan location), red circle = a second hand lower on the leg gripping
    the cable/compressing the calf for a reflux test, which is NOT the probe location.
    Deliberately NOT shown to Stage B (see _get_level_reference_image_b64) — confirmed
    real testing that including this same image in both stages let the model leak
    "I recognize a reflux-testing scene" into the LEVEL judgment itself (a new
    "reflux scene -> calf" bias appeared right after fixing the opposite "reflux scene ->
    dodd" bias), rather than using it only to identify which hand to look at. Splitting
    the two reference images by stage is a direct fix for that cross-contamination, not
    just a token-cost optimization."""
    global _reflux_ref_image_b64
    if _reflux_ref_image_b64 is None and os.path.exists(_REFLUX_REF_IMAGE_PATH):
        with open(_REFLUX_REF_IMAGE_PATH, "rb") as f:
            _reflux_ref_image_b64 = base64.b64encode(f.read()).decode()
    return _reflux_ref_image_b64


_LEVEL_REF_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "reference_images", "leg_level_landmarks.jpg")
_level_ref_image_b64 = None


def _get_level_reference_image_b64() -> str | None:
    """Loaded once, reused on every Stage B call ONLY — a real leg photo from this
    project's footage, user-annotated with the full groin-to-ankle band sequence
    (groin_sfj, upper_thigh, hunterian, dodd, knee_popliteal, calf, ankle) drawn directly
    on the leg as horizontal bands with handwritten labels, for proportion calibration.
    Deliberately does NOT include the probe-vs-hand reflux example (see
    _get_reflux_reference_image_b64) — Stage B's job is purely "given roughly where the
    probe is, match it to this reference's proportions", and showing it the reflux
    example too was confirmed to bias level judgments toward whatever that example
    happened to depict, independent of the current frame's actual content."""
    global _level_ref_image_b64
    if _level_ref_image_b64 is None and os.path.exists(_LEVEL_REF_IMAGE_PATH):
        with open(_LEVEL_REF_IMAGE_PATH, "rb") as f:
            _level_ref_image_b64 = base64.b64encode(f.read()).decode()
    return _level_ref_image_b64


# NOTE: this file used to load reflux_dodd_confirmed.jpg (a real frame with a "CONFIRMED
# GROUND TRUTH" caption spelled out in text) and hand it to Agent LN/Agent S. Removed
# entirely -- confirmed on a full-video real run to cause severe answer-anchoring (the
# model read the caption text back verbatim instead of reasoning per frame). Superseded
# by knee_cv.py's classical-CV knee-height line, which carries the same "real anchor"
# idea without spelling out any answer in words. The image file itself is left in
# reference_images/ but nothing in this module references it anymore.

_SURFACE_REF_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "reference_images", "surface_anterior_medial_posterior.jpg")
_surface_ref_image_b64 = None


def _get_surface_reference_image_b64() -> str | None:
    """Loaded once, reused on every call — two real frames from this project's own
    footage, user-annotated with ground truth (not my own inference, which was wrong
    twice before this): PANEL A shows a real anterior/medial scene outlined red
    (anterior) and green (medial); PANEL B shows a real posterior scene outlined blue.
    Both panels are visually similar at a glance (same room, same patient, same general
    leg silhouette) — that similarity is the whole point: this camera angle genuinely
    cannot be read from gestalt alone, only from the specific cues the user's own
    annotations point at, which is why a picture pair is load-bearing here instead of a
    text description of "what posterior looks like"."""
    global _surface_ref_image_b64
    if _surface_ref_image_b64 is None and os.path.exists(_SURFACE_REF_IMAGE_PATH):
        with open(_SURFACE_REF_IMAGE_PATH, "rb") as f:
            _surface_ref_image_b64 = base64.b64encode(f.read()).decode()
    return _surface_ref_image_b64


# --- Stage A: fast binary above/at-or-below-knee check ---------------------------------

LEVELS_ABOVE_KNEE = ["groin_sfj", "upper_thigh", "proximal_thigh_hunterian", "distal_thigh_dodd"]
LEVELS_AT_OR_BELOW_KNEE = ["knee_popliteal", "calf", "ankle"]

POSITION_SYSTEM_PROMPT = (
    "You look at a single frame from a webcam video of a clinician performing a leg "
    "venous ultrasound exam. Your ONLY job is a simple binary judgment: is the "
    "ultrasound probe touching the leg ABOVE the knee joint (anywhere in the thigh or "
    "groin), or AT/BELOW the knee joint (the knee itself, the calf, or the ankle)? Do "
    "NOT try to guess the exact sub-level — a separate, more careful system handles that "
    "next using your answer here.\n\n"
    "CRITICAL WARNING ABOUT THE REFERENCE IMAGE: the reference image included with this "
    "message is a real photo from THIS SAME room, patient, and clinician as the frame "
    "you are classifying — it will look strikingly similar to many frames you see, "
    "because it's the same exam. That similarity is a coincidence of filming location, "
    "NOT a signal about the current frame's answer. The reference image's OWN probe "
    "position must NEVER be copied or treated as a hint for the current frame — it "
    "exists ONLY to show what the probe device looks like versus a bare hand, nothing "
    "else.\n\n"
    "FOLLOW THIS DECISION TREE IN ORDER — do not skip steps or jump to a conclusion "
    "from the overall 'vibe' of the scene:\n"
    "STEP 1 — find every hand belonging to the CLINICIAN (gloved) that is visible "
    "ANYWHERE in the frame, whether or not it's touching the leg. EXCLUDE the patient's "
    "own hand(s) (often near the top of frame holding their own clothing/waistband — "
    "never gloved) and any bystander's hand/arm (a third person, not the clinician — "
    "this footage sometimes has one). You may find 0, 1, or 2 clinician hands.\n"
    "STEP 2 — if 0 clinician hands are visible at all, answer probe_position='uncertain', "
    "probe_visible=false, and skip the remaining steps. If exactly 1 clinician hand is "
    "visible, that IS the probe hand by definition — skip directly to step 4. If 2 "
    "clinician hands are visible, go to step 3.\n"
    "STEP 3 (only reached with 2 clinician hands visible) — identify the probe hand "
    "DIRECTLY, by device grip, not by screen position or by which hand looks like it's "
    "'doing more'. This is the single most important step: DO NOT decide by comparing "
    "which hand is higher on screen, and do NOT assume a two-hand frame is automatically "
    "a reflux/compression scene. Look at each of the two hands individually and ask: "
    "'is THIS hand closed/gripping around the probe device body — a small handheld "
    "device, often white/gray, roughly cylindrical or wand-shaped, with a visible cable "
    "coming out of it — or is it flat/open against bare skin, or gripping only the "
    "cable and not the device body itself?' The hand actually gripping the device body "
    "is the probe hand — use ONLY it from here on, completely disregard the other hand "
    "(it may be compressing the calf for a reflux test, steadying the leg, holding the "
    "cable, or anything else — irrelevant to this task either way). This exact "
    "device-grip check (not position, not scene-guessing) is confirmed reliable on real "
    "footage even when a naive 'higher hand wins' or 'this looks like the reference "
    "scene' heuristic gets it wrong. Only if you truly cannot tell which hand holds the "
    "device from its grip shape, fall back to: the hand higher on screen is the probe "
    "hand.\n"
    "STEP 4 — is the probe hand identified above ACTUALLY touching the patient's leg "
    "right now (skin contact, or holding the probe/cable directly against the skin)? If "
    "NOT (e.g. resting in the clinician's lap, holding other equipment, adjusting a "
    "mask, reaching for something), contact is broken right now — answer "
    "probe_position='uncertain', probe_visible=false, and skip step 5. If yes, "
    "continue to step 5.\n"
    "STEP 5 — using ONLY the probe hand identified above, judge its position against "
    "the knee joint line:\n"
    "- The kneecap (patella, seen from the front) and the crease behind the knee "
    "(popliteal fossa, seen from the back) mark the KNEE JOINT LINE.\n"
    "- ABOVE (answer 0): the probe hand is CLEARLY above the joint line, on the fleshy "
    "thigh, with the kneecap/fossa visibly BELOW it.\n"
    "- AT/BELOW (answer 1): the probe hand is ON or straddling the joint line, OR "
    "clearly below it on the lower leg's muscle bulk (shin/calf), with the kneecap/"
    "fossa visibly AT or ABOVE it.\n"
    "- If genuinely torn, answer 'uncertain' rather than guessing.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON, "
    "in exactly this shape:\n"
    '{"num_clinician_hands": 0|1|2, '
    '"probe_hand_position": "<one short phrase describing WHERE the identified probe '
    'hand is on the leg, e.g. \'mid-thigh, medial\' or \'just below the knee\' — this '
    'is logged for human review, be specific>", '
    '"probe_position": 0|1|"uncertain", '
    '"probe_visible": true|false, '
    '"confidence": "high"|"medium"|"low", '
    '"visual_evidence": "<one short sentence: what you saw and why, naming which hand '
    'you picked and what its grip looked like if 2 hands were visible>"}'
)

POSITION_USER_PROMPT = (
    "The FIRST image is the actual webcam frame to classify. The SECOND image is a "
    "fixed reference example (not what you are classifying, not the current frame) "
    "showing the probe-vs-compression-hand pattern described in the system instructions."
)

# --- CV-assisted variant: knee height comes from knee_cv.py (classical CV, zero VLM
# tokens), not from the model's own anatomical inference. Same STEPS 1-4 (hand
# identification, unchanged and never the broken part) -- only STEP 5 changes, from
# "infer where the knee joint is" (confirmed unreliable on this patient's footage
# across 7+ independent prompt-engineering attempts) to "read a line that's already
# drawn on the image" (a trivial perception task, not anatomical reasoning). Falls back
# to the original POSITION_SYSTEM_PROMPT (no line) whenever knee_cv can't confidently
# find a knee for a given frame -- see read_probe_position.
POSITION_SYSTEM_PROMPT_WITH_LINE = (
    "You look at a single frame from a webcam video of a clinician performing a leg "
    "venous ultrasound exam. Your ONLY job is a simple binary judgment: is the "
    "ultrasound probe touching the leg ABOVE the RED LINE drawn on the image, or ON/"
    "BELOW it? Do NOT try to guess the exact sub-level — a separate, more careful "
    "system handles that next using your answer here.\n\n"
    "THE RED LINE marks the patient's real knee joint height, computed geometrically "
    "from the leg's own silhouette (not a guess, not from the reference image) -- "
    "TRUST IT COMPLETELY. Do not try to re-derive where the knee is yourself from "
    "kneecap shape, muscle bulges, or proportions; just compare the probe's contact "
    "point to the line's height.\n\n"
    "CRITICAL WARNING ABOUT THE REFERENCE IMAGE: the reference image included with "
    "this message (separate from the red line, a fixed photo from the same room/"
    "patient/clinician) exists ONLY to show what the probe device looks like versus a "
    "bare hand -- its own probe position is NOT a hint about the current frame.\n\n"
    "FOLLOW THIS DECISION TREE IN ORDER:\n"
    "STEP 1 — find every hand belonging to the CLINICIAN (gloved) that is visible "
    "ANYWHERE in the frame, whether or not it's touching the leg. EXCLUDE the patient's "
    "own hand(s) and any bystander's hand/arm. You may find 0, 1, or 2 clinician "
    "hands.\n"
    "STEP 2 — if 0 clinician hands are visible at all, answer probe_position="
    "'uncertain', probe_visible=false, and skip the remaining steps. If exactly 1 "
    "clinician hand is visible, that IS the probe hand — skip directly to step 4. If 2 "
    "clinician hands are visible, go to step 3.\n"
    "STEP 3 (only reached with 2 clinician hands visible) — identify the probe hand "
    "DIRECTLY, by device grip, not by screen position. Look at each hand and ask: 'is "
    "THIS hand closed/gripping around the probe device body -- a small handheld "
    "device, often white/gray, with a visible cable -- or is it flat/open against bare "
    "skin, or gripping only the cable?' The hand gripping the device body is the probe "
    "hand -- use ONLY it from here on. Only if you truly cannot tell, fall back to: "
    "the hand higher on screen is the probe hand.\n"
    "STEP 4 — is the probe hand identified above ACTUALLY touching the patient's leg "
    "right now? If NOT (resting in lap, holding equipment, reaching for something), "
    "answer probe_position='uncertain', probe_visible=false, and skip step 5. If yes, "
    "continue to step 5.\n"
    "STEP 5 — compare the probe hand's contact point on the leg to the RED LINE:\n"
    "- ABOVE the line (answer 0): the contact point is clearly higher up (closer to "
    "the top of frame) than where the red line crosses the leg.\n"
    "- ON/BELOW the line (answer 1): the contact point is at the same height as the "
    "line, or lower (closer to the bottom of frame / the floor) than where it crosses "
    "the leg.\n"
    "- If the probe hand's contact point is genuinely hard to place relative to the "
    "line, answer 'uncertain' rather than guessing.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON, "
    "in exactly this shape:\n"
    '{"num_clinician_hands": 0|1|2, '
    '"probe_hand_position": "<one short phrase describing WHERE the identified probe '
    'hand is on the leg>", '
    '"probe_position": 0|1|"uncertain", '
    '"probe_visible": true|false, '
    '"confidence": "high"|"medium"|"low", '
    '"visual_evidence": "<one short sentence: which hand you picked and where its '
    'contact point sits relative to the red line>"}'
)

POSITION_USER_PROMPT_WITH_LINE = (
    "The FIRST image is the actual webcam frame to classify, with a RED LINE marking "
    "the geometrically-computed knee height. The SECOND image is a fixed reference "
    "example (not what you are classifying) showing the probe-vs-compression-hand "
    "pattern described in the system instructions."
)


def normalize_position(parsed: dict) -> dict:
    pos = parsed.get("probe_position")
    if pos not in (0, 1):
        pos = "uncertain"
    hands = parsed.get("num_clinician_hands")
    if hands not in (0, 1, 2):
        hands = "uncertain"
    return {
        "probe_position": pos,
        "num_clinician_hands": hands,  # logged per user request -- lets a human spot-check
        # "probe_hand_position": the leg position of whichever hand STEP 3's device-grip
        # check identified as the probe hand (not just "the higher hand") -- renamed from
        # the old top_hand_position to match what the field actually now represents.
        "probe_hand_position": parsed.get("probe_hand_position") or "",
        "probe_visible": bool(parsed.get("probe_visible", False)),
        "confidence": parsed.get("confidence") or "low",
        "visual_evidence": parsed.get("visual_evidence") or "",
    }


# NOTE: a crop-before-classify approach was tried here and reverted. The idea: localize
# the probe hand's x-position first, crop a vertical strip around it (full height, so the
# knee landmark stays visible) to remove the room/doctor context before asking the
# above/below-knee question, on the theory that the full frame's visual similarity to the
# reference photo (same room/doctor/curtains) was itself priming a biased answer.
# Tested directly on real frames before shipping it (not assumed to work): the
# localization sub-step itself is unreliable on this footage. A continuous x-coordinate
# estimate landed on the doctor's torso instead of the hand (off by ~25% of frame width,
# confirmed on two separate frames). A discrete 5-zone classification version did better
# but was still off by one zone on a real test frame -- close enough to look plausible,
# not close enough for a crop tight enough to matter to actually contain the real hand.
# Reverted rather than ship a change that would silently crop out the very information
# the next stage needs. If revisited, the localization step itself needs to get reliable
# first (e.g. a wider zone grid tested at volume, or a completely different grounding
# approach) before cropping is worth trying again.


def read_probe_position(frame_bgr) -> dict:
    """Stage A. Cheap and fast (reasoning_effort='none', small max_tokens) — this is a
    simple binary landmark check, not the kind of multi-step reasoning (facing-direction
    mirroring) that's been confirmed to need full chain-of-thought.

    CV-assisted knee line (knee_cv.py, classical CV, zero VLM tokens): tried first on
    every frame. When it finds a confident knee height, the frame gets that line burned
    on and the model only has to compare a contact point to a drawn line -- turning the
    exact judgment that failed under 7+ different prompt-engineering attempts into a
    trivial perception task. When CV can't find a confident knee (leg not cleanly
    segmented, no valley-then-rise pattern -- happens on a minority of frames, confirmed
    on real footage), falls back to the original text-only prompt unchanged -- same
    behavior as before this fix existed, not a regression."""
    knee_y, _bbox = knee_cv.find_knee_y(frame_bgr)
    _, buf = cv2.imencode(".jpg", knee_cv.draw_knee_line(frame_bgr, knee_y) if knee_y is not None else frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    ref_b64 = _get_reflux_reference_image_b64()
    extra_images = [(ref_b64, "image/jpeg")] if ref_b64 else None

    if knee_y is not None:
        system, user, label = POSITION_SYSTEM_PROMPT_WITH_LINE, POSITION_USER_PROMPT_WITH_LINE, "stage3a_position_cv"
    else:
        system, user, label = POSITION_SYSTEM_PROMPT, POSITION_USER_PROMPT, "stage3a_position_fallback"

    parsed, _raw = groq_client.call_vlm_json(
        system, user, image_b64=img_b64, image_media_type="image/jpeg",
        extra_images=extra_images, reasoning_effort="none", max_tokens=1024,
        label=label,
    )
    return normalize_position(parsed)


# --- Stage A2 history (no longer present as code) ---------------------------------------
# Two VLM-based attempts at a dedicated thigh-vs-knee boundary check were built and
# removed: a text-only version (described the vastus medialis muscle bulge in words --
# converged on the same wrong "below" answer as everything else) and a version shown a
# real reference image with the answer spelled out in a caption (caused severe
# answer-anchoring -- 33 of 36 ticks across a full video collapsed to "distal_thigh_dodd"
# regardless of actual content). Both deleted rather than left disabled-but-present.
# Superseded by knee_cv.py: a classical-CV (zero-token) knee-height detector whose line
# is burned directly onto the frame, turning "where is the knee" (a judgment neither VLM
# attempt could reliably make) into "is this point above or below the drawn line" (a
# trivial perception task) -- see POSITION_SYSTEM_PROMPT_WITH_LINE and
# read_probe_position below. Confirmed on real footage: 5 of 7 reflux-window frames
# corrected from wrong to right, zero added token cost on the frames where CV succeeds,
# graceful fallback to the original text-only prompt (no added cost either) when it
# doesn't.
#
# NOTE: this module used to have shared majority-vote helpers (_majority_confirm_or_flip,
# _majority_plurality) used by Agent LN, Agent S, and the non-reflux posterior check.
# Removed entirely -- a real production job hit a hard TPM (tokens-per-minute) rate-limit
# crash at 75% progress, and the vote/retry escalation (each vote up to MAX_TOKENS) was
# the confirmed cause. All three call sites now trust a single draw.

# --- Agentic reflux-frame path -----------------------------------------------------------
# Built after extensive real-footage evidence that the single combined Stage B call
# (leg_side + leg_level + surface all at once) is specifically unreliable on
# reflux/compression-testing frames (two clinician hands) for the thigh-vs-calf judgment
# -- confirmed wrong via THREE independently-worded prompts, and confirmed genuinely
# non-deterministic (not just biased) via repeated-draw testing. Two VLM-only "knee
# boundary" fixes were tried and removed (see the Stage A2 history note above); the
# knee_cv.py CV-assisted line in read_probe_position now handles the above/below-knee
# gate itself, so Agent LN below gets a more reliable narrowed vocabulary to begin with.
# This agent pair is used ONLY when Stage A reports num_clinician_hands==2 (the
# reflux/compression configuration). Non-reflux frames are NOT touched -- they keep
# using the original combined Stage B call unchanged, since that path was never
# confirmed broken and decoupling it would add cost for no proven benefit.
#
# Agent LN (level-name resolver): given the already-narrowed 0/1 half from Stage A,
# picks the specific leg_level using ONLY the level reference diagram -- no surface
# question mixed in, so its full attention (and its majority vote) is spent purely on
# the thigh/knee/calf judgment that's actually been failing.
# Agent S (surface + leg_side): the existing surface/leg_side reasoning, decoupled from
# level entirely, with the same posterior-confirmation majority vote already proven on
# the combined call.

def _level_name_system_prompt(half_desc: str, allowed_levels: list[str]) -> str:
    return (
        "You look at a single frame from a webcam video of a clinician performing a leg "
        "venous ultrasound exam DURING REFLUX/COMPRESSION TESTING (two hands on the leg "
        "-- already confirmed by an earlier pass, and the earlier pass already "
        f"identified the probe hand and confirmed it is {half_desc} the knee joint). "
        "Your ONLY job is to pick the SPECIFIC leg_level within that half -- do not "
        "second-guess whether it's above or below the knee, that is already settled.\n\n"
        f"Choose leg_level from EXACTLY this list (do not use any value outside it, "
        f"except 'uncertain'): {', '.join(allowed_levels)}.\n"
        "Landmarks: the groin crease (top of thigh), the knee, the medial and lateral "
        "malleoli (ankle bones), the popliteal fossa (back of the knee).\n\n"
        + (
            "The thigh has three levels in that list (upper_thigh, proximal_thigh_hunterian, "
            "distal_thigh_dodd) and they have no distinct visible landmark between them — judge "
            "by roughly what FRACTION of the way down the thigh (groin to knee) the probe is: "
            "top third (nearest the groin) = upper_thigh; middle third = proximal_thigh_hunterian; "
            "bottom third (nearest the knee) = distal_thigh_dodd.\n"
            if any(l in allowed_levels for l in
                   ["upper_thigh", "proximal_thigh_hunterian", "distal_thigh_dodd"]) else ""
        )
        + "The reference image included with this message (a real leg from this same "
        "exam, user-annotated — NOT the frame you are classifying) shows the FULL "
        "groin-to-ankle band sequence drawn directly on the leg with handwritten labels, "
        "each as a labeled horizontal band in proportion to real leg anatomy. STRICTLY "
        "FOLLOW this reference's band proportions when assigning leg_level: measure "
        "roughly where the probe sits along the groin-to-ankle span in the CURRENT "
        "frame, then match that fraction against the reference's band boundaries.\n\n"
        "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
        "JSON:\n"
        f'{{"leg_level": "<one of {allowed_levels}>"|"uncertain", '
        '"confidence": "high"|"medium"|"low"}'
    )


def read_level_name_reflux(frame_bgr, allowed_levels: list[str], half_desc: str,
                            occluded: bool = False) -> dict:
    """Agent LN. Single call, no majority vote, no truncation retry -- removed after a
    real production job hit a hard TPM (tokens-per-minute) rate-limit crash at 75%
    progress. Also no longer uses the "confirmed dodd" reference image -- confirmed on
    a full-video real run to cause severe anchoring (33/36 ticks collapsed to
    "distal_thigh_dodd" regardless of actual content, because the image's caption text
    spelled out the answer in words and the model just read it back). Back to the
    generic groin-to-ankle proportion diagram only; the thigh/calf boundary judgment is
    an open, unsolved accuracy problem right now, not silently patched over.

    occluded: appends OCCLUSION_NOTE when knee_cv detected a near-lens obstruction --
    see read_location. Confirmed on real footage to recover real answers instead of a
    bare "uncertain" on frames a bystander's arm partially blocks."""
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    system = _level_name_system_prompt(half_desc, allowed_levels)
    if occluded:
        system += OCCLUSION_NOTE
    ref_b64 = _get_level_reference_image_b64()
    extra_images = [(ref_b64, "image/jpeg")] if ref_b64 else None

    parsed, _raw = groq_client.call_vlm_json(
        system, "Classify this frame.", image_b64=img_b64, image_media_type="image/jpeg",
        extra_images=extra_images, reasoning_effort="default", max_tokens=8192,
        label="agent_ln",
    )
    level = parsed.get("leg_level") or "uncertain"
    if level not in allowed_levels and level != "uncertain":
        level = "uncertain"
    return {"leg_level": level, "confidence": parsed.get("confidence") or "low"}


_SURFACE_LEG_SIDE_SYSTEM_PROMPT_TAIL = (
    "HOW TO DETERMINE leg_side (left/right) — this is easy to get backwards, reason "
    "through it explicitly every time, do not default to a naive screen-left="
    "patient's-left mapping:\n"
    "1. First decide the patient's orientation relative to the camera: is their "
    "FRONT facing the camera, their BACK facing the camera (cue: you see their "
    "back/shoulder blades/back of the head, not their face), or a side profile?\n"
    "2. If the patient's FRONT faces the camera: the leg on the LEFT side of the "
    "image is the PATIENT'S OWN RIGHT leg; the leg on the RIGHT side of the image is "
    "the PATIENT'S OWN LEFT leg.\n"
    "3. If the patient's BACK faces the camera: no mirroring — LEFT of image = "
    "patient's own LEFT leg, RIGHT of image = patient's own RIGHT leg.\n"
    "4. If you cannot tell, answer leg_side='uncertain' rather than guessing.\n"
    "5. Scanning the popliteal fossa (back of the knee) does NOT by itself mean the "
    "patient turned their whole body around — check actual visible cues (face, back "
    "of head/shoulders) before concluding they turned around.\n"
    "6. THIS FOOTAGE IS OFTEN CROPPED TO JUST THE LOWER LEGS — no face, shoulders, or "
    "head in frame at all. When that's the case, do NOT guess facing direction from "
    "general vibe or default to 'facing the camera' — use the visible FOOT/FEET "
    "instead, which is a reliable cue even alone:\n"
    "   - Look specifically for individual TOE shapes and toenails, and the top-of-"
    "foot instep curve. If you can clearly make out separate toes, the patient's "
    "FRONT faces the camera.\n"
    "   - Look specifically for a rounded HEEL and the Achilles tendon cord running "
    "up the back of the ankle, with no toe detail visible (the foot's sole/toes are "
    "hidden or pointing away from the camera). If that's what you see, the patient's "
    "BACK faces the camera.\n"
    "   - Do not assume toes are visible just because a foot/leg is visible in frame "
    "— actively check which of the two patterns above the visible foot actually "
    "shows before deciding. Getting this backwards (claiming toes when only a heel "
    "is shown, or vice versa) is a common, confirmed failure mode — look twice.\n\n"
    "HOW TO DETERMINE surface (anterior/medial/posterior/lateral). The clinically "
    "important distinction is GROUP-level: {anterior, medial} are the SAME vein "
    "territory (GSV) and mixing those two up is low-stakes — but POSTERIOR is a "
    "genuinely different vein territory (SSV/popliteal/Giacomini) and getting that "
    "distinction wrong is a real naming error. Spend your care on medial-or-anterior "
    "VS posterior; don't agonize over medial vs. anterior specifically.\n"
    "This camera angle can make an anterior scene and a posterior scene look "
    "surprisingly similar at a glance — do not confidently pattern-match from vibe.\n"
    "STRONG DIRECT CUE (check this FIRST, before comparing to the reference panels "
    "below): the same heel-vs-toes foot check from the leg_side reasoning above is "
    "ALSO one of the most reliable surface signals here. A visible HEEL/Achilles "
    "tendon with no toes, together with a rounded calf-muscle bulge (gastrocnemius) "
    "on the back of the lower leg rather than the harder shin-bone ridge on the "
    "front, is a strong POSTERIOR indicator — weight it at least as heavily as the "
    "panel-composition comparison below, and prefer it if the two seem to disagree.\n"
    "There is a reference image included with this message, two panels, both real "
    "frames from this exact exam, both user-verified ground truth: PANEL A is "
    "confirmed ANTERIOR/MEDIAL (red=anterior, green=medial). PANEL B is confirmed "
    "POSTERIOR (blue). Compare the CURRENT frame against BOTH panels' overall "
    "composition (how much of the leg/body is in frame, arm/hand position relative "
    "to the probe) and judge which it more closely resembles. If neither is clear, "
    "answer at 'low'/'medium' confidence rather than forcing a guess.\n"
    "- lateral: outer side of the leg, uncommon — only if clearly visible.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON:\n"
    '{"leg_side": "left"|"right"|"uncertain", '
    '"surface": "anterior"|"medial"|"posterior"|"lateral"|"uncertain", '
    '"confidence": "high"|"medium"|"low", '
    '"probe_visible": true|false, '
    '"visual_evidence": "<one sentence: facing direction + surface cues used>"}'
)


def _surface_system_prompt(position_evidence: str = None) -> str:
    hand_id_block = (
        f"A FIRST-PASS SYSTEM ALREADY LOOKED AT THIS EXACT FRAME and identified which "
        f"hand holds the real probe (this is a reflux/compression-testing frame — two "
        f"hands are on the leg, one holding the probe, one compressing the calf or "
        f"gripping the cable; that first-pass system already solved which is which). "
        f"Its finding: \"{position_evidence}\" Trust this — use the hand it identified "
        f"for your surface judgment below, do not re-litigate which hand is the "
        f"probe.\n\n"
        if position_evidence else _HAND_ID_FALLBACK
    )
    return (
        "You look at a single frame from a webcam video of a clinician performing a leg "
        "venous ultrasound exam DURING REFLUX/COMPRESSION TESTING. Your ONLY job is "
        "leg_side and surface — NOT leg_level, a separate system already handles "
        "that.\n\n" + hand_id_block + _SURFACE_LEG_SIDE_SYSTEM_PROMPT_TAIL
    )


_FOOT_CROP_NOTE = (
    "\n\nAN EXTRA IMAGE is included with this message: a ZOOMED-IN crop of this exact "
    "frame's foot/ankle region (the same frame, cropped and enlarged, not a different "
    "moment). Full-frame images at this resolution have been confirmed, via direct "
    "testing, to be unreliable for reading fine toe-vs-heel detail — the foot is a "
    "small fraction of the full frame. USE THE ZOOMED CROP, not the full frame, to make "
    "the toe/heel determination described above; trust what you see in the crop over "
    "any impression from the full-frame view if the two seem to disagree."
)


_AGENT_S_RETRY_SUFFIX = (
    "\n\nYour previous attempt at this exact frame ran out of space mid-reasoning "
    "without ever reaching a JSON answer. This time: work through leg_side and surface "
    "(including the toe/heel crop check) in at most 2 short sentences each, reach a "
    "conclusion, and move on -- do not re-litigate a judgment once made. The instant you "
    "have both, stop reasoning and output the JSON immediately."
)


def read_surface_reflux(frame_bgr, position_evidence: str = None, occluded: bool = False) -> dict:
    """Agent S. No majority vote (see read_level_name_reflux's docstring for why -- real
    TPM rate-limit crash in production caused by the VOTE pattern specifically). Also no
    longer uses the "confirmed dodd" reference image -- same anchoring problem confirmed
    here too (visual_evidence text was citing "the provided ground truth" verbatim). Back
    to the generic Panel A/B surface composite only.

    DOES now get one single, narrowly-scoped truncation retry (same bounded pattern as
    read_level_and_surface's -- a different, safer thing than the removed vote pattern).
    Added alongside the foot-crop fix below: confirmed on real repeated draws that the
    extra image + extra reasoning instructions measurably raised this call's truncation
    rate (2 of 3 real draws on one test frame came back with no JSON at all, losing all
    information for that frame) -- unacceptable regression, a bounded retry is the
    already-proven fix for exactly this shape of problem elsewhere in this file.

    occluded: see read_level_name_reflux -- confirmed on real footage to recover a full
    confident answer on a frame that previously came back completely empty.

    Foot crop (knee_cv.foot_crop): a zoomed-in crop of the foot/ankle region, added as an
    extra image whenever knee_cv can find the leg bbox -- fixes a CONFIRMED real vision
    failure (not a prompt-wording issue): the model reliably hallucinated "toes visible"
    on full frames that unambiguously showed only a bare heel, on real footage, even when
    directly and explicitly asked to check. See knee_cv.foot_crop's docstring for the
    real before/after test that led to this fix."""
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    system = _surface_system_prompt(position_evidence)
    crop = knee_cv.foot_crop(frame_bgr)
    if crop is not None:
        system += _FOOT_CROP_NOTE
    if occluded:
        system += OCCLUSION_NOTE
    ref_b64 = _get_surface_reference_image_b64()
    extra_images = [im for im in [
        (ref_b64, "image/jpeg") if ref_b64 else None,
        (_encode_jpeg_b64(crop), "image/jpeg") if crop is not None else None,
    ] if im] or None

    parsed, raw = groq_client.call_vlm_json(
        system, "Classify this frame.", image_b64=img_b64, image_media_type="image/jpeg",
        extra_images=extra_images, reasoning_effort="default", max_tokens=8192,
        label="agent_s",
    )
    if _looks_truncated(parsed, raw):
        print(f"[stage3a] agent_s call truncated ({len(raw)} chars, no JSON) -- retrying once")
        parsed, raw = groq_client.call_vlm_json(
            system, "Classify this frame." + _AGENT_S_RETRY_SUFFIX, image_b64=img_b64,
            image_media_type="image/jpeg", extra_images=extra_images,
            reasoning_effort="default", max_tokens=8192, label="agent_s_retry",
        )
    return {
        "leg_side": parsed.get("leg_side") or "uncertain",
        "surface": parsed.get("surface") or "uncertain",
        "confidence": parsed.get("confidence") or "low",
        "probe_visible": bool(parsed.get("probe_visible", False)),
        "visual_evidence": parsed.get("visual_evidence") or "",
    }


# --- Stage B: full leg_side/surface + narrowed leg_level --------------------------------

_HAND_ID_FALLBACK = (
    "WHICH DEVICE IS 'THE PROBE' — do not confuse it with a bare hand. The actual "
    "ultrasound probe is a small handheld device connected by a cable to the "
    "ultrasound machine; during this exam the clinician sometimes uses their OTHER "
    "hand to squeeze/press the calf muscle or grip the cable to provoke/observe "
    "venous reflux — that hand is NOT the probe and must be IGNORED even though it "
    "is often closer to camera or more visually prominent. If two hands are visible "
    "at different heights, the HIGHER hand (closer to the top of the leg/further "
    "from the floor) is the one holding the probe. Recognizing a reflux-testing "
    "scene tells you WHICH HAND to look at — it tells you NOTHING about leg_level; "
    "reflux testing happens at every level of this exam, thigh included.\n\n"
)


def _level_system_prompt(allowed_levels: list[str], position_evidence: str = None,
                          has_foot_crop: bool = False) -> str:
    """has_foot_crop: this model has a hard 3-image-per-call limit, and this prompt's
    call already sends 2 fixed reference images (level bands + surface panel) alongside
    the real frame -- exactly 3, no room for a 4th. When knee_cv.foot_crop finds a crop
    (see its docstring), it REPLACES the surface panel image rather than appending
    (confirmed necessary: appending it hit a real 400 "Too many images... supports up to
    3" error). This flag switches the surface-determination text below to match:
    describing the foot crop that will actually be sent instead of the panel composite
    that won't be, so the prompt never references an image the model wasn't given
    (confirmed elsewhere in this project that mismatched image references cause the
    model to hallucinate seeing something that isn't there)."""
    hand_id_block = (
        f"A FIRST-PASS SYSTEM ALREADY LOOKED AT THIS EXACT FRAME and identified which "
        f"hand holds the real probe (this exam sometimes shows two hands on the leg "
        f"during reflux/compression testing — one holding the probe, one compressing "
        f"the calf or gripping the cable; that first-pass system already solved which "
        f"is which). Its finding: \"{position_evidence}\" Trust this — use the hand it "
        f"identified for your leg_level and surface judgments below, and do not "
        f"re-litigate which hand is the probe.\n\n"
        if position_evidence else _HAND_ID_FALLBACK
    )
    return (
        "You look at a single frame from a webcam video of a clinician performing a leg "
        "venous ultrasound exam. Your job is to describe where the ultrasound probe is "
        "touching the patient's leg — you are not diagnosing anything. A separate "
        "first-pass system has already determined roughly which half of the leg the "
        "probe is on; you only need to pick the SPECIFIC leg_level within that half.\n\n"
        f"Choose leg_level from EXACTLY this list (do not use any value outside it, "
        f"except 'uncertain'): {', '.join(allowed_levels)}.\n"
        "Landmarks: the groin crease (top of thigh), the knee, the medial and lateral "
        "malleoli (ankle bones), the popliteal fossa (back of the knee).\n\n"
        + (
            "The thigh has three levels in that list (upper_thigh, proximal_thigh_hunterian, "
            "distal_thigh_dodd) and they have no distinct visible landmark between them — judge "
            "by roughly what FRACTION of the way down the thigh (groin to knee) the probe is: "
            "top third (nearest the groin) = upper_thigh; middle third = proximal_thigh_hunterian; "
            "bottom third (nearest the knee) = distal_thigh_dodd. Do not skip straight from "
            "groin_sfj or upper_thigh to distal_thigh_dodd without considering whether the probe "
            "position actually looks closer to the middle third first.\n"
            if any(l in allowed_levels for l in
                   ["upper_thigh", "proximal_thigh_hunterian", "distal_thigh_dodd"]) else ""
        )
        + "The SECOND reference image included with this message (a real leg from this "
        "same exam, user-annotated — NOT the frame you are classifying) shows the FULL "
        "groin-to-ankle band sequence drawn directly on the leg with handwritten labels: "
        "groin/SFJ, upper thigh, hunterian, dodd, knee/popliteal, calf, ankle, each as a "
        "labeled horizontal band in proportion to real leg anatomy. STRICTLY FOLLOW this "
        "reference's band proportions when assigning leg_level: measure roughly where the "
        "probe sits along the groin-to-ankle span in the CURRENT frame, then match that "
        "fraction against the reference's band boundaries. Use it ONLY for proportion — "
        "it is not a hint about what level the current frame shows, and its own probe "
        "position has ZERO bearing on your answer.\n\n"
        + hand_id_block +
        "DISAMBIGUATING REGION BORDERS (knee vs. thigh vs. calf) — a common failure mode "
        "is jumping to the wrong adjacent region when the probe is near a joint:\n"
        "- The kneecap (patella, seen from the front) and the crease behind the knee "
        "(popliteal fossa, seen from the back) mark the KNEE JOINT LINE.\n"
        "- If the probe is on or straddling that joint line (from either the front or "
        "the back), the answer is knee_popliteal — not distal_thigh_dodd and not calf.\n"
        "- distal_thigh_dodd means the probe is CLEARLY above the joint line (roughly a "
        "hand's-width or more up), still on the fleshy thigh, kneecap/fossa visibly "
        "below the probe.\n"
        "- calf means the probe is CLEARLY below the joint line (roughly a hand's-width "
        "or more down), on the lower leg's muscle bulk, kneecap/fossa visibly above the "
        "probe.\n"
        "- Before choosing distal_thigh_dodd or calf, actively look for the kneecap/"
        "fossa landmark and judge distance from it — do not let an oblique camera angle "
        "push you toward the wrong side of the joint just because the landmark isn't "
        "centered in frame.\n\n"
        "HOW TO DETERMINE leg_side (left/right) — this is easy to get backwards, reason "
        "through it explicitly every time, do not default to a naive screen-left="
        "patient's-left mapping:\n"
        "1. First decide the patient's orientation relative to the camera: is their "
        "FRONT facing the camera, their BACK facing the camera (cue: you see their "
        "back/shoulder blades/back of the head, not their face), or a side profile?\n"
        "2. If the patient's FRONT faces the camera: this is the same as facing "
        "another person — their body is mirrored relative to the image. The leg on the "
        "LEFT side of the image is the PATIENT'S OWN RIGHT leg; the leg on the RIGHT "
        "side of the image is the PATIENT'S OWN LEFT leg.\n"
        "3. If the patient's BACK faces the camera: no mirroring. The leg on the LEFT "
        "side of the image is the patient's own LEFT leg; the leg on the RIGHT side of "
        "the image is the patient's own RIGHT leg.\n"
        "4. If you cannot tell which way the patient is facing, answer leg_side="
        "'uncertain' rather than guessing — never default to assuming the image is "
        "unmirrored.\n"
        "5. IMPORTANT: scanning the popliteal fossa (back of the knee) does NOT by "
        "itself mean the patient turned their whole body around. A clinician can access "
        "the back of the knee with only a small weight-shift or knee bend while the "
        "patient's overall stance and facing direction stay the same as moments before. "
        "Do not re-derive facing direction from scratch just because the surface being "
        "scanned is posterior — check the actual visible cues (face, back of head/"
        "shoulders) in THIS frame before concluding the patient turned around.\n"
        "6. THIS FOOTAGE IS OFTEN CROPPED TO JUST THE LOWER LEGS — no face, shoulders, "
        "or head in frame at all. When that's the case, do NOT guess facing direction "
        "from general vibe or default to 'facing the camera' — use the visible FOOT/"
        "FEET instead, which is a reliable cue even alone:\n"
        "   - Look specifically for individual TOE shapes and toenails, and the "
        "top-of-foot instep curve. If you can clearly make out separate toes, the "
        "patient's FRONT faces the camera.\n"
        "   - Look specifically for a rounded HEEL and the Achilles tendon cord "
        "running up the back of the ankle, with no toe detail visible (the foot's "
        "sole/toes are hidden or pointing away from the camera). If that's what you "
        "see, the patient's BACK faces the camera.\n"
        "   - Do not assume toes are visible just because a foot/leg is visible in "
        "frame — actively check which of the two patterns above the visible foot "
        "actually shows before deciding. Getting this backwards (claiming toes when "
        "only a heel is shown, or vice versa) is a common, confirmed failure mode — "
        "look twice.\n\n"
        "HOW TO DETERMINE surface (anterior/medial/posterior/lateral) of the leg being "
        "scanned. The clinically important distinction is GROUP-level: {anterior, "
        "medial} are the SAME vein territory (GSV) and mixing those two up specifically "
        "is low-stakes — but POSTERIOR is a genuinely different vein territory (SSV/"
        "popliteal/Giacomini) and getting that distinction wrong is a real naming "
        "error. So spend your care on medial-or-anterior VS posterior; don't agonize "
        "over medial vs. anterior specifically.\n"
        "BE HONEST ABOUT AMBIGUITY HERE: in this kind of tightly-cropped, oblique "
        "webcam angle (no face visible), the OVERALL gestalt of a leg is NOT a "
        "reliable signal for anterior/medial vs. posterior on its own — this camera "
        "angle can make an anterior scene and a posterior scene look surprisingly "
        "similar at a glance, confirmed on real footage from this exact exam. Do not "
        "confidently pattern-match from general vibe.\n"
        + (
            "STRONG DIRECT CUE, and your PRIMARY evidence for this call (there is no "
            "separate reference-panel image this time — see below for why): the same "
            "heel-vs-toes foot check from the leg_side reasoning above IS the main "
            "surface signal here. The THIRD image included with this message is NOT a "
            "fixed reference — it is a ZOOMED-IN crop of THIS SAME FIRST FRAME's foot/"
            "ankle region, enlarged because fine toe/heel detail is confirmed unreliable "
            "to read on the full first image at this resolution. Look at that crop: a "
            "visible HEEL/Achilles tendon with no toes, together with a rounded "
            "calf-muscle bulge (gastrocnemius) on the back of the lower leg rather than "
            "the harder shin-bone ridge on the front, is a strong POSTERIOR indicator. "
            "Toes clearly visible in the crop point to anterior/medial instead. If the "
            "crop is genuinely inconclusive, answer at 'low'/'medium' confidence rather "
            "than forcing a guess.\n"
            "- lateral: the probe is on the OUTER side of the leg, away from the other "
            "leg — uncommon in this kind of exam, only choose it if clearly visible.\n\n"
            if has_foot_crop else
            "STRONG DIRECT CUE (check this FIRST, before comparing to the reference "
            "panel below): the same heel-vs-toes foot check from the leg_side reasoning "
            "above is ALSO one of the most reliable surface signals here. A visible "
            "HEEL/Achilles tendon with no toes, together with a rounded calf-muscle "
            "bulge (gastrocnemius) on the back of the lower leg rather than the harder "
            "shin-bone ridge on the front, is a strong POSTERIOR indicator — weight it "
            "at least as heavily as the panel-composition comparison below, and prefer "
            "it if the two seem to disagree.\n"
            "There is a THIRD reference image included with this message (separate from "
            "the probe-vs-hand/levels one) with two panels, both real frames from this "
            "exact exam, both user-verified ground truth (not guessed): PANEL A is "
            "confirmed ANTERIOR/MEDIAL (annotated red=anterior, green=medial). PANEL B is "
            "confirmed POSTERIOR (annotated blue). Compare the CURRENT frame you are "
            "classifying against BOTH panels and judge which one it more closely resembles "
            "— not just 'does a leg appear', but the overall composition: how much of the "
            "leg(s)/body is in frame, the clinician's arm/hand position relative to the "
            "probe, how the probe is angled relative to the leg(s). If the current frame "
            "closely resembles PANEL A's composition, answer medial or anterior. If it "
            "closely resembles PANEL B's composition, answer posterior. If it resembles "
            "neither clearly, do not force a confident guess — answer at 'low' or 'medium' "
            "confidence, or 'uncertain' if genuinely unreadable, rather than asserting a "
            "surface the image doesn't actually support.\n"
            "- lateral: the probe is on the OUTER side of the leg, away from the other leg "
            "— uncommon in this kind of exam, only choose it if clearly visible (neither "
            "reference panel is an example of this).\n\n"
        )
        + "Work through the facing-direction and leg_side reasoning ONCE, reach a "
        "conclusion, and move on — do not re-litigate the same left/right judgment "
        "back and forth repeatedly. If after one careful pass you are genuinely torn, "
        "answer 'uncertain' rather than continuing to deliberate.\n\n"
        + (
            "STRICT reasoning budget: think in at most 7 short sentences total, covering "
            "in order — (1) facing direction, (2) which leg is which, (3) which hand "
            "holds the actual probe device, (4) leg_level using the joint-line landmark "
            "check and the level reference's proportions, (5) look SPECIFICALLY at the "
            "THIRD (zoomed foot-crop) image and state plainly whether it shows toes or a "
            "heel — do not skip or rush this step, it is a separate dedicated check, not "
            "a restatement of step 1, (6) surface, using step 5's finding as your primary "
            "evidence, (7) done. Your FIRST judgment on each point is final — do NOT "
            "write follow-up sentences starting with 'Wait', 'Actually', 'No,', 'Let me "
            "reconsider', 'Hmm', or any other self-correction. The instant you have all "
            "judgments, stop reasoning and output the JSON immediately — a long internal "
            "monologue is a failure, not thoroughness.\n\n"
            if has_foot_crop else
            "STRICT reasoning budget: think in at most 6 short sentences total, covering "
            "in order — (1) facing direction, (2) which leg is which, (3) which hand holds "
            "the actual probe device, (4) leg_level using the joint-line landmark check "
            "and the level reference's proportions, (5) surface, (6) done. Your FIRST judgment on each "
            "point is final — do NOT write follow-up sentences starting with 'Wait', "
            "'Actually', 'No,', 'Let me reconsider', 'Hmm', or any other self-correction. "
            "The instant you have all judgments, stop reasoning and output the JSON "
            "immediately — a long internal monologue is a failure, not thoroughness.\n\n"
        )
        + "WHEN TO ANSWER 'uncertain': only for a field where the image genuinely gives "
        "you no usable evidence — e.g. the probe/leg is completely out of frame, fully "
        "hidden, or the frame itself is unusable (blank/blurred beyond recognition). If "
        "the probe and leg ARE visible but the view is merely awkward, oblique, or "
        "partially occluded (clothing, the other hand, motion blur), do not default "
        "straight to 'uncertain' — make your best-supported judgment at 'low' or "
        "'medium' confidence instead. A low-confidence real answer grounded in partial "
        "evidence is more useful than a blank 'uncertain'; reserve 'uncertain' for when "
        "you truly have nothing to go on, not merely when you're not fully sure. "
        "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
        "JSON, in exactly this shape:\n"
        '{"leg_side": "left"|"right"|"uncertain", '
        f'"leg_level": "<one of {allowed_levels}>"|"uncertain", '
        '"surface": "anterior"|"medial"|"posterior"|"lateral"|"uncertain", '
        '"confidence": "high"|"medium"|"low", '
        '"probe_visible": true|false, '
        '"visual_evidence": "<one sentence: what landmarks/cues you actually used, '
        'INCLUDING which way the patient is facing and how that determined leg_side>"}'
    )


def _level_user_prompt(has_foot_crop: bool) -> str:
    """has_foot_crop controls what the THIRD image actually is — see
    _level_system_prompt's has_foot_crop docstring: this model has a hard 3-image cap,
    so the foot crop REPLACES the surface reference panel in the same image slot rather
    than being appended as a fourth. Keep this in sync with read_level_and_surface's
    extra_images construction — the text here must always describe the image that was
    actually sent, not a fixed assumption."""
    third_image_desc = (
        "The THIRD image is NOT a fixed reference — it is a ZOOMED-IN crop of THIS SAME "
        "FIRST FRAME's foot/ankle region, enlarged so fine toe-vs-heel detail "
        "(unreliable to read on the full first image at this resolution, confirmed by "
        "direct testing) is actually visible. Use it specifically for the toe/heel "
        "facing-direction and surface check described in the system instructions, "
        "trusting it over the full first image if they seem to disagree."
        if has_foot_crop else
        "The THIRD image is a fixed REFERENCE two-panel composite with ground-truth "
        "panels for the anterior/medial vs. posterior surface distinction. Use it only "
        "to recognize those patterns if they appear in the first image."
    )
    return (
        "The FIRST image is the actual frame from the webcam video, at the timestamp "
        "matching an ultrasound frame we need to interpret — identify the probe location "
        "on the leg IN THIS FIRST IMAGE. The SECOND image is a fixed REFERENCE EXAMPLE "
        "(not from this timestamp, not what you are classifying): the groin-to-ankle "
        "leg-level band sequence for proportion reference. Use it only to recognize "
        "that pattern if it appears in the first image. " + third_image_desc
    )

MAX_TOKENS = 16384  # Groq's hard ceiling for this model's context window — confirmed via a
# real 400 error when 25000 was tried; cannot be raised further.

# HISTORY: this module used to have a truncation retry, removed along with the
# majority-vote helpers after a real TPM crash -- but that crash came from the VOTE
# pattern (multiple full re-draws per tick, each up to MAX_TOKENS, on a dense long
# video), not from a single scoped truncation retry. Re-added below as its own, narrower
# thing: confirmed real failure mode this targets -- on a SHORT clip with only one
# Stage 3a call scheduled for the whole video, that one call truncating (a real 32.5s/
# 61151-char non-JSON response, confirmed on actual footage) meant leg_side/leg_level/
# surface fell back to "uncertain" for the ENTIRE clip, with no later refresh ever
# coming to fix it -- the "gets picked up again on the next refresh" assumption behind
# removing the retry only holds for long videos with many scheduled calls, not short
# ones. A single truncation-only retry (same bounded pattern already proven safe and
# re-validated repeatedly in stage2_fascia_classify.py, now also sitting behind
# groq_client's OTPM/TPM budget limiter that didn't exist when the original crash
# happened) closes this gap without reintroducing the multi-vote cost risk.
_RETRY_SUFFIX = (
    "\n\nYour previous attempt at this exact frame ran out of space mid-reasoning "
    "without ever reaching a JSON answer -- you were spending too much reasoning. This "
    "time: work through leg_side, leg_level, and surface in at most 2 short sentences "
    "each, reach a conclusion, and move on -- do not re-litigate a judgment once made. "
    "The instant you have all three, stop reasoning and output the JSON immediately."
)


def _looks_truncated(parsed: dict, raw: str) -> bool:
    """Same pattern as stage2_fascia_classify._looks_truncated -- true when the call
    produced no usable JSON at all and the raw text is long, i.e. the model spent its
    whole budget mid-reasoning instead of a genuinely short/empty response."""
    return not parsed and len(raw) > 2000


def normalize(parsed: dict, allowed_levels: list[str] = None) -> dict:
    """Fills safe 'uncertain'/'low' defaults so downstream code never KeyErrors on a
    malformed or empty VLM response — this is the system-boundary validation point.
    allowed_levels, if given, guards against the model returning a level outside the
    vocabulary it was actually given (e.g. hallucinating 'calf' when only thigh levels
    were offered) — falls back to 'uncertain' rather than silently accepting a value the
    prompt explicitly excluded."""
    level = parsed.get("leg_level") or "uncertain"
    if allowed_levels is not None and level not in allowed_levels and level != "uncertain":
        level = "uncertain"
    return {
        "leg_side": parsed.get("leg_side") or "uncertain",
        "leg_level": level,
        "surface": parsed.get("surface") or "uncertain",
        "confidence": parsed.get("confidence") or "low",
        "probe_visible": bool(parsed.get("probe_visible", False)),
        "visual_evidence": parsed.get("visual_evidence") or "",
    }


def read_level_and_surface(frame_bgr, allowed_levels: list[str], position_evidence: str = None,
                            occluded: bool = False, pos=None) -> dict:
    """Stage B. Full reasoning (reasoning_effort='default') — confirmed necessary for
    the facing-direction mirroring and surface disambiguation this does; only the
    leg_level VOCABULARY is narrowed here compared to the old single-call design, not
    the reasoning depth.

    position_evidence: Stage A's visual_evidence string, if it produced one — passed
    through so Stage B can reuse Stage A's already-completed probe-vs-hand judgment
    instead of re-deriving it blind from the raw frame a second time. Two independent
    chances to get the same judgment wrong is strictly worse than one, and it was
    needlessly bloating this prompt with a full copy of the same instructions.

    occluded: see read_level_name_reflux -- appends OCCLUSION_NOTE when knee_cv
    detected a near-lens obstruction, so non-reflux frames get the same "try anyway"
    treatment as the reflux agents rather than defaulting straight to uncertain.

    pos: Stage A's 0/1/"uncertain" reading — used ONLY as a last-resort fallback (see
    below) if this combined call truncates on both the first attempt and the retry.
    Confirmed real failure mode on real footage: a single webcam frame that reliably
    produces ~61-62k chars of non-terminating reasoning on EVERY attempt (not a random
    fluke -- two independent draws converged to nearly the same length), meaning a
    same-shaped retry can't fix it. The already-proven split-agent pattern (Agent LN +
    Agent S, built for reflux frames -- see the block comment above their definitions)
    reasons about leg_level and surface SEPARATELY, each a much smaller/simpler judgment
    than this combined call's leg_side+leg_level+surface-with-facing-direction-mirroring
    -- far less likely to hit the same runaway-reasoning failure. Falling back to it here
    reuses existing, tested code rather than adding new speculative logic."""
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    crop = knee_cv.foot_crop(frame_bgr)  # see knee_cv.foot_crop's docstring — fixes a
    # confirmed real vision failure (hallucinated "toes visible" on heel-only frames),
    # not a prompt-wording issue.
    has_foot_crop = crop is not None
    system = _level_system_prompt(allowed_levels, position_evidence, has_foot_crop=has_foot_crop)
    if occluded:
        system += OCCLUSION_NOTE
    user = _level_user_prompt(has_foot_crop=has_foot_crop)
    level_ref_b64 = _get_level_reference_image_b64()
    # Groq's real hard cap for this model is 3 images per call (confirmed via a real 400
    # error), and this call already sends the real frame + level reference = 2. The foot
    # crop REPLACES the surface reference panel rather than appending a 4th image when
    # available — see _level_system_prompt's has_foot_crop docstring for why this is safe
    # (the prompt text is switched to match exactly what's actually sent, never claiming
    # an image exists that wasn't included). Falls back to the original surface panel
    # when knee_cv can't find a crop for this frame — unchanged behavior, not a
    # regression for frames the new fix doesn't apply to.
    third_image_b64 = _encode_jpeg_b64(crop) if has_foot_crop else _get_surface_reference_image_b64()
    extra_images = [im for im in [
        (level_ref_b64, "image/jpeg") if level_ref_b64 else None,
        (third_image_b64, "image/jpeg") if third_image_b64 else None,
    ] if im] or None
    # No posterior majority-vote here (removed after a real TPM crash caused by that
    # specific multi-vote pattern -- see history note above MAX_TOKENS) -- but DOES get
    # one single, narrowly-scoped truncation retry, which is a different, bounded thing.
    parsed, raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/jpeg",
                                             extra_images=extra_images,
                                             reasoning_effort="default", max_tokens=MAX_TOKENS,
                                             label="stage3a_level_surface")
    if _looks_truncated(parsed, raw):
        print(f"[stage3a] level_surface call truncated ({len(raw)} chars, no JSON) -- retrying once")
        parsed, raw = groq_client.call_vlm_json(system, user + _RETRY_SUFFIX, image_b64=img_b64,
                                                 image_media_type="image/jpeg", extra_images=extra_images,
                                                 reasoning_effort="default", max_tokens=MAX_TOKENS,
                                                 label="stage3a_level_surface_retry")
        if _looks_truncated(parsed, raw) and pos in (0, 1):
            # Both attempts truncated on this exact frame -- confirmed real footage
            # case where a same-shaped retry doesn't help (the retry converged to
            # almost the same length as the first attempt). Fall back to the smaller,
            # already-proven split-agent judgments instead of giving up to all-uncertain.
            print(f"[stage3a] level_surface truncated again after retry -- falling back to split agents (LN+S)")
            half_desc = "ABOVE" if pos == 0 else "AT/BELOW"
            level_result = read_level_name_reflux(frame_bgr, allowed_levels, half_desc, occluded=occluded)
            surface_result = read_surface_reflux(frame_bgr, position_evidence, occluded=occluded)
            return {
                "leg_side": surface_result["leg_side"],
                "leg_level": level_result["leg_level"],
                "surface": surface_result["surface"],
                "confidence": _min_confidence(level_result["confidence"], surface_result["confidence"]),
                "probe_visible": surface_result["probe_visible"],
                "visual_evidence": (
                    f"[fallback after 2x truncation] [Agent LN: leg_level="
                    f"{level_result['leg_level']} conf={level_result['confidence']}] "
                    f"[Agent S: {surface_result['visual_evidence']}]"
                ),
            }
    return normalize(parsed, allowed_levels)


# --- Combined entry point ----------------------------------------------------------------

def read_location(frame_bgr) -> dict:
    """Runs Stage A then Stage B (see module docstring). Returns the same shape as the
    old single-call read_location (leg_side/leg_level/surface/confidence/probe_visible/
    visual_evidence) so pipeline.py and stage3_vein_naming.py need no changes, PLUS extra
    fields from Stage A for optional UI display / manual log review, per the user's
    request: "probe_position_stage_a" (0/1/"uncertain"), "num_clinician_hands" (0/1/2/
    "uncertain" -- lets a human spot-check whether the hand COUNT itself was right,
    separate from the above/below-knee call), and "probe_hand_position" (the model's own
    one-phrase description of where the identified probe hand actually is)."""
    # Occlusion awareness (knee_cv.occlusion_score, zero VLM tokens to detect) --
    # confirmed on real footage: a bystander's/camera-operator's arm sometimes fills a
    # large chunk of frame with near-lens blur for several consecutive seconds. An
    # earlier version of this fix skipped straight to "uncertain" on detection to save
    # tokens -- reverted per explicit direction: skipping loses real, recoverable
    # answers. Confirmed directly: telling the model to ignore the blurry region and
    # judge from whatever IS visible recovers a correct or near-correct answer on 2 of
    # 3 tested frames, and a full confident (if not verified correct) answer on the
    # 3rd, instead of a bare "uncertain" -- so this now flows into every downstream
    # call as an added instruction, not a skip. No extra Groq calls, just more prompt
    # text on the calls that were happening anyway.
    occluded = knee_cv.occlusion_score(frame_bgr) >= knee_cv.OCCLUSION_SCORE_THRESHOLD

    position = read_probe_position(frame_bgr)
    pos = position["probe_position"]
    is_reflux = position["num_clinician_hands"] == 2
    # The above/below-knee judgment (pos) is now CV-assisted inside read_probe_position
    # itself (see knee_cv.py) -- no separate Stage A2 call needed here anymore.

    if pos == 0:
        allowed_levels = LEVELS_ABOVE_KNEE
    elif pos == 1:
        allowed_levels = LEVELS_AT_OR_BELOW_KNEE
    else:
        # Stage A itself uncertain (or probe not visible) -- don't over-constrain Stage B,
        # let it consider the full vocabulary rather than force a guess into the wrong half.
        allowed_levels = [l for l in anatomy_knowledge.LEG_LEVELS if l != "uncertain"]

    if is_reflux and pos in (0, 1):
        # Agentic reflux path (Agent LN + Agent S) -- see block comment above their
        # definitions for why this replaces the combined Stage B call specifically for
        # reflux/compression-testing frames, and ONLY those. Cost note: this path costs
        # MORE per frame than the combined call (multiple agents, some with majority
        # votes) -- it is not a cost optimization on its own, it's an accuracy fix
        # scoped narrowly to the subset of frames actually confirmed broken. Non-reflux
        # frames (the majority of most videos) are completely unaffected and keep
        # paying the original, single-call cost.
        half_desc = "ABOVE" if pos == 0 else "AT/BELOW"
        level_result = read_level_name_reflux(frame_bgr, allowed_levels, half_desc, occluded=occluded)
        surface_result = read_surface_reflux(frame_bgr, position["visual_evidence"] or None, occluded=occluded)
        result = {
            "leg_side": surface_result["leg_side"],
            "leg_level": level_result["leg_level"],
            "surface": surface_result["surface"],
            "confidence": _min_confidence(level_result["confidence"], surface_result["confidence"]),
            "probe_visible": surface_result["probe_visible"],
            "visual_evidence": (
                f"[Agent LN: leg_level={level_result['leg_level']} "
                f"conf={level_result['confidence']}] [Agent S: {surface_result['visual_evidence']}]"
                + (" [occlusion-aware]" if occluded else "")
            ),
        }
    else:
        result = read_level_and_surface(frame_bgr, allowed_levels,
                                         position_evidence=position["visual_evidence"] or None,
                                         occluded=occluded, pos=pos)

    result["probe_position_stage_a"] = pos
    result["num_clinician_hands"] = position["num_clinician_hands"]
    result["probe_hand_position"] = position["probe_hand_position"]
    # If Stage A saw the probe but Stage B somehow didn't, prefer Stage A's read on
    # probe_visible -- Stage A ran on a simpler question and is less likely to be wrong
    # about basic visibility.
    if position["probe_visible"] and not result["probe_visible"]:
        result["probe_visible"] = True
    return result
