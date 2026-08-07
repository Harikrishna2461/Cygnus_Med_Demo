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
    probe is, match it to PANEL B's proportions", and showing it the reflux example too
    was confirmed to bias level judgments toward whatever that example happened to
    depict, independent of the current frame's actual content."""
    global _level_ref_image_b64
    if _level_ref_image_b64 is None and os.path.exists(_LEVEL_REF_IMAGE_PATH):
        with open(_LEVEL_REF_IMAGE_PATH, "rb") as f:
            _level_ref_image_b64 = base64.b64encode(f.read()).decode()
    return _level_ref_image_b64


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
    "WHICH DEVICE IS 'THE PROBE' — do not confuse it with a bare hand. See the "
    "reference image included with this message: the GREEN circle marks the hand "
    "actually holding the probe device against the skin — a small handheld device "
    "with a visible cable running "
    "from it — that is the TRUE scan location. The RED circle marks a second hand, "
    "LOWER on the leg, gripping the cable and/or compressing the calf muscle to provoke "
    "venous reflux — that hand's location is NOT the probe location and must be "
    "IGNORED even though it is often closer to camera, more visually prominent, or "
    "moving more than the actual scanning hand. RELIABLE SHORTCUT, confirmed to hold "
    "across this entire exam's footage: when you see two hands on the leg at once, at "
    "different heights, the HIGHER hand (the one closer to the top of the leg/further "
    "from the floor) is the one holding the probe, and the LOWER hand is the "
    "compression/cable hand — this configuration is consistent throughout this exam, "
    "so you can use hand height directly as your primary signal, not just device shape.\n"
    "CRITICAL: the reflux/compression pattern above tells you WHICH HAND to look at — it "
    "tells you NOTHING about the answer itself. Seeing a reflux-compression scene does "
    "NOT mean the probe is at/below the knee, and does not mean it's above the knee "
    "either — reflux testing happens at every level of this exam, thigh included. Once "
    "you've identified the correct (higher) hand, judge its position against the knee "
    "joint line exactly the same way you would with only one hand visible. Do not let "
    "recognizing 'this is a reflux-testing frame' bias your answer either direction.\n\n"
    "HOW TO JUDGE ABOVE VS. AT/BELOW THE KNEE:\n"
    "- The kneecap (patella, seen from the front) and the crease behind the knee "
    "(popliteal fossa, seen from the back) mark the KNEE JOINT LINE.\n"
    "- ABOVE (answer 0): the probe is CLEARLY above the joint line, on the fleshy "
    "thigh, with the kneecap/fossa visibly BELOW the probe.\n"
    "- AT/BELOW (answer 1): the probe is ON or straddling the joint line, OR clearly "
    "below it on the lower leg's muscle bulk (shin/calf), with the kneecap/fossa "
    "visibly AT or ABOVE the probe.\n"
    "- If genuinely torn or the probe/leg isn't visible, answer 'uncertain' rather "
    "than guessing.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON, "
    "in exactly this shape:\n"
    '{"probe_position": 0|1|"uncertain", '
    '"probe_visible": true|false, '
    '"confidence": "high"|"medium"|"low", '
    '"visual_evidence": "<one short sentence: what you saw and why>"}'
)

POSITION_USER_PROMPT = (
    "The FIRST image is the actual webcam frame to classify. The SECOND image is a "
    "fixed reference example (not what you are classifying, not the current frame) "
    "showing the probe-vs-compression-hand pattern described in the system instructions."
)


def normalize_position(parsed: dict) -> dict:
    pos = parsed.get("probe_position")
    if pos not in (0, 1):
        pos = "uncertain"
    return {
        "probe_position": pos,
        "probe_visible": bool(parsed.get("probe_visible", False)),
        "confidence": parsed.get("confidence") or "low",
        "visual_evidence": parsed.get("visual_evidence") or "",
    }


def read_probe_position(frame_bgr) -> dict:
    """Stage A. Cheap and fast (reasoning_effort='none', small max_tokens) — this is a
    simple binary landmark check, not the kind of multi-step reasoning (facing-direction
    mirroring) that's been confirmed to need full chain-of-thought. Only uses the
    probe-vs-hand reference image, not the surface one — irrelevant to this question."""
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    ref_b64 = _get_reflux_reference_image_b64()
    extra_images = [(ref_b64, "image/jpeg")] if ref_b64 else None
    parsed, _raw = groq_client.call_vlm_json(
        POSITION_SYSTEM_PROMPT, POSITION_USER_PROMPT, image_b64=img_b64, image_media_type="image/jpeg",
        extra_images=extra_images, reasoning_effort="none", max_tokens=1024,
    )
    return normalize_position(parsed)


# --- Stage B: full leg_side/surface + narrowed leg_level --------------------------------

def _level_system_prompt(allowed_levels: list[str]) -> str:
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
        "WHICH DEVICE IS 'THE PROBE' — do not confuse it with a bare hand. The actual "
        "ultrasound probe is a small handheld device connected by a cable to the "
        "ultrasound machine; during this exam the clinician sometimes uses their OTHER "
        "hand to squeeze/press the calf muscle or grip the cable to provoke/observe "
        "venous reflux — that hand is NOT the probe and must be IGNORED even though it "
        "is often closer to camera or more visually prominent. RELIABLE SHORTCUT, "
        "confirmed to hold across this entire exam's footage: when you see two hands on "
        "the leg at once, at different heights, the HIGHER hand (closer to the top of "
        "the leg/further from the floor) is the one holding the probe, and the LOWER "
        "hand is the compression/cable hand. CRITICAL: recognizing a reflux-testing "
        "scene tells you WHICH HAND to look at — it tells you NOTHING about leg_level. "
        "Reflux testing happens at every level of this exam, thigh included; once "
        "you've identified the higher hand, judge its position exactly the same way you "
        "would with only one hand visible. Do not let recognizing 'this is a reflux "
        "scene' bias leg_level toward calf OR away from it.\n\n"
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
        "shoulders) in THIS frame before concluding the patient turned around.\n\n"
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
        "Work through the facing-direction and leg_side reasoning ONCE, reach a "
        "conclusion, and move on — do not re-litigate the same left/right judgment "
        "back and forth repeatedly. If after one careful pass you are genuinely torn, "
        "answer 'uncertain' rather than continuing to deliberate.\n\n"
        "STRICT reasoning budget: think in at most 6 short sentences total, covering "
        "in order — (1) facing direction, (2) which leg is which, (3) which hand holds "
        "the actual probe device, (4) leg_level using the joint-line landmark check "
        "and the level reference's proportions, (5) surface, (6) done. Your FIRST judgment on each "
        "point is final — do NOT write follow-up sentences starting with 'Wait', "
        "'Actually', 'No,', 'Let me reconsider', 'Hmm', or any other self-correction. "
        "The instant you have all judgments, stop reasoning and output the JSON "
        "immediately — a long internal monologue is a failure, not thoroughness.\n\n"
        "WHEN TO ANSWER 'uncertain': only for a field where the image genuinely gives "
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


LEVEL_USER_PROMPT = (
    "The FIRST image is the actual frame from the webcam video, at the timestamp "
    "matching an ultrasound frame we need to interpret — identify the probe location on "
    "the leg IN THIS FIRST IMAGE. The remaining images are fixed REFERENCE EXAMPLES (not "
    "from this timestamp, not what you are classifying), described in the system "
    "instructions: the SECOND image is the groin-to-ankle leg-level band sequence for "
    "proportion reference. The THIRD image is a two-panel composite with ground-truth "
    "panels for the anterior/medial vs. posterior surface distinction. Use both "
    "reference images only to recognize those patterns if they appear in the first "
    "image."
)

_RETRY_SUFFIX = (
    "\n\nYour previous attempt at this exact image ran out of space mid-reasoning "
    "without ever reaching a JSON answer — you were looping/re-litigating instead of "
    "concluding. This time: reason in 2 sentences maximum (facing direction, then your "
    "combined leg_side/leg_level/surface conclusion), then output ONLY the JSON. Do not "
    "second-guess yourself even once."
)

MAX_TOKENS = 16384  # Groq's hard ceiling for this model's context window — confirmed via a
# real 400 error when 25000 was tried; cannot be raised further.


def _looks_truncated(parsed: dict, raw: str) -> bool:
    """True when the call produced no usable JSON at all (extract_json fell back to {})
    and the raw text is long — i.e. the model spent its whole budget mid-<think> instead
    of a genuinely short/empty response. Distinguishes a truncation failure from a
    legitimate (if rare) empty reply."""
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


def read_level_and_surface(frame_bgr, allowed_levels: list[str]) -> dict:
    """Stage B. Full reasoning (reasoning_effort='default') — confirmed necessary for
    the facing-direction mirroring and surface disambiguation this does; only the
    leg_level VOCABULARY is narrowed here compared to the old single-call design, not
    the reasoning depth."""
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    system = _level_system_prompt(allowed_levels)
    user = LEVEL_USER_PROMPT
    level_ref_b64 = _get_level_reference_image_b64()
    surface_ref_b64 = _get_surface_reference_image_b64()
    extra_images = [im for im in [
        (level_ref_b64, "image/jpeg") if level_ref_b64 else None,
        (surface_ref_b64, "image/jpeg") if surface_ref_b64 else None,
    ] if im] or None
    parsed, raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/jpeg",
                                             extra_images=extra_images,
                                             reasoning_effort="default", max_tokens=MAX_TOKENS)
    if _looks_truncated(parsed, raw):
        parsed, _raw = groq_client.call_vlm_json(system, user + _RETRY_SUFFIX, image_b64=img_b64,
                                                   image_media_type="image/jpeg", extra_images=extra_images,
                                                   reasoning_effort="default", max_tokens=MAX_TOKENS)
    return normalize(parsed, allowed_levels)


# --- Combined entry point ----------------------------------------------------------------

def read_location(frame_bgr) -> dict:
    """Runs Stage A then Stage B (see module docstring). Returns the same shape as the
    old single-call read_location (leg_side/leg_level/surface/confidence/probe_visible/
    visual_evidence) so pipeline.py and stage3_vein_naming.py need no changes, PLUS an
    extra "probe_position_stage_a" field (0/1/"uncertain") for optional UI display of
    the fast first-pass judgment, per the user's request."""
    position = read_probe_position(frame_bgr)
    pos = position["probe_position"]
    if pos == 0:
        allowed_levels = LEVELS_ABOVE_KNEE
    elif pos == 1:
        allowed_levels = LEVELS_AT_OR_BELOW_KNEE
    else:
        # Stage A itself uncertain (or probe not visible) -- don't over-constrain Stage B,
        # let it consider the full vocabulary rather than force a guess into the wrong half.
        allowed_levels = [l for l in anatomy_knowledge.LEG_LEVELS if l != "uncertain"]

    result = read_level_and_surface(frame_bgr, allowed_levels)
    result["probe_position_stage_a"] = pos
    # If Stage A saw the probe but Stage B somehow didn't, prefer Stage A's read on
    # probe_visible -- Stage A ran on a simpler question and is less likely to be wrong
    # about basic visibility.
    if position["probe_visible"] and not result["probe_visible"]:
        result["probe_visible"] = True
    return result
