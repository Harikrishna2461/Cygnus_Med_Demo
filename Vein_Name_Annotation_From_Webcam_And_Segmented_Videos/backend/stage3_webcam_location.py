"""Stage 3a: look at one webcam frame and describe where the probe is on the leg."""
import base64
import os

import cv2

import anatomy_knowledge
import groq_client

_REF_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reference_images", "reflux_probe_vs_compression_hand.jpg")
_ref_image_b64 = None


def _get_reflux_reference_image_b64() -> str | None:
    """Loaded once, reused on every call — a real annotated frame from this project's own
    footage (not a stock/generic image) showing the exact probe-vs-compression-hand
    confusion pattern the user flagged: green circle = the hand actually holding the
    cabled probe device (the real scan location), red circle = a second hand lower on the
    leg gripping the cable / compressing the calf for a reflux test, which is NOT the
    probe location. Kept small (640px wide) to limit added latency/tokens per call —
    confirmed this failure mode kept recurring with text-only instructions across two
    prompt-strengthening attempts, so a concrete visual anchor is now load-bearing, not
    optional polish."""
    global _ref_image_b64
    if _ref_image_b64 is None and os.path.exists(_REF_IMAGE_PATH):
        with open(_REF_IMAGE_PATH, "rb") as f:
            _ref_image_b64 = base64.b64encode(f.read()).decode()
    return _ref_image_b64

SYSTEM_PROMPT = (
    "You look at a single frame from a webcam video of a clinician performing a leg "
    "venous ultrasound exam. Your job is ONLY to describe where the ultrasound probe is "
    "touching the patient's leg — you are not diagnosing anything.\n\n"
    "Use these leg-level names when you can tell: "
    + ", ".join(l for l in anatomy_knowledge.LEG_LEVELS if l != "uncertain") + ".\n"
    "Landmarks: the groin crease (top of thigh), the knee, the medial and lateral "
    "malleoli (ankle bones), the popliteal fossa (back of the knee).\n\n"
    "The thigh has three levels in that list (upper_thigh, proximal_thigh_hunterian, "
    "distal_thigh_dodd) and they have no distinct visible landmark between them — judge "
    "by roughly what FRACTION of the way down the thigh (groin to knee) the probe is: "
    "top third (nearest the groin) = upper_thigh; middle third = proximal_thigh_hunterian; "
    "bottom third (nearest the knee) = distal_thigh_dodd. Do not skip straight from "
    "groin_sfj or upper_thigh to distal_thigh_dodd without considering whether the probe "
    "position actually looks closer to the middle third first.\n\n"
    "WHICH DEVICE IS 'THE PROBE' — do not confuse it with a bare hand. See the labeled "
    "REFERENCE EXAMPLE IMAGE included with this message (a real frame from this same "
    "kind of exam): the GREEN circle marks the hand actually holding the probe device "
    "against the skin — a small handheld device with a visible cable running from it — "
    "that is the TRUE scan location. The RED circle marks a second hand, LOWER on the "
    "leg (nearer the calf), gripping the cable and/or compressing the calf muscle to "
    "provoke venous reflux — that hand's location is NOT the probe location and must be "
    "IGNORED even though it is often closer to camera, more visually prominent, or "
    "moving more than the actual scanning hand.\n"
    "General rule for the CURRENT frame you are asked to classify (not the reference "
    "example): The actual ultrasound probe is a small handheld device (often wand/"
    "rectangular shaped, sometimes with a model label printed on it) connected by a "
    "cable to the ultrasound machine. During a venous exam the clinician sometimes uses "
    "their OTHER hand to squeeze/press the calf muscle or grip the cable to provoke/"
    "observe reflux — exactly like the red-circled hand in the reference example. When "
    "you see two hands on the leg at once, at DIFFERENT heights along the leg, find the "
    "one actually holding/touching the device itself (not just the cable) against the "
    "skin — that hand's location is the answer, regardless of which hand is lower, "
    "closer to camera, or more visually prominent. If you truly cannot tell which hand "
    "holds the device, say so in visual_evidence and lower your confidence, but still "
    "prefer the hand that looks like it's holding a rigid object over one that looks "
    "like bare fingers/cable gripping alone.\n\n"
    "DISAMBIGUATING REGION BORDERS (knee vs. thigh vs. calf) — a common failure mode is "
    "jumping to the wrong adjacent region when the probe is near a joint:\n"
    "- The kneecap (patella, seen from the front) and the crease behind the knee "
    "(popliteal fossa, seen from the back) mark the KNEE JOINT LINE.\n"
    "- If the probe is on or straddling that joint line (from either the front or the "
    "back), the answer is knee_popliteal — not distal_thigh_dodd and not calf.\n"
    "- distal_thigh_dodd means the probe is CLEARLY above the joint line (roughly a "
    "hand's-width or more up), still on the fleshy thigh, kneecap/fossa visibly below "
    "the probe.\n"
    "- calf means the probe is CLEARLY below the joint line (roughly a hand's-width or "
    "more down), on the lower leg's muscle bulk, kneecap/fossa visibly above the probe.\n"
    "- Before choosing distal_thigh_dodd or calf, actively look for the kneecap/fossa "
    "landmark and judge distance from it — do not let an oblique camera angle push you "
    "toward the wrong side of the joint just because the landmark isn't centered in "
    "frame.\n\n"
    "HOW TO DETERMINE leg_side (left/right) — this is easy to get backwards, reason "
    "through it explicitly every time, do not default to a naive screen-left=patient's-"
    "left mapping:\n"
    "1. First decide the patient's orientation relative to the camera: is their FRONT "
    "facing the camera, their BACK facing the camera (cue: you see their back/shoulder "
    "blades/back of the head, not their face), or a side profile?\n"
    "2. If the patient's FRONT faces the camera: this is the same as facing another "
    "person — their body is mirrored relative to the image. The leg on the LEFT side "
    "of the image is the PATIENT'S OWN RIGHT leg; the leg on the RIGHT side of the "
    "image is the PATIENT'S OWN LEFT leg.\n"
    "3. If the patient's BACK faces the camera: no mirroring. The leg on the LEFT side "
    "of the image is the patient's own LEFT leg; the leg on the RIGHT side of the "
    "image is the patient's own RIGHT leg.\n"
    "4. If you cannot tell which way the patient is facing, answer leg_side='uncertain' "
    "rather than guessing — never default to assuming the image is unmirrored.\n"
    "5. IMPORTANT: scanning the popliteal fossa (back of the knee) does NOT by itself "
    "mean the patient turned their whole body around. A clinician can access the back "
    "of the knee with only a small weight-shift or knee bend while the patient's overall "
    "stance and facing direction stay the same as moments before. Do not re-derive "
    "facing direction from scratch just because the surface being scanned is posterior "
    "— check the actual visible cues (face, back of head/shoulders) in THIS frame "
    "before concluding the patient turned around.\n\n"
    "HOW TO DETERMINE surface (anterior/medial/posterior/lateral) of the leg being "
    "scanned — this distinction is clinically significant (medial vs. posterior "
    "corresponds to genuinely different veins), so get it right rather than defaulting "
    "to whichever sounds more dramatic:\n"
    "- medial: the probe is on the INNER side of the leg, toward the midline / the "
    "other leg. This is by far the MOST COMMON surface in this kind of exam (most GSV "
    "scanning is medial) — you will typically see the probe angled toward or between "
    "both legs, gel visible on the inner thigh/calf, and the leg's front-inner muscle "
    "bulk (no crease/fold) under and around the probe.\n"
    "- posterior: REQUIRES a specific landmark — either the popliteal fossa (the crease/"
    "fold directly behind the knee joint) is visibly where the probe is, OR you can "
    "clearly see the patient's buttocks/back of both thighs together (genuine back-of-"
    "body view). An oblique camera angle, the clinician reaching across, or partial hip/"
    "thigh curvature at the top edge of frame are NOT enough on their own — do not choose "
    "posterior just because the view feels like it's 'from behind'; actively look for "
    "the popliteal crease or genuine buttocks visibility first. If you can't specifically "
    "point to one of those two landmarks, prefer medial over posterior. A common, "
    "confirmed mistake: claiming 'buttocks visible' on a frame that actually shows a "
    "front-facing thigh with a kneecap and gel — the REFERENCE EXAMPLE IMAGE included "
    "with this message is exactly that scene (its own caption confirms it is MEDIAL, "
    "front/inner thigh, not posterior) — if the frame you're classifying looks similar "
    "(standing leg, front-inner thigh, kneecap and/or gel visible, clinician reaching in "
    "from the side), that pattern is medial, not posterior, even at an oblique angle.\n"
    "- anterior: the patient's front faces the camera and the probe is on the "
    "front-facing surface of the leg, toward the camera (shin/front-of-thigh, no medial "
    "inner-thigh angling toward the other leg).\n"
    "- lateral: the probe is on the OUTER side of the leg, away from the other leg.\n"
    "medial/lateral do NOT flip with facing direction (medial always means toward the "
    "body's midline); anterior/posterior DO depend on whether you're seeing the front "
    "or back of the patient's body.\n\n"
    "USING A PREVIOUS READING, IF ONE IS GIVEN (see the user message): a webcam video is "
    "continuous — the patient, camera, and clinician do not teleport between two "
    "readings a few seconds apart. Treat any previous reading you're given as a starting "
    "prior, not a fact to ignore, and not a fact to blindly copy either:\n"
    "- leg_side essentially never changes within a short continuous clip unless you can "
    "actually see the clinician move to the leg's other side or the patient's whole body "
    "visibly reposition. Do not flip leg_side just because the patient bent a knee, "
    "shifted slightly, or the probe moved onto a posterior surface (see point 5 above). "
    "If this frame looks like a continuation of the same scan, keep the previous "
    "leg_side unless you see clear, specific contrary evidence.\n"
    "- leg_level should move smoothly along the leg. If the previous reading was e.g. "
    "distal_thigh_dodd and this frame looks broadly similar, prefer that level or an "
    "immediately adjacent one (proximal_thigh_hunterian or knee_popliteal) over a "
    "distant jump (e.g. ankle) unless the image clearly shows the probe has moved a long "
    "way since then.\n"
    "- Still, always prioritize what you actually SEE in THIS frame over the previous "
    "reading when they genuinely conflict — the previous reading is a helpful prior for "
    "resolving ambiguity, never a hard constraint that overrides clear visual evidence.\n"
    "- CRITICAL — in this kind of exam footage it is VERY COMMON for the camera to be "
    "cropped tight on the leg(s) with the patient's face, head, and torso NOT visible "
    "anywhere in frame (you may see only leg(s), the probe, gloved hands/arms, and "
    "clothing at the very top edge). When that's the case, you have ZERO evidence of "
    "facing direction — none. Do not guess it from weak/indirect cues (toe angle, "
    "clothing wrinkles, which way a knee is bent) and do not treat 'the back of the knee "
    "is visible' as proof the patient turned around (see point 5 above — it usually just "
    "means the clinician repositioned their own view of the SAME leg).\n"
    "- Even so, NEVER simply copy a previous reading without actively checking THIS "
    "frame for it first — you must properly evaluate every image on its own evidence, "
    "every time. Before relying on a given previous leg_side, actively check THIS frame "
    "for continuity evidence: does the visible skin tone/build, clothing, probe/cable, "
    "clinician, and room background look like a direct continuation of the same person "
    "and setup? If yes, and facing direction genuinely isn't visible, the previous "
    "leg_side is your best-supported answer (say so in visual_evidence). If instead "
    "anything looks discontinuous — different clothing, different visible body type, a "
    "different pose (e.g. now seated when the prior reading was standing), a different "
    "person's face now visible, or the probe sitting unused in a stand rather than "
    "touching skin — that is real evidence of a change, and you must NOT carry the "
    "previous leg_side forward; reason fresh from whatever this frame actually shows, "
    "and answer 'uncertain' if that frame alone doesn't resolve it. The goal is to stop "
    "leg_side from flip-flopping on a single continuous, unmoving exam (which is a real "
    "past failure of this system) WITHOUT blindly freezing an old answer once the scene "
    "has genuinely moved on to something new.\n\n"
    "Work through the facing-direction and leg_side reasoning ONCE, reach a conclusion, "
    "and move on — do not re-litigate the same left/right judgment back and forth "
    "repeatedly. If after one careful pass you are genuinely torn, answer 'uncertain' "
    "rather than continuing to deliberate.\n\n"
    "STRICT reasoning budget: think in at most 6 short sentences total, covering in "
    "order — (1) facing direction, (2) which leg is which (using the previous reading as "
    "a prior if given), (3) which hand holds the actual probe device, (4) leg_level "
    "using the joint-line landmark check, (5) surface, (6) done. Your FIRST judgment on "
    "each point is final — do NOT write follow-up sentences starting with 'Wait', "
    "'Actually', 'No,', 'Let me reconsider', 'Hmm', or any other self-correction. The "
    "instant you have all judgments, stop reasoning and output the JSON immediately — a "
    "long internal monologue is a failure, not thoroughness.\n\n"
    "WHEN TO ANSWER 'uncertain': only for a field where the image genuinely gives you no "
    "usable evidence — e.g. the probe/leg is completely out of frame, fully hidden, or "
    "the frame itself is unusable (blank/blurred beyond recognition). If the probe and "
    "leg ARE visible but the view is merely awkward, oblique, or partially occluded "
    "(clothing, the other hand, motion blur), do not default straight to 'uncertain' — "
    "make your best-supported judgment (using the previous reading as a prior if it "
    "helps) at 'low' or 'medium' confidence instead. A low-confidence real answer "
    "grounded in partial evidence is more useful than a blank 'uncertain'; reserve "
    "'uncertain' for when you truly have nothing to go on, not merely when you're not "
    "fully sure. Respond with ONLY a compact JSON object, no markdown, no prose outside "
    "the JSON, in exactly this shape:\n"
    '{"leg_side": "left"|"right"|"uncertain", '
    '"leg_level": "<one of the leg-level names above>"|"uncertain", '
    '"surface": "anterior"|"medial"|"posterior"|"lateral"|"uncertain", '
    '"confidence": "high"|"medium"|"low", '
    '"probe_visible": true|false, '
    '"visual_evidence": "<one sentence: what landmarks/cues you actually used, '
    'INCLUDING which way the patient is facing and how that determined leg_side>"}'
)

USER_PROMPT_BASE = (
    "The FIRST image is the actual frame from the webcam video, at the timestamp "
    "matching an ultrasound frame we need to interpret — identify the probe location on "
    "the leg IN THIS FIRST IMAGE. The SECOND image is a fixed labeled REFERENCE EXAMPLE "
    "(not from this timestamp, not what you are classifying) showing the probe-vs-"
    "compression-hand pattern described in the system instructions — use it only to "
    "recognize that pattern if it appears in the first image."
)


def build_prompt(previous_location: dict = None) -> tuple[str, str]:
    """Pure — no image/model/network. previous_location, if given, is the last non-empty
    reading from a few seconds earlier in the SAME video — folded into the user message
    (not the system prompt) since it's per-call data, not fixed instructions. Only the
    fields that matter for continuity are included, and only when they're not themselves
    'uncertain' (an uncertain previous reading is not a useful prior)."""
    user = USER_PROMPT_BASE
    if previous_location:
        useful = {k: previous_location.get(k) for k in ("leg_side", "leg_level", "surface")
                  if previous_location.get(k) and previous_location.get(k) != "uncertain"}
        if useful:
            user += (
                "\n\nPREVIOUS READING (from a few seconds earlier in this same continuous "
                f"video, use as a prior per the system instructions): {useful}"
            )
    return SYSTEM_PROMPT, user


def normalize(parsed: dict) -> dict:
    """Fills safe 'uncertain'/'low' defaults so downstream code never KeyErrors on a
    malformed or empty VLM response — this is the system-boundary validation point."""
    return {
        "leg_side": parsed.get("leg_side") or "uncertain",
        "leg_level": parsed.get("leg_level") or "uncertain",
        "surface": parsed.get("surface") or "uncertain",
        "confidence": parsed.get("confidence") or "low",
        "probe_visible": bool(parsed.get("probe_visible", False)),
        "visual_evidence": parsed.get("visual_evidence") or "",
    }


MAX_TOKENS = 16384  # Groq's hard ceiling for this model's context window — confirmed via a
# real 400 error when 25000 was tried; cannot be raised further.

_RETRY_SUFFIX = (
    "\n\nYour previous attempt at this exact image ran out of space mid-reasoning "
    "without ever reaching a JSON answer — you were looping/re-litigating instead of "
    "concluding. This time: reason in 2 sentences maximum (facing direction, then your "
    "combined leg_side/leg_level/surface conclusion), then output ONLY the JSON. Do not "
    "second-guess yourself even once."
)


def _looks_truncated(parsed: dict, raw: str) -> bool:
    """True when the call produced no usable JSON at all (extract_json fell back to {})
    and the raw text is long — i.e. the model spent its whole budget mid-<think> instead
    of a genuinely short/empty response. Distinguishes a truncation failure from a
    legitimate (if rare) empty reply."""
    return not parsed and len(raw) > 2000


def read_location(frame_bgr, previous_location: dict = None) -> dict:
    """previous_location: the last non-empty reading from earlier in the same video, used
    as a stabilizing prior (see build_prompt) — confirmed empirically necessary: on a real
    clip, four consecutive readings 8 seconds apart of a near-static real-world scene (the
    probe barely moved) swung wildly between calf/knee_popliteal/ankle and flipped
    leg_side twice, purely because each call reasons from zero with no memory of the last
    answer. The model still makes the final call every time — this only gives it context,
    it does not decide anything in Python."""
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    system, user = build_prompt(previous_location)
    # reasoning_effort="default" (full chain-of-thought, not the fast/no-think mode used
    # elsewhere) — confirmed necessary on real footage: the facing-direction mirroring
    # this prompt asks for is genuinely multi-step spatial reasoning, and with reasoning
    # off the same near-identical frame got mirrored correctly on some ticks and backwards
    # on others. This call is less frequent than segmentation/naming (gated on
    # WEBCAM_LOCATION_MIN_INTERVAL_SEC) so the added latency per call matters less here.
    #
    # max_tokens set to Groq's actual ceiling (16384) — confirmed empirically that even
    # this max isn't always enough on its own: this model can loop/re-litigate its
    # left-right judgment indefinitely regardless of budget (one real frame burned 60K+
    # chars of "Wait, actually..." without ever reaching JSON). The real fix is the
    # anti-looping instructions in SYSTEM_PROMPT (ban self-correction phrasing, hard
    # sentence budget) plus this retry as a safety net — NOT just raising max_tokens.
    ref_b64 = _get_reflux_reference_image_b64()
    extra_images = [(ref_b64, "image/jpeg")] if ref_b64 else None
    parsed, raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/jpeg",
                                             extra_images=extra_images,
                                             reasoning_effort="default", max_tokens=MAX_TOKENS)
    if _looks_truncated(parsed, raw):
        parsed, _raw = groq_client.call_vlm_json(system, user + _RETRY_SUFFIX, image_b64=img_b64,
                                                   image_media_type="image/jpeg", extra_images=extra_images,
                                                   reasoning_effort="default", max_tokens=MAX_TOKENS)
    return normalize(parsed)
