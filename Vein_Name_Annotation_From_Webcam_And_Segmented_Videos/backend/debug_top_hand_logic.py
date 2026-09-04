"""
Standalone debug script -- NOT wired into the production pipeline.

Tests the specific decision tree proposed by the user:
  STEP 1: ask the VLM how many clinician hands are TOUCHING the leg (0/1/2), and if 2,
          which one is higher on screen (the "top hand").
  STEP 2 (2 hands, top-hand ambiguous only): ask the VLM directly which hand is
          gripping the probe device (device-grip fallback).
  STEP 3 (2 hands): assign leg_level from the identified probe hand's position (fresh
          single call, full 7-level vocabulary, level reference diagram only -- no
          "confirmed answer" reference image, per the anchoring lesson from earlier).
  1 hand: reuse the EXISTING production read_location() pipeline as-is ("current
          logic").
  0 hands: leg_level = "uncertain", no further calls.

Run directly: python debug_top_hand_logic.py
"""
import base64
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import groq_client
import stage3_webcam_location as s3a

FRAMES_DIR = (
    r"C:\Users\Krish\AppData\Local\Temp\claude\c--Users-Krish-Downloads-Cygnus-Med-Demo-"
    r"Vein-Name-Annotation-From-Webcam-And-Segmented-Videos\05d15157-2e6c-4c11-bf8e-"
    r"d1d324c344c3\scratchpad\allframes"
)

# 6 frames, mixing reflux (two-hand) and normal (single/no-hand) content, spanning the
# video -- t=10 (established above-knee), t=29/56/91.5 (reflux window, mixed dodd/calf
# ground truth), t=38.5 (reflux, confirmed medial/calf-boundary earlier), t=118 (late,
# previously logged as calf/medial).
TEST_FRAMES = ["t0010.0.jpg", "t0029.0.jpg", "t0038.5.jpg", "t0056.0.jpg", "t0091.5.jpg", "t0118.0.jpg"]

STEP1_SYSTEM = (
    "You look at a single frame from a webcam video of a clinician performing a leg "
    "venous ultrasound exam. Count how many of the clinician's (gloved) hands are "
    "ACTUALLY TOUCHING the patient's leg right now -- skin contact, or holding the "
    "probe/cable directly against the skin. Do NOT count a hand that is merely visible "
    "but not touching the leg (e.g. resting in the clinician's lap, holding other "
    "equipment). Do not count the patient's own hand.\n\n"
    "If exactly 2 clinician hands are touching the leg: determine which one is higher "
    "on screen (closer to the top of the frame / further from the floor) -- this is "
    "the 'top hand'. Only mark top_hand_determinable=true if one hand is clearly and "
    "unambiguously higher than the other; if they are at a similar height or you "
    "genuinely cannot tell, mark it false.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON:\n"
    '{"num_hands_on_leg": 0|1|2, '
    '"top_hand_determinable": true|false, '
    '"top_hand_desc": "<short phrase: position of the higher hand, empty string if N/A>", '
    '"bottom_hand_desc": "<short phrase: position of the lower hand, empty string if N/A>"}'
)

STEP2_SYSTEM = (
    "You look at a single frame from a webcam video of a clinician performing a leg "
    "venous ultrasound exam. Two of the clinician's hands are touching the leg, and "
    "their relative screen height alone doesn't clearly show which is which. Directly "
    "identify which hand is GRIPPING the ultrasound probe device itself -- a small "
    "handheld device, often white/gray, with a visible cable -- as opposed to a hand "
    "that is flat/open against skin or gripping only the cable.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON:\n"
    '{"probe_hand_desc": "<short phrase: position of the hand gripping the device>", '
    '"confidence": "high"|"medium"|"low"}'
)

STEP3_SYSTEM_TEMPLATE = (
    "You look at a single frame from a webcam video of a clinician performing a leg "
    "venous ultrasound exam. A separate system already identified which hand is "
    "holding the probe: \"{probe_hand_desc}\". Using ONLY that hand's position, assign "
    "the leg_level.\n\n"
    "Choose leg_level from EXACTLY this list (or 'uncertain'): {levels}\n\n"
    "The reference image included (a real leg from this same exam, user-annotated) "
    "shows the FULL groin-to-ankle band sequence in proportion -- measure roughly "
    "where the probe sits along that span and match it to the reference's band "
    "boundaries.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON:\n"
    '{{"leg_level": "<one of the list above>"|"uncertain", "confidence": "high"|"medium"|"low"}}'
)

ALL_LEVELS = [l for l in s3a.anatomy_knowledge.LEG_LEVELS if l != "uncertain"]


def _call(system, user, img_b64, extra_images, label):
    parsed, raw = groq_client.call_vlm_json(
        system, user, image_b64=img_b64, image_media_type="image/jpeg",
        extra_images=extra_images, reasoning_effort="default", max_tokens=8192,
        label=label,
    )
    return parsed


def classify(frame_bgr, tag: str) -> dict:
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()

    step1 = _call(STEP1_SYSTEM, "Classify this frame.", img_b64, None, f"{tag}_step1")
    n = step1.get("num_hands_on_leg")

    if n == 0:
        return {"path": "0-hand", "leg_level": "uncertain", "step1": step1}

    if n == 1:
        # "current logic" = the existing production pipeline, unchanged.
        result = s3a.read_location(frame_bgr)
        return {"path": "1-hand (current logic)", "leg_level": result["leg_level"],
                "surface": result["surface"], "leg_side": result["leg_side"], "step1": step1}

    if n == 2:
        if step1.get("top_hand_determinable"):
            probe_hand_desc = step1.get("top_hand_desc") or ""
            method = "top-hand heuristic"
        else:
            step2 = _call(STEP2_SYSTEM, "Classify this frame.", img_b64, None, f"{tag}_step2")
            probe_hand_desc = step2.get("probe_hand_desc") or ""
            method = "device-grip fallback"

        level_ref_b64 = s3a._get_level_reference_image_b64()
        extra_images = [(level_ref_b64, "image/jpeg")] if level_ref_b64 else None
        step3_system = STEP3_SYSTEM_TEMPLATE.format(probe_hand_desc=probe_hand_desc, levels=", ".join(ALL_LEVELS))
        step3 = _call(step3_system, "Classify this frame.", img_b64, extra_images, f"{tag}_step3")
        return {"path": f"2-hand ({method})", "leg_level": step3.get("leg_level") or "uncertain",
                "probe_hand_desc": probe_hand_desc, "step1": step1}

    return {"path": "unclear", "leg_level": "uncertain", "step1": step1}


def main():
    for fname in TEST_FRAMES:
        path = os.path.join(FRAMES_DIR, fname)
        frame = cv2.imread(path)
        if frame is None:
            print(f"{fname}: FRAME NOT FOUND")
            continue
        result = classify(frame, tag=fname.replace(".", "_"))
        print(f"\n=== {fname} ===")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
