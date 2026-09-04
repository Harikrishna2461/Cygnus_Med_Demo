"""
Standalone debug script -- NOT wired into the production pipeline.

Isolated test: for each frame, ask the VLM to identify the clinician's two hands (no
pre-drawn circles this time -- it has to find them itself), determine which one is
actually gripping the probe device, and report each hand's leg-level position. Output
schema: {"hand1_posn": "...", "hand2_posn": "...", "probe_hand": "hand1"|"hand2"}.

Run directly: python debug_two_hand_level.py
Optionally point FRAMES_DIR / edit the timestamp list below at a different frame set.
"""
import base64
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import groq_client

FRAMES_DIR = (
    r"C:\Users\Krish\AppData\Local\Temp\claude\c--Users-Krish-Downloads-Cygnus-Med-Demo-"
    r"Vein-Name-Annotation-From-Webcam-And-Segmented-Videos\05d15157-2e6c-4c11-bf8e-"
    r"d1d324c344c3\scratchpad\allframes"
)

# The reflux-window frames the user flagged (t=24s to t=46s) -- whatever actually exists
# on disk in that range, picked up by glob rather than hardcoding filenames that might
# not match the extracted set exactly.
TIMESTAMPS = [24.0, 29.0, 31.0, 33.5, 38.5, 43.5, 46.0]


def _frame_path(ts: float) -> str | None:
    candidates = glob.glob(os.path.join(FRAMES_DIR, f"t{ts:06.1f}.jpg"))
    return candidates[0] if candidates else None


SYSTEM = (
    "You are looking at a single photo from a webcam video of a clinician performing a "
    "leg venous ultrasound exam. Identify the TWO hands belonging to the clinician "
    "(gloved) that are visible in the frame -- if only one is visible, still fill in "
    "hand1 and leave hand2 as 'not visible'. Do not confuse the patient's own hand "
    "(never gloved, often near the top of frame holding clothing) with the clinician's.\n\n"
    "For EACH clinician hand you find, determine its position on the leg -- leg level "
    "(e.g. groin, upper thigh, mid/distal thigh, knee, upper calf, calf, ankle) and, if "
    "you can tell, roughly anterior/medial/posterior surface.\n\n"
    "Then determine which of the two hands is the one actually GRIPPING the ultrasound "
    "probe device itself -- a small handheld device, often white/gray, with a visible "
    "cable coming out of it -- as opposed to a hand that is flat/open against the skin "
    "(compressing the calf for a reflux test) or gripping only the cable.\n\n"
    "IMPORTANT: do not mistake a rounded muscle bulge on the thigh for the knee joint. "
    "The real kneecap is a firmer, flatter landmark where the leg's contour narrows "
    "before hinging into the shin -- if you're not certain exactly where the knee is, "
    "say so at lower confidence rather than asserting a level you're not sure of.\n\n"
    "Respond with ONLY a compact JSON object, no markdown, no prose outside the JSON:\n"
    '{"hand1_posn": "<leg level / position of one clinician hand>", '
    '"hand2_posn": "<leg level / position of the other clinician hand, or '
    '\'not visible\'>", '
    '"probe_hand": "hand1" or "hand2", '
    '"confidence": "high"|"medium"|"low"}'
)
USER = "Identify the two hands and answer now."

_RETRY_SUFFIX = (
    "\n\nYour previous attempt at this exact image ran out of space mid-reasoning "
    "without ever reaching a JSON answer. This time: reason in at most 3 short "
    "sentences (which hand grips the probe, then each hand's leg level), then output "
    "ONLY the JSON. Do not second-guess yourself repeatedly."
)
MAX_TOKENS = 16384  # same hard ceiling as the production Stage B call


def classify_frame(frame_bgr, label: str) -> dict:
    _, buf = cv2.imencode(".jpg", frame_bgr)
    img_b64 = base64.b64encode(buf).decode()
    parsed, raw = groq_client.call_vlm_json(
        SYSTEM, USER, image_b64=img_b64, image_media_type="image/jpeg",
        reasoning_effort="default", max_tokens=MAX_TOKENS, label=label,
    )
    if not parsed and len(raw) > 2000:
        # Truncated mid-reasoning without ever reaching JSON -- same failure mode
        # already confirmed/handled in the production stage3_webcam_location.py.
        parsed, _raw = groq_client.call_vlm_json(
            SYSTEM, USER + _RETRY_SUFFIX, image_b64=img_b64, image_media_type="image/jpeg",
            reasoning_effort="default", max_tokens=MAX_TOKENS, label=f"{label}_retry",
        )
    return parsed


def main():
    results = {}
    for ts in TIMESTAMPS:
        path = _frame_path(ts)
        if not path:
            print(f"t={ts}: NO FRAME FOUND, skipping")
            continue
        frame = cv2.imread(path)
        for attempt in range(3):
            try:
                parsed = classify_frame(frame, label=f"two_hand_level_t{ts}")
                break
            except Exception as exc:
                print(f"t={ts}: error ({exc}), retrying in 20s...")
                time.sleep(20)
        else:
            parsed = {}
        results[ts] = parsed
        print(f"t={ts}: {json.dumps(parsed)}")
        time.sleep(15)  # stay well under Groq's output-tokens-per-minute cap

    print("\n=== summary ===")
    for ts, parsed in results.items():
        print(f"t={ts}: hand1={parsed.get('hand1_posn')!r} hand2={parsed.get('hand2_posn')!r} "
              f"probe_hand={parsed.get('probe_hand')!r} confidence={parsed.get('confidence')!r}")


if __name__ == "__main__":
    main()
