"""
Build step 3 smoke test: one real Groq call against the annotated image produced by
smoketest_step1_segmentation.py. Confirms API key, model id, <think>-stripping, and JSON
parsing all work before any prompt-builder logic is written. Run after step 1's smoketest.
"""
import base64
import os

import config
import groq_client

IMG_PATH = os.path.join(config.OUTPUTS_DIR, "smoketest_step1_annotated.png")

SYSTEM = (
    "You read annotated venous ultrasound frames. Two curved lines are drawn: a YELLOW "
    "line (superficial fascia boundary) and an ORANGE line (deep fascia boundary). One or "
    "more numbered GREEN contours mark candidate vein lumens. Reason about each numbered "
    "contour's position relative to the two lines, then respond with ONLY a compact JSON "
    "object, no markdown, no prose outside the JSON."
)
USER = (
    "Blob 1: contour sits with its centroid approximately at the yellow (superficial) line, "
    "spanning from just above it to below it, well above the orange (deep) line.\n\n"
    "Classify blob 1 as one of:\n"
    '  "N1" = entirely below the orange (deep) line (deep vein)\n'
    '  "N2" = at or between the two lines (saphenous trunk in its fascial compartment)\n'
    '  "N3" = entirely above the yellow (superficial) line (superficial tributary)\n\n'
    'Respond as JSON: {"1": {"n_class": "N1"|"N2"|"N3", "reasoning": "<one sentence>"}}'
)


def main():
    with open(IMG_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    print(f"[smoketest] calling Groq model={config.GROQ_VLM_MODEL} ...")
    parsed, raw = groq_client.call_vlm_json(SYSTEM, USER, image_b64=img_b64, image_media_type="image/png")

    print("\n--- RAW RESPONSE ---")
    print(raw)
    print("\n--- PARSED JSON ---")
    print(parsed)

    if not parsed:
        raise SystemExit("[smoketest] FAILED: no JSON object could be extracted from the response")
    print("\n[smoketest] OK: got a parseable JSON response.")


if __name__ == "__main__":
    main()
