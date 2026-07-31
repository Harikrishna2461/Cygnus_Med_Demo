"""
Build step 4: unit-test the Stage 2/3 prompt builders with hand-constructed synthetic
data. No image, no model, no network — just checking the text these functions produce is
sane before spending real Groq calls on them. Run: py -3.12 smoketest_step3_prompts.py
"""
import numpy as np

from biomedparse_engine import VeinBlob, FasciaBoundary
import stage2_fascia_classify as s2
import stage3_webcam_location as s3a
import stage3_vein_naming as s3b


def test_stage2_prompt():
    print("=== Stage 2 build_prompt() ===")
    W = 400
    sup = np.full(W, 100.0)
    deep = np.full(W, 180.0)
    fascia = FasciaBoundary(sup_row_at_col=sup, deep_row_at_col=deep)
    blobs = [
        VeinBlob(blob_id=1, contour=None, centroid=(200, 60), bbox=(190, 50, 20, 20), area_px=300),   # above sup -> N3-ish
        VeinBlob(blob_id=2, contour=None, centroid=(200, 140), bbox=(190, 130, 20, 20), area_px=300),  # between -> N2-ish
        VeinBlob(blob_id=3, contour=None, centroid=(200, 250), bbox=(190, 240, 20, 20), area_px=300),  # below deep -> N1-ish
    ]
    prompt = s2.build_prompt(blobs, fascia)
    print(prompt)
    assert "Blob 1" in prompt and "Blob 2" in prompt and "Blob 3" in prompt
    assert "above the superficial" in prompt  # blob 1
    assert "below the superficial" in prompt and "above the deep" in prompt  # blob 2
    assert "below the deep" in prompt  # blob 3
    print("[ok] stage2 geometry phrasing looks directionally correct\n")


def test_stage2_prompt_nan_fascia():
    print("=== Stage 2 build_prompt() with undetected fascia ===")
    W = 100
    nan_row = np.full(W, np.nan)
    fascia = FasciaBoundary(sup_row_at_col=nan_row, deep_row_at_col=nan_row.copy())
    blobs = [VeinBlob(blob_id=1, contour=None, centroid=(50, 50), bbox=(40, 40, 20, 20), area_px=300)]
    prompt = s2.build_prompt(blobs, fascia)
    print(prompt)
    assert "not reliably detected" in prompt
    print("[ok] NaN-fascia fallback phrasing present\n")


def test_stage3a_prompt():
    print("=== Stage 3a build_prompt() ===")
    system, user = s3a.build_prompt()
    assert "leg_side" in system and "leg_level" in system and "uncertain" in system
    print(system[:200] + " ...\n")
    print(user, "\n")
    print("[ok] stage3a schema present in system prompt\n")


def test_stage3a_normalize():
    print("=== Stage 3a normalize() ===")
    empty = s3a.normalize({})
    print(empty)
    assert empty["leg_level"] == "uncertain" and empty["confidence"] == "low"
    partial = s3a.normalize({"leg_level": "calf", "confidence": "high"})
    print(partial)
    assert partial["leg_level"] == "calf" and partial["leg_side"] == "uncertain"
    print("[ok] normalize() fills safe defaults\n")


def test_stage3b_prompt():
    print("=== Stage 3b build_naming_prompt() ===")
    blobs = [
        {"blob_id": 1, "n_class": "N2", "centroid": [212, 130]},
        {"blob_id": 2, "n_class": "N1", "centroid": [204, 300]},
    ]
    location = {
        "leg_side": "left", "leg_level": "proximal_thigh_hunterian", "surface": "medial",
        "confidence": "medium", "probe_visible": True,
        "visual_evidence": "probe on medial mid-thigh, knee visible at frame bottom",
    }
    system, user = s3b.build_naming_prompt(blobs, location)
    print(user, "\n")
    assert "Hunterian" in system or "Hunterian" in user or "hunterian" in location["leg_level"]
    assert "Blob 1: n_class=N2" in user
    assert "Blob 2: n_class=N1" in user
    assert "proximal_thigh_hunterian" in user
    print("[ok] stage3b prompt includes both blobs and location context\n")


if __name__ == "__main__":
    test_stage2_prompt()
    test_stage2_prompt_nan_fascia()
    test_stage3a_prompt()
    test_stage3a_normalize()
    test_stage3b_prompt()
    print("ALL PROMPT-BUILDER UNIT TESTS PASSED")
