"""
Build step 6: Stage 3 end-to-end. 3a reads a real (cropped) webcam frame for probe
location; 3b combines that with the real Stage-2 output from smoketest_step4 (or a fresh
run) to name the veins. Uses the actual modules, real Groq calls.
Run: py -3.12 smoketest_step5_stage3_e2e.py
"""
import os
import cv2

import config
import biomedparse_engine as bpe
import stage2_fascia_classify as s2
import stage3_webcam_location as s3a
import stage3_vein_naming as s3b
import renderer

TEST_ULTRASOUND = os.path.join(
    config.TASK4_DIR, "Task_4_Test_Images", "Seen_Frames", "Screenshot 2026-07-16 101239.png"
)
TEST_WEBCAM = os.path.join(config.OUTPUTS_DIR, "webcam_inset_crop.png")


def main():
    # --- Stage 3a: real cropped webcam-inset frame from Miscellaneous/Session_Recordings ---
    webcam_frame = cv2.imread(TEST_WEBCAM)
    print(f"[smoketest] webcam frame shape: {webcam_frame.shape}")
    location = s3a.read_location(webcam_frame)
    print("[smoketest] Stage 3a location result:")
    for k, v in location.items():
        print(f"    {k}: {v}")

    # --- Stage 2, to get real N-classed blobs to feed Stage 3b ---
    us_frame = cv2.imread(TEST_ULTRASOUND)
    blobs, fascia = bpe.segment_frame(us_frame)
    s2.classify_blobs(us_frame, blobs, fascia)
    print(f"[smoketest] Stage 2 produced {len(blobs)} classified blob(s): "
          + ", ".join(f"{b.blob_id}={b.n_class}" for b in blobs))

    # --- Stage 3b: name the veins using 3a's location + Stage 2's N-classes ---
    blob_dicts = [{"blob_id": b.blob_id, "n_class": b.n_class, "centroid": list(b.centroid)} for b in blobs]
    annotated = renderer.draw_intermediate_frame(us_frame, blobs, fascia)
    names = s3b.name_veins(blob_dicts, location, annotated_ultrasound_frame_bgr=annotated)
    print("[smoketest] Stage 3b vein names:")
    for bid, info in names.items():
        print(f"    blob {bid}: {info['vein_name']}  ({info['reasoning']})")
    assert names, "Stage 3b returned no names at all — check the raw response"

    final = renderer.draw_final_frame(us_frame, blobs, fascia, {bid: v["vein_name"] for bid, v in names.items()})
    out_path = os.path.join(config.OUTPUTS_DIR, "smoketest_step5_final.png")
    cv2.imwrite(out_path, final)
    print(f"[smoketest] wrote {out_path}")
    print("[smoketest] OK")


if __name__ == "__main__":
    main()
