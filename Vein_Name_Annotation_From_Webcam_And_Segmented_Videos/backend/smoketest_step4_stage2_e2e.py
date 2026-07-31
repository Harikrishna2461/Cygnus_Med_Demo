"""
Build step 5: Stage 2 end-to-end on one static frame using the real module (not the ad-hoc
smoketest_step2_groq.py prompt) — real segmentation -> real blobs -> real Groq N1/N2/N3 call.
Run: py -3.12 smoketest_step4_stage2_e2e.py
"""
import os
import cv2

import config
import biomedparse_engine as bpe
import stage2_fascia_classify as s2
import renderer

TEST_IMAGE = os.path.join(
    config.TASK4_DIR, "Task_4_Test_Images", "Seen_Frames", "Screenshot 2026-07-16 101239.png"
)


def main():
    frame = cv2.imread(TEST_IMAGE)
    blobs, fascia = bpe.segment_frame(frame)
    print(f"[smoketest] segmented {len(blobs)} blob(s)")

    s2.classify_blobs(frame, blobs, fascia)

    for b in blobs:
        print(f"  blob {b.blob_id}: n_class={b.n_class}  reasoning={b.n_class_reasoning}")
        assert b.n_class in ("N1", "N2", "N3"), f"blob {b.blob_id} was not classified"

    annotated = renderer.draw_intermediate_frame(frame, blobs, fascia)
    out_path = os.path.join(config.OUTPUTS_DIR, "smoketest_step4_intermediate.png")
    cv2.imwrite(out_path, annotated)
    print(f"[smoketest] wrote {out_path}")
    print("[smoketest] OK")


if __name__ == "__main__":
    main()
