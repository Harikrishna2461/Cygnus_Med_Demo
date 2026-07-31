"""
Build step 1+2 smoke test: load both BioMedParse checkpoints, segment one static
ultrasound frame, and render the numbered-blob + fascia-line overlay for eyeballing.
No Flask, no video, no Groq. Run: py -3.12 smoketest_step1_segmentation.py
"""
import os
import time
import cv2

import config
import biomedparse_engine as bpe
import renderer

TEST_IMAGE = os.path.join(
    config.TASK4_DIR, "Task_4_Test_Images", "Seen_Frames", "Screenshot 2026-07-16 101239.png"
)


def main():
    print(f"[smoketest] reading test image: {TEST_IMAGE}")
    frame = cv2.imread(TEST_IMAGE)
    if frame is None:
        raise SystemExit(f"Could not read {TEST_IMAGE}")
    print(f"[smoketest] frame shape: {frame.shape}")

    t0 = time.time()
    blobs, fascia = bpe.segment_frame(frame)
    dt = time.time() - t0
    print(f"[smoketest] segment_frame took {dt:.2f}s")
    print(f"[smoketest] found {len(blobs)} vein blob(s)")
    for b in blobs:
        print(f"  blob {b.blob_id}: centroid={b.centroid}, area_px={b.area_px}, bbox={b.bbox}")

    import numpy as np
    n_valid_fascia_cols = int(np.sum(~np.isnan(fascia.sup_row_at_col)))
    print(f"[smoketest] fascia valid columns: {n_valid_fascia_cols} / {frame.shape[1]}")

    annotated = renderer.draw_intermediate_frame(frame, blobs, fascia)
    out_path = os.path.join(config.OUTPUTS_DIR, "smoketest_step1_annotated.png")
    cv2.imwrite(out_path, annotated)
    print(f"[smoketest] wrote annotated image to {out_path}")

    # Second run to confirm the lazy-singleton model cache actually avoids a reload.
    t1 = time.time()
    bpe.segment_frame(frame)
    dt2 = time.time() - t1
    print(f"[smoketest] second segment_frame call took {dt2:.2f}s (should be much faster than first)")


if __name__ == "__main__":
    main()
