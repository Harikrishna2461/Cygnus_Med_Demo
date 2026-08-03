"""
Runs ONLY ROI detection+crop (no BioMedParse, no Groq classification/naming) across all
ultrasound fragments in outputs/test_fragments/, so ROI correctness can be checked quickly
across the whole video. Saves one representative cropped-frame preview per fragment.
Run: python3 test_roi_batch.py
"""
import glob
import os
import cv2

import config
import roi_pipeline

FRAG_DIR = os.path.join(config.OUTPUTS_DIR, "test_fragments")
OUT_DIR = os.path.join(config.OUTPUTS_DIR, "roi_batch_test")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    us_fragments = sorted(glob.glob(os.path.join(FRAG_DIR, "us_*.mp4")))
    print(f"Found {len(us_fragments)} ultrasound fragments\n")

    results = []
    for path in us_fragments:
        tag = os.path.splitext(os.path.basename(path))[0]
        print(f"--- {tag} ---")
        try:
            result = roi_pipeline.run_pipeline(path, OUT_DIR, use_registry=True, use_agent=True)
            print(f"  ROI: {result['roi']}  method={result['method']}  "
                  f"confidence={result['confidence']}  view={result['view_type']}")
            # Save a mid-clip preview frame from the cropped output for quick eyeballing
            cap = cv2.VideoCapture(result["output_path"])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
            ok, frame = cap.read()
            cap.release()
            preview_path = os.path.join(OUT_DIR, f"{tag}_preview.png")
            if ok:
                cv2.imwrite(preview_path, frame)
                print(f"  preview: {preview_path}")
            results.append((tag, result["roi"], result["method"], result["confidence"]))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append((tag, None, "FAILED", 0))
        print()

    print("=== Summary ===")
    for tag, roi, method, conf in results:
        print(f"{tag}: roi={roi} method={method} confidence={conf}")


if __name__ == "__main__":
    main()
