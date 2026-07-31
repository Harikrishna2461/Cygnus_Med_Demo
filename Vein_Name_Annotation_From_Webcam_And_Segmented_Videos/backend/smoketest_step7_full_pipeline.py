"""
Build step 9: full two-pass pipeline on a trimmed (unpaired, per project plan's flagged
test-data gap) ultrasound+webcam clip pair. Run: py -3.12 smoketest_step7_full_pipeline.py
"""
import os
import time

import config
import pipeline

ULTRASOUND = os.path.join(config.OUTPUTS_DIR, "test_ultrasound_20s.mp4")
WEBCAM = os.path.join(config.OUTPUTS_DIR, "test_webcam_20s.mp4")
INTERMEDIATE_OUT = os.path.join(config.OUTPUTS_DIR, "smoketest_step7_intermediate.mp4")
FINAL_OUT = os.path.join(config.OUTPUTS_DIR, "smoketest_step7_final.mp4")
ARTIFACT = os.path.join(config.OUTPUTS_DIR, "smoketest_step7_artifact.json")


def progress(stage, frac):
    print(f"  [{stage}] {frac*100:5.1f}%")


def main():
    t0 = time.time()
    pipeline.run_full_pipeline(ULTRASOUND, WEBCAM, INTERMEDIATE_OUT, FINAL_OUT, ARTIFACT, progress_cb=progress)
    dt = time.time() - t0
    print(f"\n[smoketest] full pipeline took {dt:.1f}s")

    for path in (INTERMEDIATE_OUT, FINAL_OUT, ARTIFACT):
        assert os.path.exists(path) and os.path.getsize(path) > 0, f"missing/empty output: {path}"
        print(f"[smoketest] {path}: {os.path.getsize(path)/1024:.1f} KB")

    import video_io
    for path in (INTERMEDIATE_OUT, FINAL_OUT):
        info = video_io.probe_video(path)
        print(f"[smoketest] {os.path.basename(path)} readback: {info}")
        assert info["frame_count"] > 0

    print("[smoketest] OK")


if __name__ == "__main__":
    main()
