"""
Validate the copied ROI-cropping system (from ROI_Identification/) against the real
reference ultrasound video before wiring it into the main pipeline.
Run: python3 smoketest_step9_roi_crop.py
"""
import os
import config
import roi_pipeline

SRC = os.path.join(config.OUTPUTS_DIR, "ed_ref", "qml_us_20s.mp4")
OUT_DIR = os.path.join(config.OUTPUTS_DIR, "ed_ref", "roi_test")


def main():
    result = roi_pipeline.run_pipeline(SRC, OUT_DIR, use_registry=True, use_agent=True)
    print("\n[smoketest] result:", result)
    assert os.path.exists(result["output_path"])
    print(f"[smoketest] cropped video written: {result['output_path']}")

    import video_io
    info = video_io.probe_video(result["output_path"])
    print(f"[smoketest] cropped video info: {info}")


if __name__ == "__main__":
    main()
