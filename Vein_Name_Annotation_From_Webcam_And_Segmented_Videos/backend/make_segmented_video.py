"""
Produces a pure BioMedParse segmentation video (vein contours + fascia lines, numbered,
no N1/N2/N3 classification and no Groq/VLM calls at all) for one ultrasound video. Fast:
only ROI-crop (Stage 0) + local GPU segmentation, no network calls.
Run: python3 make_segmented_video.py <input_video> <output_video>
"""
import os
import sys
import time

import config
import roi_pipeline
import biomedparse_engine as bpe
import renderer
import video_io


def main(input_path: str, output_path: str, roi_out_dir: str):
    t0 = time.time()
    print(f"[segment] ROI-cropping {input_path} ...")
    cropped_path = None
    try:
        result = roi_pipeline.run_pipeline(input_path, roi_out_dir, use_registry=True, use_agent=True)
        cropped_path = result["output_path"]
        print(f"[segment] ROI: {result['roi']} method={result['method']}")
    except Exception as exc:
        print(f"[segment] ROI crop failed ({exc}); using uncropped video.")
        cropped_path = input_path

    info = video_io.probe_video(cropped_path)
    duration = max(info["duration_sec"], 1e-6)
    hold_frames = max(1, round(config.SEG_SAMPLE_INTERVAL_SEC * config.OUTPUT_FPS))

    writer = None
    n_frames_done = 0
    n_blobs_total = 0
    for ts, frame in video_io.iter_sample_frames(cropped_path, config.SEG_SAMPLE_INTERVAL_SEC):
        blobs, fascia = bpe.segment_frame(frame)
        n_blobs_total += len(blobs)
        annotated = renderer.draw_intermediate_frame(frame, blobs, fascia)  # blob numbers only, no n_class
        if writer is None:
            h, w = frame.shape[:2]
            writer = video_io.OutputVideoWriter(output_path, fps=config.OUTPUT_FPS, frame_size=(w, h))
        for _ in range(hold_frames):
            writer.write(annotated)
        n_frames_done += 1
        if n_frames_done % 20 == 0:
            print(f"[segment] {ts:.1f}s / {duration:.1f}s  ({n_frames_done} samples, "
                  f"{n_blobs_total} blobs so far)")

    if writer:
        writer.release()
    dt = time.time() - t0
    print(f"[segment] done in {dt:.1f}s -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python3 make_segmented_video.py <input_video> <output_video> <roi_out_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
