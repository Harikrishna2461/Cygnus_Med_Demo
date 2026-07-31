"""
Build step 8: video I/O smoke test. Reads ~15s worth of sampled frames from a real
ultrasound-like source video and writes a short test clip, to find out which fourcc this
machine's OpenCV build can actually produce before wiring up the full pipeline.
Run: py -3.12 smoketest_step6_video_io.py
"""
import os

import config
import video_io

SRC = os.path.join(config.CYGNUS_ROOT, "Vein_Segmented_Videos", "202402201648_24", "202402201648_24.mp4")
OUT = os.path.join(config.OUTPUTS_DIR, "smoketest_step6_clip.mp4")


def main():
    info = video_io.probe_video(SRC)
    print(f"[smoketest] source: {info}")

    writer = None
    n_written = 0
    max_samples = 15  # ~15s at 1 sample/sec
    for ts, frame in video_io.iter_sample_frames(SRC, interval_sec=1.0):
        if writer is None:
            h, w = frame.shape[:2]
            writer = video_io.OutputVideoWriter(OUT, fps=config.OUTPUT_FPS, frame_size=(w, h))
            print(f"[smoketest] opened writer with fourcc={writer.fourcc_used}")
        # hold each sampled frame for OUTPUT_FPS frames (1 second of output per sample)
        for _ in range(config.OUTPUT_FPS):
            writer.write(frame)
        n_written += 1
        print(f"  sample @ {ts:.1f}s written")
        if n_written >= max_samples:
            break
    writer.release()

    size_bytes = os.path.getsize(OUT)
    print(f"[smoketest] wrote {OUT} ({size_bytes/1024:.1f} KB) using fourcc={writer.fourcc_used}")

    readback = video_io.probe_video(OUT)
    print(f"[smoketest] readback probe: {readback}")
    assert readback["frame_count"] > 0, "written video has 0 frames on readback"
    print("[smoketest] OK")


if __name__ == "__main__":
    main()
