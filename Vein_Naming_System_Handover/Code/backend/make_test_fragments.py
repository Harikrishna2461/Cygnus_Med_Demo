"""
Splits the real reference ultrasound+webcam video pair into aligned short fragments,
so ROI cropping (or any pipeline stage) can be tested quickly on any specific time
window instead of waiting on the full 18-minute video every time.
Run: python3 make_test_fragments.py
"""
import os
import subprocess
import imageio_ffmpeg

import config

SRC_DIR = r"C:\Users\Krish\Desktop\data from Ed\data"
US_SRC = os.path.join(SRC_DIR, "QML_recording_2026-07-02T07-31-18-948Z.mp4")
WC_SRC = os.path.join(SRC_DIR, "QMLwebcam_recording_2026-07-02T07-31-18-948Z.mp4")

OUT_DIR = os.path.join(config.OUTPUTS_DIR, "test_fragments")
FRAGMENT_SEC = 120  # 2-minute fragments
TOTAL_SEC = 1092    # both source videos are ~18:12


def _fmt(sec: int) -> str:
    return f"{sec // 60:02d}m{sec % 60:02d}s"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    starts = list(range(0, TOTAL_SEC, FRAGMENT_SEC))
    for start in starts:
        length = min(FRAGMENT_SEC, TOTAL_SEC - start)
        tag = f"{_fmt(start)}-{_fmt(start + length)}"
        for src, prefix in [(US_SRC, "us"), (WC_SRC, "wc")]:
            dst = os.path.join(OUT_DIR, f"{prefix}_{tag}.mp4")
            subprocess.run(
                [ffmpeg, "-y", "-ss", str(start), "-i", src, "-t", str(length),
                 "-c", "copy", dst],
                capture_output=True,
            )
            size = os.path.getsize(dst) if os.path.exists(dst) else 0
            print(f"{dst}  ({size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
