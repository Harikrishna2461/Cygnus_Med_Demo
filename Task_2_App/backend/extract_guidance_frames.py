"""
extract_guidance_frames.py

Extract frames from the guidance concat MP4 videos into their respective
subfolder so the /api/vein-frame endpoint can serve them as N1/N2/N3
annotated reference images.

Run from Task_2_App root:
  python backend/extract_guidance_frames.py
"""

from __future__ import annotations
import os
import cv2

GUIDANCE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "guidance")
FRAMES_PER_VIDEO = 60   # frames to extract from each video

_VIDEO_TO_SUBFOLDER = {
    "sfj_concat.mp4":     "sfj",
    "ssv_concat.mp4":     "ssv",
    "spj_concat.mp4":     "spj",
    "gsv_thi_concat.mp4": "gsv_thigh",
    "gsv_cal_concat.mp4": "gsv_calf",
}


def extract_frames(video_path: str, out_dir: str, n: int) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    step = max(1, total // n)
    indices = list(range(0, total, step))[:n]

    for idx in indices:
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        if os.path.isfile(out_path):
            saved += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            saved += 1

    cap.release()
    return saved


def main() -> None:
    for video_name, subfolder in _VIDEO_TO_SUBFOLDER.items():
        video_path = os.path.join(GUIDANCE_DIR, video_name)
        if not os.path.isfile(video_path):
            print(f"  [skip] {video_name} not found")
            continue
        out_dir = os.path.join(GUIDANCE_DIR, subfolder)
        saved = extract_frames(video_path, out_dir, FRAMES_PER_VIDEO)
        print(f"  {video_name} -> {subfolder}/: {saved} frames")


if __name__ == "__main__":
    main()
