import cv2
import json
import os

video_path = "full.mp4"
segments_file = "stable_segments.json"

with open(segments_file, 'r') as f:
    segments = json.load(f)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs("keyframes", exist_ok=True)

for seg_idx, seg in enumerate(segments):
    frame_num = seg['mid_frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if ret:
        time_sec = frame_num / fps
        filename = f"keyframes/seg_{seg_idx:03d}_{frame_num:06d}_{time_sec:.1f}s.jpg"
        cv2.imwrite(filename, frame)

        if (seg_idx + 1) % 50 == 0:
            print(f"Extracted {seg_idx + 1} keyframes...")

cap.release()
print(f"Extracted all {len(segments)} keyframes to keyframes/ folder")
