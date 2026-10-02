import cv2
import json
import os

video_path = "full.mp4"

with open("movement_analysis.json", 'r') as f:
    movements = json.load(f)

sorted_movements = sorted(movements, key=lambda x: x['diff_before_after'], reverse=True)
top_movements = sorted_movements[:50]

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs("top_keyframes", exist_ok=True)

print("Extracting top 50 keyframes for inspection:")
for i, mov in enumerate(top_movements):
    frame_num = mov['mid_frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if ret:
        time_sec = frame_num / fps
        filename = f"top_keyframes/{i+1:02d}_seg{mov['segment_idx']:3d}_f{frame_num:6d}_{time_sec:.1f}s.jpg"
        cv2.imwrite(filename, frame)
        print(f"{i+1:2d}. Frame {frame_num:6d} @ {time_sec:8.1f}s (diff={mov['diff_before_after']:6.2f})")

cap.release()
print(f"\nExtracted top 50 keyframes to top_keyframes/ folder")
