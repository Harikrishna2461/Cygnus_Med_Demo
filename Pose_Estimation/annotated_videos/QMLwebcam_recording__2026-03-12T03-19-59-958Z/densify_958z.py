import json
import cv2
from pathlib import Path

video_path = Path("full.mp4")
annotation_file = "full_leg_level_annotation.json"

with open(annotation_file, 'r') as f:
    data = json.load(f)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

left_points = {int(k): v for k, v in data['left'].items()}
sorted_frames = sorted(left_points.keys())

print(f"Original 958Z: {len(left_points)} points")
print(f"Total frames: {total_frames}, FPS: {fps}")
print(f"Duration: {total_frames/fps:.1f}s")

densified = {}

for i in range(len(sorted_frames) - 1):
    frame_a = sorted_frames[i]
    frame_b = sorted_frames[i + 1]
    level_a = left_points[frame_a]
    level_b = left_points[frame_b]

    densified[frame_a] = level_a

    frames_between = frame_b - frame_a
    if frames_between > 10:
        steps = max(2, frames_between // 5)
        for step in range(1, steps):
            inter_frame = frame_a + (frame_b - frame_a) * step // steps
            inter_level = level_a + (level_b - level_a) * step / steps
            inter_level = round(inter_level, 2)
            densified[inter_frame] = inter_level

densified[sorted_frames[-1]] = left_points[sorted_frames[-1]]

densified_sorted = {k: densified[k] for k in sorted(densified.keys())}

output = {
    "left": densified_sorted,
    "right": {}
}

with open(annotation_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Densified 958Z: {len(densified_sorted)} points (added {len(densified_sorted) - len(left_points)})")
