import cv2
import json
import os
from collections import defaultdict

video_path = "full.mp4"
output_json = "stable_segments.json"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

print(f"Video: {video_path}")
print(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration_sec:.1f}s")

prev_gray = None
diffs = []
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (240, 135))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray).mean()
        diffs.append(diff)
    else:
        diffs.append(0)

    prev_gray = gray
    frame_num += 1

    if frame_num % 1000 == 0:
        print(f"Processed {frame_num} frames...")

cap.release()

MOTION_THRESHOLD = 2.2
MIN_STABLE_DURATION = 0.6
min_stable_frames = int(MIN_STABLE_DURATION * fps)

segments = []
stable_start = 0
in_stable = diffs[0] < MOTION_THRESHOLD

for i in range(1, len(diffs)):
    is_stable = diffs[i] < MOTION_THRESHOLD

    if is_stable != in_stable:
        segment_len_frames = i - stable_start
        segment_len_sec = segment_len_frames / fps

        if in_stable and segment_len_frames >= min_stable_frames:
            mid_frame = stable_start + segment_len_frames // 2
            segments.append({
                "start_frame": stable_start,
                "end_frame": i,
                "length_frames": segment_len_frames,
                "length_sec": segment_len_sec,
                "mid_frame": mid_frame,
                "mid_frame_sec": mid_frame / fps
            })

        stable_start = i
        in_stable = is_stable

if in_stable and (len(diffs) - stable_start) >= min_stable_frames:
    segment_len_frames = len(diffs) - stable_start
    segment_len_sec = segment_len_frames / fps
    mid_frame = stable_start + segment_len_frames // 2
    segments.append({
        "start_frame": stable_start,
        "end_frame": len(diffs),
        "length_frames": segment_len_frames,
        "length_sec": segment_len_sec,
        "mid_frame": mid_frame,
        "mid_frame_sec": mid_frame / fps
    })

with open(output_json, 'w') as f:
    json.dump(segments, f, indent=2)

print(f"\nDetected {len(segments)} stable segments")
for i, seg in enumerate(segments):
    print(f"  Segment {i}: frames {seg['start_frame']}-{seg['end_frame']}, {seg['length_sec']:.1f}s, mid={seg['mid_frame']}")
