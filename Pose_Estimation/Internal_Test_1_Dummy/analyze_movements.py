import cv2
import json
import numpy as np

video_path = "full.mp4"
segments_file = "stable_segments.json"

with open(segments_file, 'r') as f:
    segments = json.load(f)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

movements = []

for seg_idx, seg in enumerate(segments):
    start_frame = seg['start_frame']
    end_frame = seg['end_frame']

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 5))
    ret, frame_before = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, seg['mid_frame'])
    ret, frame_mid = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, end_frame + 5))
    ret, frame_after = cap.read()

    if frame_before is not None and frame_mid is not None and frame_after is not None:
        gray_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
        gray_mid = cv2.cvtColor(frame_mid, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)

        diff_before_mid = cv2.absdiff(gray_before, gray_mid).mean()
        diff_mid_after = cv2.absdiff(gray_mid, gray_after).mean()
        diff_before_after = cv2.absdiff(gray_before, gray_after).mean()

        movements.append({
            "segment_idx": seg_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "mid_frame": seg['mid_frame'],
            "mid_frame_sec": seg['mid_frame_sec'],
            "duration_sec": seg['length_sec'],
            "diff_before_mid": diff_before_mid,
            "diff_mid_after": diff_mid_after,
            "diff_before_after": diff_before_after,
            "avg_diff": (diff_before_mid + diff_mid_after) / 2
        })

    if (seg_idx + 1) % 50 == 0:
        print(f"Analyzed {seg_idx + 1} segments...")

cap.release()

with open("movement_analysis.json", 'w') as f:
    json.dump(movements, f, indent=2)

sorted_by_diff = sorted(movements, key=lambda x: x['diff_before_after'], reverse=True)

print(f"\nTop 50 segments with largest position changes:")
for i, mov in enumerate(sorted_by_diff[:50]):
    print(f"{i+1}. Seg {mov['segment_idx']:3d} @ {mov['mid_frame_sec']:8.1f}s (diff={mov['diff_before_after']:6.2f}, dur={mov['duration_sec']:5.1f}s)")

print(f"\nTotal movements analyzed: {len(movements)}")
print(f"Movement analysis saved to movement_analysis.json")
