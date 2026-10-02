import cv2
import numpy as np
import json

video_path = "full.mp4"

with open("movement_analysis.json", 'r') as f:
    movements = json.load(f)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

sorted_movements = sorted(movements, key=lambda x: x['diff_before_after'], reverse=True)
significant_frames = [m for m in sorted_movements if m['diff_before_after'] > 12.0]

measurements = {}

print("Analyzing leg position for each significant frame...")
for i, mov in enumerate(significant_frames):
    frame_num = mov['mid_frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    skin_lower = np.array([0, 20, 100])
    skin_upper = np.array([20, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_c, h_c = cv2.boundingRect(largest_contour)

        if h_c > 50:
            top_y = y
            bottom_y = y + h_c
            leg_level = bottom_y / h if h > 0 else 0.5
            leg_level = max(0, min(1, leg_level))

            frame_str = str(frame_num)
            measurements[frame_str] = round(leg_level, 2)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(significant_frames)}: Frame {frame_num} -> leg_level={leg_level:.2f}")
    else:
        print(f"  Warning: No skin detected in frame {frame_num}")

cap.release()

annotation_output = {
    "left": measurements,
    "right": {}
}

with open("full_leg_level_annotation.json", 'w') as f:
    json.dump(annotation_output, f, indent=2)

print(f"\nCreated annotation with {len(measurements)} points")
print(f"Saved to: full_leg_level_annotation.json")

if measurements:
    values = list(measurements.values())
    print(f"Min leg level: {min(values):.2f}, Max: {max(values):.2f}")
