import cv2
import numpy as np
import json

video_path = "full.mp4"

with open("movement_analysis.json", 'r') as f:
    movements = json.load(f)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

sorted_movements = sorted(movements, key=lambda x: x['diff_before_after'], reverse=True)
significant_frames = [m for m in sorted_movements if m['diff_before_after'] > 12.0]

measurements = {}

print(f"Refining leg position measurement for {len(significant_frames)} frames...")
print(f"Frame size: {w_frame}x{h_frame}")

for i, mov in enumerate(significant_frames):
    frame_num = mov['mid_frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    skin_lower = np.array([0, 20, 100])
    skin_upper = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        leg_y_positions = []
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            if h_c > 40 and w_c > 10:
                mid_y = y + h_c / 2
                leg_y_positions.append(mid_y)

        if leg_y_positions:
            avg_y = np.mean(leg_y_positions)
            leg_level = avg_y / h_frame
            leg_level = max(0, min(1, leg_level))

            frame_str = str(frame_num)
            measurements[frame_str] = round(leg_level, 3)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(significant_frames)}: Frame {frame_num} -> leg_level={leg_level:.3f}")

cap.release()

annotation_output = {
    "left": measurements,
    "right": {}
}

with open("full_leg_level_annotation.json", 'w') as f:
    json.dump(annotation_output, f, indent=2)

print(f"\nRefined annotation with {len(measurements)} points")
if measurements:
    values = list(measurements.values())
    print(f"Range: {min(values):.3f} - {max(values):.3f}")
    print(f"Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")
