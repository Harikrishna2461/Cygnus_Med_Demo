import cv2
import json
import os
import numpy as np

video_path = "full.mp4"
segments_file = "stable_segments.json"

with open(segments_file, 'r') as f:
    segments = json.load(f)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs("sheets", exist_ok=True)

COLS = 5
ROWS = 4
CELL_W = 400
CELL_H = 225

for sheet_idx in range(0, len(segments), ROWS * COLS):
    sheet_segments = segments[sheet_idx:sheet_idx + ROWS * COLS]
    sheet_img = np.full((ROWS * CELL_H, COLS * CELL_W, 3), 40, dtype=np.uint8)

    for cell_idx, seg in enumerate(sheet_segments):
        row = cell_idx // COLS
        col = cell_idx % COLS

        frame_num = seg['mid_frame']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((240, 426, 3), dtype=np.uint8)

        frame_resized = cv2.resize(frame, (CELL_W, CELL_H))

        y_start = row * CELL_H
        x_start = col * CELL_W
        sheet_img[y_start:y_start+CELL_H, x_start:x_start+CELL_W] = frame_resized

        label = f"S{sheet_idx+cell_idx} F{frame_num}"
        time_label = f"{seg['mid_frame_sec']:.1f}s"

        cv2.putText(sheet_img, label, (x_start+5, y_start+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(sheet_img, time_label, (x_start+5, y_start+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    sheet_num = sheet_idx // (ROWS * COLS)
    sheet_path = f"sheets/sheet_{sheet_num:03d}.jpg"
    cv2.imwrite(sheet_path, sheet_img)
    print(f"Saved {sheet_path}")

cap.release()
print("Contact sheets generated.")
