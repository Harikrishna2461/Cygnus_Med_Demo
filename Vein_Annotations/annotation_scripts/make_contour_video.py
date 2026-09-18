import json
import sys
import colorsys
import cv2
import numpy as np
from pycocotools import mask as maskUtils


def ann_to_polygons(a):
    """Return a list of Nx1x2 int32 point arrays for drawing, for either
    polygon-format or RLE-format COCO segmentations."""
    seg = a["segmentation"]
    if isinstance(seg, list):
        polys = []
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = np.array(
                list(zip(poly[0::2], poly[1::2])), dtype=np.int32
            ).reshape(-1, 1, 2)
            polys.append(pts)
        return polys
    else:
        # RLE (dict) segmentation
        if isinstance(seg["counts"], list):
            rle = maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1])
        else:
            rle = seg
        m = maskUtils.decode(rle)
        contours, _ = cv2.findContours(
            m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return [c for c in contours if len(c) >= 3]


def track_color(track_id):
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


def make_contour_video(video_path, json_path, out_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {im["id"]: im for im in data["images"]}

    anns_by_image = {}
    for a in data["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    sorted_ids = sorted(images_by_id.keys())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    annotated_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img_id = sorted_ids[frame_idx] if frame_idx < len(sorted_ids) else None
        anns = anns_by_image.get(img_id, []) if img_id is not None else []

        if anns:
            for a in anns:
                polys = ann_to_polygons(a)
                if not polys:
                    continue
                track_id = a.get("attributes", {}).get("track_id", 0)
                color = track_color(track_id)
                cv2.polylines(frame, polys, True, color, 2, cv2.LINE_AA)
                x, y, w, h = a["bbox"]
                cx, cy = int(x + w / 2), int(y)
                label = f"vein {track_id}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (cx - 2, cy - th - 6),
                    (cx + tw + 2, cy - 2),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (cx, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            annotated_count += 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(
        f"done: {out_path} | frames written: {frame_idx}/{total_frames} | "
        f"annotated frames: {annotated_count}"
    )


if __name__ == "__main__":
    video_path, json_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    make_contour_video(video_path, json_path, out_path)
