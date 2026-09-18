import json
import colorsys
import cv2
import numpy as np

JSON_PATH = r"job_47_annotations_2026_09_17_05_46_01_coco 1.0/annotations/instances_default.json"
VIDEO_PATH = r"raw_video/QML_recording_2026-09-10_17-56-49.mkv"


def load_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


def anns_by_image(data):
    d = {}
    for a in data["annotations"]:
        d.setdefault(a["image_id"], []).append(a)
    return d


def track_color(track_id):
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def get_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def contact_sheet(frame_indices, data, out_path, cols=6, cell=(280, 220), label_ids=True):
    cap = cv2.VideoCapture(VIDEO_PATH)
    aim = anns_by_image(data)
    rows = (len(frame_indices) + cols - 1) // cols
    cw, ch = cell
    sheet = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for idx, fidx in enumerate(frame_indices):
        frame = get_frame(cap, fidx)
        cell_img = np.zeros((ch, cw, 3), dtype=np.uint8)
        if frame is not None:
            img_id = fidx + 1
            anns = aim.get(img_id, [])
            f2 = frame.copy()
            for a in anns:
                poly = a["segmentation"][0]
                pts = np.array(list(zip(poly[0::2], poly[1::2])), dtype=np.int32).reshape(-1, 1, 2)
                tid = a["attributes"]["track_id"]
                color = track_color(tid)
                cv2.polylines(f2, [pts], True, color, 2)
                if label_ids:
                    x, y, w, h = a["bbox"]
                    cv2.putText(f2, str(tid), (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            small = cv2.resize(f2, (cw, ch - 16))
            cell_img[0:ch-16, :] = small
        cv2.putText(cell_img, f"f{fidx} t={fidx/30:.2f}s", (2, ch - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        r, c = idx // cols, idx % cols
        sheet[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = cell_img
    cv2.imwrite(out_path, sheet)
    cap.release()
    print("saved", out_path)


def zoom_frame(frame_idx, data, out_path, pad=80, scale=4, grid=True):
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame = get_frame(cap, frame_idx)
    cap.release()
    if frame is None:
        print("no frame")
        return
    aim = anns_by_image(data)
    img_id = frame_idx + 1
    anns = aim.get(img_id, [])
    f2 = frame.copy()
    for a in anns:
        poly = a["segmentation"][0]
        pts = np.array(list(zip(poly[0::2], poly[1::2])), dtype=np.int32).reshape(-1, 1, 2)
        tid = a["attributes"]["track_id"]
        color = track_color(tid)
        cv2.polylines(f2, [pts], True, color, 2)
        x, y, w, h = a["bbox"]
        cv2.putText(f2, str(tid), (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    big = cv2.resize(f2, (f2.shape[1]*scale//4, f2.shape[0]*scale//4)) if scale != 4 else f2
    cv2.imwrite(out_path, f2)
    print("saved", out_path, "shape", f2.shape)
