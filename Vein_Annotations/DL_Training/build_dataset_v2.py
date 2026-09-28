"""
Build v2 segmentation dataset at higher resolution (default 320x320, v1 was 256x256).

Same sources as v1 (task_44/45/47 latest CVAT exports + longitudinal clip), but:
  - higher resolution -> thin veins and boundary pixels survive downscaling
  - masks are resized with INTER_AREA then thresholded (smoother, more accurate edges than nearest)
  - written straight to disk as .npy memmaps (no big RAM peak, no 2x copy on save)

Output: Vein_Annotations/segmentation_dataset_v2/
    images.npy  - uint8 [N, 3, R, R]  (RGB)
    masks.npy   - uint8 [N, R, R]     (0/1)
    meta.json   - list of {source, frame_index, has_vein, view}
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from build_dataset_v1 import TASKS, LONGITUDINAL, polys_for_ann, mask_from_green_contour

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "segmentation_dataset_v2"
OUT_DIR.mkdir(exist_ok=True)

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 320


def small_mask(full_mask):
    m = cv2.resize(full_mask, (RES, RES), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.uint8)


def task_length(task):
    with open(task["json"], "r", encoding="utf-8") as f:
        n_json = len(json.load(f)["images"])
    cap = cv2.VideoCapture(str(task["video"]))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return min(n_vid, n_json) if n_vid > 0 else n_json


def long_length(task):
    a = cv2.VideoCapture(str(task["raw_video"])); b = cv2.VideoCapture(str(task["annotated_video"]))
    n = min(int(a.get(cv2.CAP_PROP_FRAME_COUNT)), int(b.get(cv2.CAP_PROP_FRAME_COUNT)))
    a.release(); b.release()
    return n


def main():
    lengths = [task_length(t) for t in TASKS] + [long_length(LONGITUDINAL)]
    total = sum(lengths)
    print("frames per source:", lengths, "total:", total, "res:", RES)

    imgs = np.lib.format.open_memmap(OUT_DIR / "images.npy", mode="w+", dtype=np.uint8, shape=(total, 3, RES, RES))
    masks = np.lib.format.open_memmap(OUT_DIR / "masks.npy", mode="w+", dtype=np.uint8, shape=(total, RES, RES))
    meta = []
    pos = 0
    t0 = time.time()

    for task, n in zip(TASKS, lengths[:-1]):
        with open(task["json"], "r", encoding="utf-8") as f:
            data = json.load(f)
        sorted_ids = sorted(im["id"] for im in data["images"])
        first = next(im for im in data["images"] if im["id"] == sorted_ids[0])
        src_w, src_h = first["width"], first["height"]
        anns_by_image = {}
        for a in data["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)

        cap = cv2.VideoCapture(str(task["video"]))
        for idx in range(n):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{task['name']}: video ended early at frame {idx}/{n}")
            full = np.zeros((src_h, src_w), dtype=np.uint8)
            for a in anns_by_image.get(sorted_ids[idx], []):
                polys = polys_for_ann(a)
                if polys:
                    cv2.fillPoly(full, polys, 255)
            fs = cv2.resize(frame, (RES, RES), interpolation=cv2.INTER_AREA)
            imgs[pos] = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            m = small_mask(full)
            masks[pos] = m
            meta.append({"source": task["name"], "frame_index": idx, "has_vein": bool(m.max() > 0), "view": "transverse"})
            pos += 1
            if (idx + 1) % 1000 == 0:
                print(f"[{task['name']}] {idx+1}/{n} ({time.time()-t0:.0f}s)")
        cap.release()
        print(f"[{task['name']}] done, vein frames so far: {sum(x['has_vein'] for x in meta)}")

    n = lengths[-1]
    cap_raw = cv2.VideoCapture(str(LONGITUDINAL["raw_video"]))
    cap_ann = cv2.VideoCapture(str(LONGITUDINAL["annotated_video"]))
    src_w = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH)); src_h = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for idx in range(n):
        ok1, raw = cap_raw.read(); ok2, ann = cap_ann.read()
        if not (ok1 and ok2):
            raise RuntimeError("longitudinal clip ended early")
        fs = cv2.resize(raw, (RES, RES), interpolation=cv2.INTER_AREA)
        imgs[pos] = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        m = small_mask(mask_from_green_contour(ann, src_w, src_h))
        masks[pos] = m
        meta.append({"source": LONGITUDINAL["name"], "frame_index": idx, "has_vein": bool(m.max() > 0), "view": "longitudinal"})
        pos += 1
    cap_raw.release(); cap_ann.release()

    assert pos == total, (pos, total)
    imgs.flush(); masks.flush()
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"saved {total} frames to {OUT_DIR} in {time.time()-t0:.0f}s | with vein: {sum(x['has_vein'] for x in meta)}")


if __name__ == "__main__":
    main()
