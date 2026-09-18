"""
Build a segmentation dataset (image + binary vein mask tensors) from the
three raw videos and their COCO annotations in Vein_Annotations/.

Every single frame of every video is used (not just annotated ones) -
unannotated frames simply get an all-zero mask, which is a valid negative
example (e.g. probe lifted, no vein in view).

Output (single folder): Vein_Annotations/segmentation_dataset/
    images.pt   - uint8 tensor [N, 3, H, W]   (RGB)
    masks.pt    - uint8 tensor [N, 1, H, W]   (0/255)
    meta.json   - list of dicts: {source, frame_index, has_vein}
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils

ROOT = Path(__file__).resolve().parent.parent  # Vein_Annotations/
OUT_DIR = ROOT / "segmentation_dataset"
OUT_DIR.mkdir(exist_ok=True)

TARGET_W, TARGET_H = 256, 256

TASKS = [
    {
        "name": "task_44",
        "video": ROOT / "Task_44" / "raw_video" / "QML_recording_2026-09-02_12-27-38.mkv",
        "json": ROOT / "Task_44" / "task_44_annotations_2026_09_11_10_06_08_coco 1.0" / "annotations" / "instances_default.json",
    },
    {
        "name": "task_45",
        "video": ROOT / "Task_45" / "raw_video" / "QML_recording_2026-09-02_12-31-19.mkv",
        "json": ROOT / "Task_45" / "task_45_annotations_2026_09_09_08_38_06_coco 1.0" / "annotations" / "instances_default.json",
    },
    {
        "name": "task_47",
        "video": ROOT / "Task_47" / "raw_video" / "QML_recording_2026-09-10_17-56-49.mkv",
        "json": ROOT / "Task_47" / "job_47_annotations_2026_09_17_05_46_01_coco 1.0" / "annotations" / "instances_default.json",
    },
]


def polys_for_ann(ann):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        polys = []
        for p in seg:
            if len(p) < 6:
                continue
            pts = np.array(list(zip(p[0::2], p[1::2])), dtype=np.int32).reshape(-1, 1, 2)
            polys.append(pts)
        return polys
    else:
        if isinstance(seg["counts"], list):
            rle = maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1])
        else:
            rle = seg
        m = maskUtils.decode(rle)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if len(c) >= 3]


def count_frames_exact(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def process_task(task):
    print(f"[{task['name']}] loading annotations...")
    with open(task["json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {im["id"]: im for im in data["images"]}
    sorted_ids = sorted(images_by_id.keys())

    anns_by_image = {}
    for a in data["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    src_w = images_by_id[sorted_ids[0]]["width"]
    src_h = images_by_id[sorted_ids[0]]["height"]

    cap = cv2.VideoCapture(str(task["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {task['video']}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{task['name']}] video reports {total} frames, json has {len(sorted_ids)} images")

    n = min(total, len(sorted_ids)) if total > 0 else len(sorted_ids)

    imgs = np.empty((n, TARGET_H, TARGET_W, 3), dtype=np.uint8)
    masks = np.empty((n, TARGET_H, TARGET_W), dtype=np.uint8)
    meta = []

    t0 = time.time()
    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break

        img_id = sorted_ids[idx]
        anns = anns_by_image.get(img_id, [])

        full_mask = np.zeros((src_h, src_w), dtype=np.uint8)
        for a in anns:
            polys = polys_for_ann(a)
            if polys:
                cv2.fillPoly(full_mask, polys, 255)

        frame_small = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(full_mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        imgs[idx] = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        masks[idx] = mask_small
        meta.append(
            {
                "source": task["name"],
                "frame_index": idx,
                "has_vein": bool(mask_small.max() > 0),
            }
        )

        idx += 1
        if idx % 1000 == 0:
            elapsed = time.time() - t0
            print(f"[{task['name']}] {idx}/{n} frames ({elapsed:.1f}s, {idx/elapsed:.1f} fps)")

    cap.release()
    if idx < n:
        imgs = imgs[:idx]
        masks = masks[:idx]
    print(f"[{task['name']}] done: {idx} frames, {sum(m['has_vein'] for m in meta)} with vein")
    return imgs, masks, meta


def main():
    all_imgs = []
    all_masks = []
    all_meta = []

    for task in TASKS:
        imgs, masks, meta = process_task(task)
        all_imgs.append(imgs)
        all_masks.append(masks)
        all_meta.extend(meta)

    imgs_cat = np.concatenate(all_imgs, axis=0)  # N,H,W,3
    masks_cat = np.concatenate(all_masks, axis=0)  # N,H,W

    print("total frames:", imgs_cat.shape[0])

    images_t = torch.from_numpy(imgs_cat).permute(0, 3, 1, 2).contiguous()  # N,3,H,W
    masks_t = torch.from_numpy(masks_cat).unsqueeze(1).contiguous()  # N,1,H,W

    torch.save(images_t, OUT_DIR / "images.pt")
    torch.save(masks_t, OUT_DIR / "masks.pt")
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f)

    print("saved to", OUT_DIR)
    print("images.pt:", images_t.shape, images_t.dtype)
    print("masks.pt:", masks_t.shape, masks_t.dtype)


if __name__ == "__main__":
    main()
