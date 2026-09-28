"""
Build v1 segmentation dataset (image + binary vein mask tensors).

Changes vs build_dataset.py:
  - task_47 now points at the latest manually-refined CVAT export
    (task_47_annotations_2026_09_22_10_19_58) instead of the Sept 17 export.
  - Adds the longitudinal-view clip (Vein_Annotations/Longitudinal_View_Data), which has
    no CVAT/COCO json - instead it ships a raw video and a visually-annotated video where
    the vein boundary is drawn as a solid green contour. The green contour is isolated by
    color thresholding, closed, and filled to synthesize a binary mask. This gives the
    model longitudinal (long-axis) vein examples, which the transverse-only task_44/45/47
    data doesn't cover at all.

Output (single folder): Vein_Annotations/segmentation_dataset_v1/
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

ROOT = Path(__file__).resolve().parent.parent  # Vein_Annotations/
OUT_DIR = ROOT / "segmentation_dataset_v1"
OUT_DIR.mkdir(exist_ok=True)

TARGET_W, TARGET_H = 256, 256

TASKS = [
    {
        "name": "task_44",
        "video": ROOT / "Task_44" / "raw_video" / "QML_recording_2026-09-02_12-27-38.mkv",
        "json": ROOT / "Task_44" / "task_44_annotations_2026_09_18_05_15_01_coco 1.0" / "annotations" / "instances_default.json",
    },
    {
        "name": "task_45",
        "video": ROOT / "Task_45" / "raw_video" / "QML_recording_2026-09-02_12-31-19.mkv",
        "json": ROOT / "Task_45" / "task_45_annotations_2026_09_18_05_53_35_coco 1.0" / "annotations" / "instances_default.json",
    },
    {
        "name": "task_47",
        "video": ROOT / "Task_47" / "raw_video" / "QML_recording_2026-09-10_17-56-49.mkv",
        "json": ROOT / "Task_47" / "task_47_annotations_2026_09_22_10_19_58_coco 1.0" / "annotations" / "instances_default.json",
    },
]

LONGITUDINAL = {
    "name": "longitudinal_52",
    "raw_video": ROOT / "Longitudinal_View_Data" / "Raw_202207221607_52-Long2.mp4",
    "annotated_video": ROOT / "Longitudinal_View_Data" / "Vein_Annotated_202207221607_52-Long2.mp4",
}

# Green contour overlay color range (BGR), tuned against the sample frame diff.
GREEN_LOWER = np.array([0, 120, 0], dtype=np.uint8)
GREEN_UPPER = np.array([120, 255, 120], dtype=np.uint8)


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
        from pycocotools import mask as maskUtils  # only needed for RLE annotations
        if isinstance(seg["counts"], list):
            rle = maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1])
        else:
            rle = seg
        m = maskUtils.decode(rle)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if len(c) >= 3]


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


def mask_from_green_contour(annotated_frame, src_w, src_h):
    """Isolate the green vein-boundary overlay and fill it into a solid binary mask."""
    green_hit = cv2.inRange(annotated_frame, GREEN_LOWER, GREEN_UPPER)
    # also require green to dominate red+blue, to reject blown-out white/gray specular pixels
    b, g, r = cv2.split(annotated_frame.astype(np.int16))
    dominant = ((g - r) > 25) & ((g - b) > 25)
    green_hit = green_hit & (dominant.astype(np.uint8) * 255)

    if green_hit.max() == 0:
        return np.zeros((src_h, src_w), dtype=np.uint8)

    # close small gaps in the traced contour line before filling
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(green_hit, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 10]
    if not contours:
        return np.zeros((src_h, src_w), dtype=np.uint8)

    full_mask = np.zeros((src_h, src_w), dtype=np.uint8)
    cv2.drawContours(full_mask, contours, -1, 255, thickness=cv2.FILLED)
    return full_mask


def process_longitudinal(task):
    """No COCO json for this clip - masks are synthesized from the green contour overlay
    baked into the 'annotated' video, paired frame-for-frame with the raw video.
    Uses every single frame of the clip (it's only ~120 frames total)."""
    print(f"[{task['name']}] loading raw + annotated video pair...")
    cap_raw = cv2.VideoCapture(str(task["raw_video"]))
    cap_ann = cv2.VideoCapture(str(task["annotated_video"]))
    if not cap_raw.isOpened() or not cap_ann.isOpened():
        raise RuntimeError(f"cannot open longitudinal video pair for {task['name']}")

    n_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    n_ann = int(cap_ann.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_raw, n_ann)
    print(f"[{task['name']}] raw={n_raw} frames, annotated={n_ann} frames, using {n}")

    src_w = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))

    imgs = np.empty((n, TARGET_H, TARGET_W, 3), dtype=np.uint8)
    masks = np.empty((n, TARGET_H, TARGET_W), dtype=np.uint8)
    meta = []

    idx = 0
    while idx < n:
        ok1, raw_frame = cap_raw.read()
        ok2, ann_frame = cap_ann.read()
        if not ok1 or not ok2:
            break

        full_mask = mask_from_green_contour(ann_frame, src_w, src_h)

        frame_small = cv2.resize(raw_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(full_mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        imgs[idx] = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        masks[idx] = mask_small
        meta.append(
            {
                "source": task["name"],
                "frame_index": idx,
                "has_vein": bool(mask_small.max() > 0),
                "view": "longitudinal",
            }
        )
        idx += 1

    cap_raw.release()
    cap_ann.release()
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
        for m in meta:
            m["view"] = "transverse"
        all_imgs.append(imgs)
        all_masks.append(masks)
        all_meta.extend(meta)

    long_imgs, long_masks, long_meta = process_longitudinal(LONGITUDINAL)
    all_imgs.append(long_imgs)
    all_masks.append(long_masks)
    all_meta.extend(long_meta)

    imgs_cat = np.concatenate(all_imgs, axis=0)  # N,H,W,3
    masks_cat = np.concatenate(all_masks, axis=0)  # N,H,W

    print("total frames:", imgs_cat.shape[0])
    print("longitudinal frames:", len(long_meta), "with vein:", sum(m["has_vein"] for m in long_meta))

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
