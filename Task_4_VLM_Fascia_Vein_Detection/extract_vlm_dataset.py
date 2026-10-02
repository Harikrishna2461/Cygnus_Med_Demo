"""
Extract annotated frames for VLM fine-tuning.

Output structure:
  veins_vlm_finetuning/
    transverse_view/
      frames/   <- raw JPEG
      masks/    <- coloured RGB PNG  (black bg, each class its own colour)
    longitudinal_view/
      frames/
      masks/

Transverse frames where a dark circular vein-like region exists WITHOUT an
annotation are discarded (they would teach the model wrong negative examples).
"""

import re, json, numpy as np, cv2
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"c:\Users\Krish\Downloads\videos")
OUTPUT_DIR = BASE_DIR / "veins_vlm_finetuning"

for sub in ["transverse_view/frames", "transverse_view/masks",
            "longitudinal_view/frames", "longitudinal_view/masks"]:
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# ── class metadata ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "GSV Prox.", "GSV Distal", "Tributary", "SSV",
    "CFV", "FV", "DFV", "AASV", "PASV",
    "Hunt. Perf.", "Dodd Perf.", "Boyd Perf.", "Cockett Perf.", "Ankle Perf.",
    "Escape Point", "Re-entry Point",
    "Start Color Doppler", "End Color Doppler",
    "Start Positive Flow", "End Positive Flow",
    "Start Pulse Wave", "End Pulse Wave",
    "Positive Duration", "PV", "Deep Vein (Calf)", "Thrombose",
]

# BGR colours (used when saving PNGs via cv2.imwrite)
CLASS_COLORS_BGR = [
    ( 80,  80, 255), ( 80, 200,  80), (255,  80,  80), (  0, 220, 255),
    (220,   0, 220), (220, 220,   0), (  0, 140, 255), (255,   0, 160),
    (255, 140,   0), (140,   0, 255), (  0, 255, 140), (140, 255,   0),
    (255, 140, 140), (140, 140, 255), (140, 255, 140), (140, 255, 255),
    (255, 255, 140), (255, 140, 255), ( 30, 105, 210), ( 50, 205,  50),
    (205,  90, 106), ( 50, 205, 205), (180, 130,  70), ( 60,  20, 220),
    ( 23, 221, 100), (200,  25,  25),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def group_chunks(masks_dir):
    """
    Returns (is_format_b, chunk_groups).
    chunk_groups: {chunk_num (int): [(cls_id_or_None, Path), ...]}
    """
    all_npz = sorted(masks_dir.glob("*.npz"))
    is_b = any(re.match(r"obj_\d+_", f.name) for f in all_npz)
    groups = defaultdict(list)
    for npz in all_npz:
        if is_b:
            m = re.match(r"obj_(\d+)_chunk_(\d+)", npz.name)
            if m:
                groups[int(m.group(2))].append((int(m.group(1)), npz))
        else:
            m = re.match(r"chunk_(\d+)", npz.name)
            if m:
                groups[int(m.group(1))].append((None, npz))
    return is_b, dict(sorted(groups.items()))


def load_chunk(files, is_format_b):
    """
    Load one chunk into {frame_idx: {cls_id: uint8_mask}}.
    Stores only frames that have at least one non-zero mask.
    """
    result = {}
    if is_format_b:
        for cls_id, npz_path in files:
            data = np.load(npz_path, allow_pickle=True)
            for key in data.keys():
                fi   = int(key)
                mask = data[key]
                if mask.any():
                    if fi not in result:
                        result[fi] = {}
                    result[fi][cls_id] = mask.copy()
            data.close()
    else:
        _, npz_path = files[0]
        data = np.load(npz_path, allow_pickle=True)
        for key in data.keys():
            fi = int(key)
            md = data[key].item()          # scalar obj → dict
            active = {cid: m.copy() for cid, m in md.items() if m.any()}
            if active:
                result[fi] = active
        data.close()
    return result


def classify_view(masks_dict):
    """
    Classify frame as 'transverse', 'longitudinal', or 'unknown'.

    Transverse (cross-section): mask regions are roughly circular  → aspect ratio ≈ 1
    Longitudinal (along vessel): mask regions are elongated stripes → aspect ratio ≪ 1
    """
    ratios, areas = [], []
    for mask in masks_dict.values():
        if not mask.any():
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            # rotated rect gives true min/max dimensions even for tilted ovals
            if len(cnt) >= 5:
                (_, _), (w, h), _ = cv2.minAreaRect(cnt)
                if max(w, h) > 0:
                    ratios.append(min(w, h) / max(w, h))
                    areas.append(area)

    if not ratios:
        return "unknown"

    total    = sum(areas)
    weighted = sum(r * a for r, a in zip(ratios, areas)) / total

    if weighted >= 0.50:
        return "transverse"
    if weighted <= 0.28:
        return "longitudinal"
    return "unknown"


def make_mask_png(masks_dict, h, w):
    """Coloured BGR image: background=black, each class=its colour."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, mask in masks_dict.items():
        if cls_id < len(CLASS_COLORS_BGR) and mask.any():
            img[mask.astype(bool)] = CLASS_COLORS_BGR[cls_id]
    return img


def has_unannotated_veins(frame_bgr, masks_dict, roi):
    """
    Returns True if a dark, circular blob is found INSIDE the ROI that has
    < 30 % overlap with the existing annotations.

    Strategy:
      1. Convert ROI to grayscale, blur heavily (kills speckle).
      2. Otsu threshold → dark-region binary mask.
      3. Morphological open/close to merge nearby blobs.
      4. Keep blobs that are the right size + circular enough.
      5. Flag any blob not substantially covered by an annotation.
    """
    cx1, cy1, cx2, cy2 = roi
    h, w = frame_bgr.shape[:2]

    # combined annotation in ROI space
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks_dict.values():
        if mask.any():
            combined |= mask.astype(np.uint8)
    ann_roi = combined[cy1:cy2, cx1:cx2]

    gray    = cv2.cvtColor(frame_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)

    # Otsu on inverted image → bright blobs = dark regions in original
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  k, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 15000:      # too small = noise, too large = background
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.38:              # must be reasonably circular
            continue
        # bounding box aspect ratio: reject very elongated blobs
        x, y, bw, bh = cv2.boundingRect(cnt)
        if max(bw, bh) == 0 or min(bw, bh) / max(bw, bh) < 0.30:
            continue

        blob = np.zeros_like(ann_roi)
        cv2.drawContours(blob, [cnt], -1, 255, -1)
        blob_px = int((blob > 0).sum())
        if blob_px == 0:
            continue
        overlap = int(np.logical_and(blob > 0, ann_roi > 0).sum())
        if overlap / blob_px < 0.30:        # < 30 % covered → unannotated vein
            return True

    return False


# ── main extraction ────────────────────────────────────────────────────────────

SKIP_DIRS = {"annotated_output", "veins_vlm_finetuning"}

stats = dict(transverse=0, longitudinal=0, unknown=0,
             filtered_incomplete=0, saved_t=0, saved_l=0)

for case_dir in sorted(BASE_DIR.iterdir()):
    if not case_dir.is_dir() or case_dir.name in SKIP_DIRS:
        continue

    video_file = case_dir / (case_dir.name + ".mp4")
    masks_dir  = case_dir / "masks"
    roi_file   = case_dir / "roi.json"

    if not video_file.exists() or not masks_dir.exists():
        print(f"[SKIP] {case_dir.name}")
        continue

    with open(roi_file) as f:
        roi = json.load(f)["crop_region"]   # [x1, y1, x2, y2]

    is_format_b, chunk_groups = group_chunks(masks_dir)
    cap   = cv2.VideoCapture(str(video_file))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*60}")
    print(f"Case : {case_dir.name}  |  fmt={'B' if is_format_b else 'A'}"
          f"  |  {len(chunk_groups)} chunks  |  {total} video frames")

    for chunk_num in sorted(chunk_groups):
        files       = chunk_groups[chunk_num]
        chunk_masks = load_chunk(files, is_format_b)

        if not chunk_masks:
            continue

        frame_list = sorted(chunk_masks)
        first_fi   = frame_list[0]

        # Seek video to start of this chunk then read sequentially
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_fi)
        cur = first_fi

        for fi in frame_list:
            # Skip forward to fi using grab() (decode-free, fast)
            while cur < fi:
                if not cap.grab():
                    break
                cur += 1

            ret, frame = cap.read()
            cur += 1
            if not ret:
                break

            masks_dict = chunk_masks[fi]
            view       = classify_view(masks_dict)
            stats[view] += 1

            if view == "unknown":
                continue

            if view == "transverse" and has_unannotated_veins(frame, masks_dict, roi):
                stats["filtered_incomplete"] += 1
                continue

            stem     = f"{case_dir.name}_frame{fi:06d}"
            h, w     = frame.shape[:2]
            mask_png = make_mask_png(masks_dict, h, w)

            if view == "transverse":
                out_f = OUTPUT_DIR / "transverse_view" / "frames"
                out_m = OUTPUT_DIR / "transverse_view" / "masks"
                stats["saved_t"] += 1
            else:
                out_f = OUTPUT_DIR / "longitudinal_view" / "frames"
                out_m = OUTPUT_DIR / "longitudinal_view" / "masks"
                stats["saved_l"] += 1

            cv2.imwrite(str(out_f / (stem + ".jpg")), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out_m / (stem + "_mask.png")), mask_png)

        chunk_masks.clear()     # free memory before next chunk

        saved_so_far = stats["saved_t"] + stats["saved_l"]
        if saved_so_far % 2000 == 0 and saved_so_far > 0:
            print(f"  chunk {chunk_num:04d} done | saved so far: {saved_so_far}")

    cap.release()
    print(f"  Case done")

print(f"\n{'='*60}")
print(f"DONE")
print(f"  Transverse classified : {stats['transverse']}")
print(f"  Longitudinal classified: {stats['longitudinal']}")
print(f"  Unknown / skipped     : {stats['unknown']}")
print(f"  Filtered (incomplete) : {stats['filtered_incomplete']}")
print(f"  Saved transverse      : {stats['saved_t']}")
print(f"  Saved longitudinal    : {stats['saved_l']}")
print(f"  Total saved           : {stats['saved_t'] + stats['saved_l']}")
print(f"\nOutput → {OUTPUT_DIR}")
