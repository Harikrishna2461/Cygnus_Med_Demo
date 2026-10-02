import re, json, numpy as np, cv2
from pathlib import Path

BASE_DIR   = Path(r"c:\Users\Krish\Downloads\videos")
OUTPUT_DIR = BASE_DIR / "annotated_output"
OUTPUT_DIR.mkdir(exist_ok=True)

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

CLASS_COLORS_BGR = [
    ( 80,  80, 255),  # 0  GSV Prox.
    ( 80, 200,  80),  # 1  GSV Distal
    (255,  80,  80),  # 2  Tributary
    (  0, 220, 255),  # 3  SSV
    (220,   0, 220),  # 4  CFV
    (220, 220,   0),  # 5  FV
    (  0, 140, 255),  # 6  DFV
    (255,   0, 160),  # 7  AASV
    (255, 140,   0),  # 8  PASV
    (140,   0, 255),  # 9  Hunt. Perf.
    (  0, 255, 140),  # 10 Dodd Perf.
    (140, 255,   0),  # 11 Boyd Perf.
    (255, 140, 140),  # 12 Cockett Perf.
    (140, 140, 255),  # 13 Ankle Perf.
    (140, 255, 140),  # 14 Escape Point
    (140, 255, 255),  # 15 Re-entry Point
    (255, 255, 140),  # 16 Start Color Doppler
    (255, 140, 255),  # 17 End Color Doppler
    ( 30, 105, 210),  # 18 Start Positive Flow
    ( 50, 205,  50),  # 19 End Positive Flow
    (205,  90, 106),  # 20 Start Pulse Wave
    ( 50, 205, 205),  # 21 End Pulse Wave
    (180, 130,  70),  # 22 Positive Duration
    ( 60,  20, 220),  # 23 PV
    ( 23, 221, 100),  # 24 Deep Vein (Calf)
    (200,  25,  25),  # 25 Thrombose
]


def load_masks(masks_dir):
    """
    Format A  chunk_*.npz        -> value is scalar object array wrapping {cls_id: mask}
    Format B  obj_N_chunk_*.npz  -> value is mask directly; cls_id from filename
    Returns {frame_idx: {cls_id: uint8 mask (H,W)}}
    """
    result = {}
    for npz_path in sorted(Path(masks_dir).glob("*.npz")):
        m = re.match(r"obj_(\d+)_", npz_path.name)
        data = np.load(npz_path, allow_pickle=True)
        if m:
            cls_id = int(m.group(1))
            for key in data.keys():
                fi = int(key)
                if fi not in result:
                    result[fi] = {}
                result[fi][cls_id] = data[key]
        else:
            for key in data.keys():
                result[int(key)] = data[key].item()
    return result


def annotate_frame(frame, masks_dict, roi, alpha=0.4):
    cx1, cy1, cx2, cy2 = roi
    h, w = frame.shape[:2]
    result     = frame.copy()
    color_fill = np.zeros_like(frame)
    deferred   = []

    for cls_id in sorted(masks_dict):
        if cls_id >= len(CLASS_NAMES):
            continue
        mask = masks_dict[cls_id]
        if not mask.any():
            continue
        bgr = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
        color_fill[mask.astype(bool)] = bgr
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        deferred.append((contours, bgr, cls_id))

    filled = color_fill.any(axis=2)
    if filled.any():
        blended = cv2.addWeighted(color_fill, alpha, result, 1.0 - alpha, 0)
        result[filled] = blended[filled]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for contours, bgr, cls_id in deferred:
        cv2.drawContours(result, contours, -1, bgr, 2)
        label = CLASS_NAMES[cls_id]
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            tx = max(3, min(cx, w - tw - 6))
            ty = max(th + 6, min(cy, h - 3))
            cv2.rectangle(result, (tx-3, ty-th-3), (tx+tw+3, ty+3), (15, 15, 15), -1)
            cv2.putText(result, label, (tx, ty), font, 0.45, bgr, 1, cv2.LINE_AA)

    cv2.rectangle(result, (cx1, cy1), (cx2, cy2), (50, 255, 50), 2)
    return result


cases = [d for d in sorted(BASE_DIR.iterdir())
         if d.is_dir() and d.name != "annotated_output"]

for case_dir in cases:
    video_file = case_dir / (case_dir.name + ".mp4")
    masks_dir  = case_dir / "masks"
    roi_file   = case_dir / "roi.json"

    if not video_file.exists() or not masks_dir.exists():
        print(f"[SKIP] {case_dir.name} - missing files")
        continue

    print(f"\n{'='*60}")
    print(f"Processing: {case_dir.name}")

    with open(roi_file) as f:
        roi = json.load(f)["crop_region"]

    cap   = cv2.VideoCapture(str(video_file))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    vw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  Video: {total} frames @ {fps:.2f} fps | {vw}x{vh}")

    print(f"  Loading masks ...", end="", flush=True)
    masks = load_masks(masks_dir)
    annotated_count = sum(1 for md in masks.values() if any(m.any() for m in md.values()))
    print(f" {len(masks)} frames loaded, {annotated_count} with actual annotations")

    out_path = OUTPUT_DIR / (case_dir.name + "_annotated.mp4")
    cap    = cv2.VideoCapture(str(video_file))
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (vw, vh))

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {fi}, stopping")
            break
        masks_dict = masks.get(fi, {})
        out_frame  = annotate_frame(frame, masks_dict, roi) if masks_dict else frame
        writer.write(out_frame)
        if fi % 2000 == 0:
            print(f"  {fi}/{total} frames written ...")

    cap.release()
    writer.release()
    print(f"  Done -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

print(f"\nAll videos saved to: {OUTPUT_DIR}")
