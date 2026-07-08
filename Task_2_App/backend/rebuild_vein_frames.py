"""
rebuild_vein_frames.py

Classifies and extracts vein frames using class.txt obj_id mapping,
validated by annotation-colour hue analysis.

Primary classifier: class.txt obj_id -> class name (e.g. obj_10 = Dodd_Perf)
Cross-check:        hue of contour ring + text-label pixels in seg video
If the two disagree, hue wins (some sessions re-use IDs for different veins).

Run from Task_2_App root:
  python backend/rebuild_vein_frames.py
"""

from __future__ import annotations
import json, os, re
import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
VEIN_VIDEOS_BASE = r"C:\Users\Krish\Downloads\Vein_Segmented_Videos"
OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "assets", "vein_frames")
FRAMES_PER_OBJ = 9999   # effectively unlimited — extract from every chunk
MIN_MASK_PX    = 200

# ── class.txt mapping (verified against hue observations) ─────────────────────
_CLASS_BY_OBJID: dict[int, str] = {
    0:  "GSV_Prox",
    1:  "GSV_Distal",
    2:  "Tributary",
    3:  "SSV",
    4:  "CFV",
    5:  "FV",
    6:  "DFV",
    7:  "AASV",
    8:  "PASV",
    9:  "Hunt_Perf",
    10: "Dodd_Perf",
    11: "Boyd_Perf",
    12: "Cockett_Perf",
    13: "Ankle_Perf",
    23: "PV",
    24: "Deep_Vein_Calf",
}

# What hue-class each specific name maps to (for conflict detection)
_EXPECTED_HUE: dict[str, str] = {
    "GSV_Prox":       "GSV",
    "GSV_Distal":     "GSV",
    "Tributary":      "Tributary",
    "SSV":            "SSV",
    "CFV":            "FV_CFV",
    "FV":             "FV_CFV",
    "DFV":            "FV_CFV",
    "PV":             "FV_CFV",
    "Deep_Vein_Calf": "FV_CFV",
    "AASV":           "AASV",
    "PASV":           "AASV",
    "Hunt_Perf":      "Perforator",
    "Dodd_Perf":      "Perforator",
    "Boyd_Perf":      "Perforator",
    "Cockett_Perf":   "Perforator",
    "Ankle_Perf":     "Perforator",
}

# ── Hue -> generic class (OpenCV HSV hue = 0-180) ─────────────────────────────
def _hue_to_generic(h: float | None) -> str | None:
    if h is None:
        return None
    if h < 12 or h > 168:   return "FV_CFV"
    if 12 <= h < 28:         return "Perforator"
    if 28 <= h < 50:         return "Tributary"
    if 50 <= h < 95:         return "GSV"
    if 95 <= h < 140:        return "SSV"
    if 140 <= h <= 168:      return "AASV"
    return None


def resolve_class(obj_id: int, hue_generic: str | None) -> str | None:
    """
    Return the most specific folder name.
    Prefers class.txt name when hue agrees; falls back to generic hue class on mismatch.
    """
    if hue_generic is None:
        return None
    specific = _CLASS_BY_OBJID.get(obj_id)
    if specific is None:
        return hue_generic
    if _EXPECTED_HUE.get(specific) == hue_generic:
        return specific   # class.txt and hue agree
    return hue_generic    # mismatch — trust hue


# ── Hue estimation via contour ring + text label area ────────────────────────

def _estimate_hue(
    mask_dir: str, seg_vid: str, obj_id: int, roi: list[int], n_samples: int = 6
) -> float | None:
    chunks = sorted([
        f for f in os.listdir(mask_dir)
        if re.match(rf"^obj_{obj_id}_chunk", f)
    ])
    if not chunks:
        return None
    step = max(1, len(chunks) // n_samples)
    sampled = chunks[::step][:n_samples]
    cap = cv2.VideoCapture(seg_vid)
    all_hues: list[float] = []
    ker = np.ones((7, 7), np.uint8)

    for chunk_file in sampled:
        try:
            data = np.load(os.path.join(mask_dir, chunk_file), allow_pickle=True)
            keys = sorted(int(k) for k in data.files)
            if not keys:
                data.close(); continue
            frame_idx = keys[len(keys) // 2]
            key_str = str(frame_idx).zfill(6)
            if key_str not in data.files:
                key_str = str(frame_idx)
            raw = data[key_str]; data.close()

            mask = np.asarray(raw)
            if mask.dtype == object:
                mask = mask.astype(np.uint8)
            mask = (mask > 0).astype(np.uint8)
            if mask.sum() < MIN_MASK_PX:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            fc = frame[roi[1]:roi[3], roi[0]:roi[2]]
            mc = mask[roi[1]:roi[3], roi[0]:roi[2]]

            ring = (cv2.dilate(mc, ker, iterations=2).astype(bool)) & (~mc.astype(bool))

            rows_with_mask = np.where(np.any(mc > 0, axis=1))[0]
            if rows_with_mask.size == 0:
                continue
            top = int(rows_with_mask[0])
            text_top = max(0, top - 30)
            text_band = fc[text_top:top, :]

            pixels: list[np.ndarray] = []
            if ring.any():
                pixels.append(fc[ring])
            if text_band.size > 0:
                pixels.append(text_band.reshape(-1, 3))
            if not pixels:
                continue

            combined = np.vstack(pixels).astype(np.uint8)
            hsv = cv2.cvtColor(combined.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            good = (hsv[:, 1] > 80) & (hsv[:, 2] > 60)
            if good.sum() >= 3:
                all_hues.extend(hsv[good, 0].tolist())
        except Exception:
            pass

    cap.release()
    return float(np.median(all_hues)) if all_hues else None


# ── Frame selection ───────────────────────────────────────────────────────────

def _select_frames(mask_dir: str, obj_id: int, n: int) -> list[tuple[str, int]]:
    """
    Return (chunk_path, frame_idx) for every chunk of obj_id.
    n is ignored (kept for signature compat) — all chunks are used.
    Takes up to 3 evenly-spaced frames per chunk for maximum coverage.
    """
    chunks = sorted([
        f for f in os.listdir(mask_dir)
        if re.match(rf"^obj_{obj_id}_chunk", f)
    ])
    if not chunks:
        return []
    result = []
    for chunk_file in chunks:
        try:
            data = np.load(os.path.join(mask_dir, chunk_file), allow_pickle=True)
            keys = sorted(int(k) for k in data.files)
            data.close()
            if not keys:
                continue
            # Up to 3 frames per chunk: first, middle, last
            picks: list[int] = []
            if len(keys) == 1:
                picks = [keys[0]]
            elif len(keys) == 2:
                picks = [keys[0], keys[-1]]
            else:
                picks = [keys[0], keys[len(keys) // 2], keys[-1]]
            for frame_idx in picks:
                result.append((os.path.join(mask_dir, chunk_file), frame_idx))
        except Exception:
            pass
    return result


# ── Save segmented frame ───────────────────────────────────────────────────────

def _get_seg_frame(
    seg_cap: cv2.VideoCapture,
    chunk_path: str,
    frame_idx: int,
    roi: list[int],
) -> np.ndarray | None:
    try:
        data = np.load(chunk_path, allow_pickle=True)
        key_str = str(frame_idx).zfill(6)
        if key_str not in data.files:
            key_str = str(frame_idx)
        raw = data[key_str]; data.close()
        mask = np.asarray(raw)
        if mask.dtype == object:
            mask = mask.astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() < MIN_MASK_PX:
            return None
    except Exception:
        return None

    seg_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = seg_cap.read()
    if not ret:
        return None

    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2].copy()


# ── Find seg video (handles _seg.mp4 and intersection_video.mp4) ──────────────

def _find_seg_video(session_dir: str, session_name: str) -> str | None:
    for candidate in [
        os.path.join(session_dir, session_name + "_seg.mp4"),
        os.path.join(session_dir, "intersection_video.mp4"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


# ── Process one session ───────────────────────────────────────────────────────

def process_session(session_name: str) -> None:
    session_dir = os.path.join(VEIN_VIDEOS_BASE, session_name)
    seg_vid  = _find_seg_video(session_dir, session_name)
    mask_dir = os.path.join(session_dir, "masks")
    roi_file = os.path.join(session_dir, "roi.json")

    if seg_vid is None:
        print(f"  [skip] no segmented video")
        return
    if not os.path.isdir(mask_dir):
        print(f"  [skip] no masks/")
        return

    roi = [150, 50, 495, 410]
    if os.path.isfile(roi_file):
        try:
            r = json.load(open(roi_file)).get("crop_region", roi)
            roi = [r[0], r[1], r[2], r[3]]
        except Exception:
            pass

    files = os.listdir(mask_dir)
    obj_ids = sorted(set(
        int(m.group(1))
        for f in files
        for m in [re.match(r"^obj_(\d+)_chunk", f)]
        if m
    ))
    if not obj_ids:
        print(f"  [skip] no obj_N_chunk mask files (different format)")
        return

    print(f"  seg={os.path.basename(seg_vid)}  obj_ids={obj_ids}")
    seg_cap = cv2.VideoCapture(seg_vid)

    for obj_id in obj_ids:
        hue      = _estimate_hue(mask_dir, seg_vid, obj_id, roi)
        generic  = _hue_to_generic(hue)
        cls      = resolve_class(obj_id, generic)
        hue_s    = f"{hue:.0f}" if hue is not None else "None"

        if cls is None:
            # hue detection failed — try class.txt name as last resort
            cls = _CLASS_BY_OBJID.get(obj_id)
            if cls is None:
                print(f"  obj_{obj_id}: hue={hue_s} -> SKIP")
                continue
            print(f"  obj_{obj_id}: hue={hue_s} -> {cls} (class.txt fallback, hue undetected)")
        else:
            print(f"  obj_{obj_id}: hue={hue_s} -> {cls}")

        frame_list = _select_frames(mask_dir, obj_id, FRAMES_PER_OBJ)
        out_dir    = os.path.join(OUT_BASE, cls)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        for chunk_path, frame_idx in frame_list:
            out_path = os.path.join(out_dir, f"{session_name}_obj{obj_id:02d}_f{frame_idx:06d}.jpg")
            if os.path.isfile(out_path):
                saved += 1
                continue
            img = _get_seg_frame(seg_cap, chunk_path, frame_idx, roi)
            if img is not None:
                cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                saved += 1

        print(f"    saved {saved}/{len(frame_list)} frames -> {cls}/")

    seg_cap.release()


def main() -> None:
    import shutil
    if os.path.isdir(OUT_BASE):
        shutil.rmtree(OUT_BASE)
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"Output: {os.path.abspath(OUT_BASE)}\n")

    sessions = [
        d for d in sorted(os.listdir(VEIN_VIDEOS_BASE))
        if os.path.isdir(os.path.join(VEIN_VIDEOS_BASE, d))
    ]
    for sess in sessions:
        print(f"\n=== {sess} ===")
        process_session(sess)

    print("\n\n=== Final summary ===")
    for d in sorted(os.listdir(OUT_BASE)):
        dpath = os.path.join(OUT_BASE, d)
        if os.path.isdir(dpath):
            n = len([f for f in os.listdir(dpath) if f.endswith(".jpg")])
            print(f"  {d:20s}: {n} frames")


if __name__ == "__main__":
    main()
