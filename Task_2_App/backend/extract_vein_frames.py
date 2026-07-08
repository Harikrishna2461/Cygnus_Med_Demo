"""
extract_vein_frames.py

One-time setup script: scan all sessions in Vein_Segmented_Videos, find frames
that contain each vein type from the segmentation masks, extract them from the
segmented video (_seg.mp4), apply the ROI crop, and save to:

  assets/vein_frames/<VeinName>/

Run from the Task_2_App directory:
  python backend/extract_vein_frames.py
"""

from __future__ import annotations
import json
import os
import re
import sys

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
VEIN_VIDEOS_BASE = r"C:\Users\Krish\Downloads\Vein_Segmented_Videos"
OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "assets", "vein_frames")

FRAMES_PER_VEIN_PER_SESSION = 8  # frames to extract per vein type per session

# ── Class mapping (index → folder name) ───────────────────────────────────────
CLASS_NAMES = [
    "GSV_Prox",         # 0
    "GSV_Distal",       # 1
    "Tributary",        # 2
    "SSV",              # 3
    "CFV",              # 4
    "FV",               # 5
    "DFV",              # 6
    "AASV",             # 7
    "PASV",             # 8
    "Hunt_Perf",        # 9
    "Dodd_Perf",        # 10
    "Boyd_Perf",        # 11
    "Cockett_Perf",     # 12
    "Ankle_Perf",       # 13
    "Escape_Point",     # 14
    "Re_entry_Point",   # 15
    "Start_Color_Doppler", "End_Color_Doppler",
    "Start_Positive_Flow", "End_Positive_Flow",
    "Start_Pulse_Wave",    "End_Pulse_Wave",
    "Positive_Duration",
    "PV",               # 23
    "Deep_Vein_Calf",   # 24
    "Thrombose",        # 25
]

# Only extract these types — the ones relevant to our app regions
WANTED_OBJ_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24}


def _sample_chunks(chunk_files: list[str], n: int) -> list[str]:
    """Pick n evenly-spaced chunk files."""
    if len(chunk_files) <= n:
        return chunk_files
    step = len(chunk_files) / n
    return [chunk_files[int(i * step)] for i in range(n)]


def _get_middle_frame(npz_path: str) -> int | None:
    """Load an NPZ chunk and return the middle frame index."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        keys = []
        for k in data.files:
            try:
                keys.append(int(k))
            except (ValueError, TypeError):
                pass
        data.close()
        if not keys:
            return None
        keys.sort()
        return keys[len(keys) // 2]
    except Exception as exc:
        print(f"    [warn] Could not load {os.path.basename(npz_path)}: {exc}")
        return None


def _extract_and_save(
    video_path: str,
    frame_idx: int,
    crop: list[int],
    out_path: str,
) -> bool:
    """Seek to frame_idx, crop, and save as JPEG."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    x1, y1, x2, y2 = crop
    cropped = frame[y1:y2, x1:x2]
    cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


def process_session(session_dir: str, session_name: str) -> None:
    seg_video = os.path.join(session_dir, session_name + "_seg.mp4")
    mask_dir  = os.path.join(session_dir, "masks")
    roi_file  = os.path.join(session_dir, "roi.json")

    if not os.path.isfile(seg_video):
        print(f"  [skip] no _seg.mp4 found")
        return
    if not os.path.isdir(mask_dir):
        print(f"  [skip] no masks/ directory")
        return

    # Crop region
    crop = [150, 50, 495, 410]
    if os.path.isfile(roi_file):
        try:
            with open(roi_file) as f:
                r = json.load(f).get("crop_region", crop)
                crop = [r[0], r[1], r[2], r[3]]
        except Exception:
            pass

    # Discover all obj_ids present
    all_masks = [f for f in os.listdir(mask_dir) if f.endswith(".npz")]
    obj_ids: set[int] = set()
    for fname in all_masks:
        m = re.match(r"^obj_(\d+)_chunk", fname)
        if m:
            obj_ids.add(int(m.group(1)))

    if not obj_ids:
        print(f"  [skip] no obj_*.npz mask files found")
        return

    print(f"  obj_ids present: {sorted(obj_ids)}")

    for obj_id in sorted(obj_ids):
        if obj_id not in WANTED_OBJ_IDS or obj_id >= len(CLASS_NAMES):
            continue

        vein_name = CLASS_NAMES[obj_id]
        chunk_files = sorted([
            os.path.join(mask_dir, f) for f in all_masks
            if re.match(rf"^obj_{obj_id}_chunk", f)
        ])

        if not chunk_files:
            continue

        sampled = _sample_chunks(chunk_files, FRAMES_PER_VEIN_PER_SESSION)

        out_dir = os.path.join(OUT_BASE, vein_name)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        for chunk_path in sampled:
            fidx = _get_middle_frame(chunk_path)
            if fidx is None:
                continue
            out_path = os.path.join(out_dir, f"{session_name}_f{fidx:06d}.jpg")
            if os.path.isfile(out_path):
                saved += 1
                continue
            if _extract_and_save(seg_video, fidx, crop, out_path):
                saved += 1

        print(f"    {vein_name:20s}: {saved}/{len(sampled)} frames saved")


def main() -> None:
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUT_BASE)}\n")

    sessions = [
        d for d in sorted(os.listdir(VEIN_VIDEOS_BASE))
        if os.path.isdir(os.path.join(VEIN_VIDEOS_BASE, d))
    ]

    for session_name in sessions:
        session_dir = os.path.join(VEIN_VIDEOS_BASE, session_name)
        print(f"\n=== {session_name} ===")
        process_session(session_dir, session_name)

    # Summary
    print("\n\n=== Summary ===")
    if os.path.isdir(OUT_BASE):
        for vein_dir in sorted(os.listdir(OUT_BASE)):
            vpath = os.path.join(OUT_BASE, vein_dir)
            if os.path.isdir(vpath):
                count = len([f for f in os.listdir(vpath) if f.endswith(".jpg")])
                print(f"  {vein_dir:25s}: {count} frames")


if __name__ == "__main__":
    main()
