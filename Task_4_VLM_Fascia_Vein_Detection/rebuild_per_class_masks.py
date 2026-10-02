"""
Replace combined colour masks with per-class binary masks.

Reads the already-extracted frame filenames, reloads the NPZ masks,
and writes one 0/255 binary PNG per (frame × class).

Output filename: {case}_frame{fi:06d}_cls{cls_id:02d}_{safe_name}.png
  255 = vein present, 0 = background

Frames are NOT re-extracted (video is not read again).
"""
import re, sys, numpy as np, cv2
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR   = Path(r"c:\Users\Krish\Downloads\videos")
OUTPUT_DIR = BASE_DIR / "veins_vlm_finetuning"
SKIP_DIRS  = {"annotated_output", "veins_vlm_finetuning"}

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

def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

def group_chunks(masks_dir):
    all_npz = sorted(masks_dir.glob("*.npz"))
    is_b = any(re.match(r"obj_\d+_", f.name) for f in all_npz)
    groups = defaultdict(list)
    for npz in all_npz:
        if is_b:
            m = re.match(r"obj_(\d+)_chunk_(\d+)", npz.name)
            if m: groups[int(m.group(2))].append((int(m.group(1)), npz))
        else:
            m = re.match(r"chunk_(\d+)", npz.name)
            if m: groups[int(m.group(1))].append((None, npz))
    return is_b, dict(sorted(groups.items()))

def load_chunk(files, is_format_b):
    result = {}
    if is_format_b:
        for cls_id, npz_path in files:
            data = np.load(npz_path, allow_pickle=True)
            for key in data.keys():
                fi = int(key); mask = data[key]
                if mask.any():
                    if fi not in result: result[fi] = {}
                    result[fi][cls_id] = mask.copy()
            data.close()
    else:
        _, npz_path = files[0]
        data = np.load(npz_path, allow_pickle=True)
        for key in data.keys():
            fi = int(key); md = data[key].item()
            active = {c: m.copy() for c, m in md.items() if m.any()}
            if active: result[fi] = active
        data.close()
    return result


# ── Step 1: wipe old combined masks ───────────────────────────────────────────
for view in ("transverse_view", "longitudinal_view"):
    old_masks = list((OUTPUT_DIR / view / "masks").glob("*.png"))
    print(f"Deleting {len(old_masks)} old combined masks in {view}/masks/")
    for f in old_masks:
        f.unlink()

total_written = 0

# ── Step 2: for each case, rebuild per-class binary masks ─────────────────────
for case_dir in sorted(BASE_DIR.iterdir()):
    if not case_dir.is_dir() or case_dir.name in SKIP_DIRS:
        continue
    masks_dir = case_dir / "masks"
    if not masks_dir.exists():
        continue

    # Collect which frames were saved (from both views)
    saved_frames = {}   # frame_idx -> view  ('transverse_view' or 'longitudinal_view')
    pattern = re.compile(rf"^{re.escape(case_dir.name)}_frame(\d+)\.jpg$")
    for view in ("transverse_view", "longitudinal_view"):
        for jpg in (OUTPUT_DIR / view / "frames").glob(f"{case_dir.name}_frame*.jpg"):
            m = pattern.match(jpg.name)
            if m:
                saved_frames[int(m.group(1))] = view

    if not saved_frames:
        print(f"[SKIP] {case_dir.name} – no saved frames found")
        continue

    print(f"\n{case_dir.name} | {len(saved_frames)} frames to remask")

    is_format_b, chunk_groups = group_chunks(masks_dir)
    written_this_case = 0

    for chunk_num in sorted(chunk_groups):
        chunk_masks = load_chunk(chunk_groups[chunk_num], is_format_b)

        for fi, masks_dict in chunk_masks.items():
            if fi not in saved_frames:
                continue
            view     = saved_frames[fi]
            out_dir  = OUTPUT_DIR / view / "masks"
            stem     = f"{case_dir.name}_frame{fi:06d}"

            for cls_id, mask in masks_dict.items():
                if cls_id >= len(CLASS_NAMES) or not mask.any():
                    continue
                cname   = safe_name(CLASS_NAMES[cls_id])
                outname = f"{stem}_cls{cls_id:02d}_{cname}.png"
                # binary: 255 where vein present, 0 elsewhere
                cv2.imwrite(str(out_dir / outname),
                            (mask.astype(np.uint8) * 255))
                written_this_case += 1

        chunk_masks.clear()

    total_written += written_this_case
    print(f"  {written_this_case} per-class masks written")

print(f"\nDone. Total per-class masks written: {total_written}")

# ── Step 3: final counts ──────────────────────────────────────────────────────
for view in ("transverse_view", "longitudinal_view"):
    nf = len(list((OUTPUT_DIR / view / "frames").glob("*.jpg")))
    nm = len(list((OUTPUT_DIR / view / "masks").glob("*.png")))
    print(f"  {view}: {nf} frames, {nm} masks  (avg {nm/nf:.1f} masks/frame)")
