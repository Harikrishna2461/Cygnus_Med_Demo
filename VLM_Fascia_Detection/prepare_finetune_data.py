"""
Converts Task_3_Experimental labeled frames into BiomedParse fine-tuning format.

Output structure (inside BiomedParse/biomedparse_datasets/Fascia_Detection/):
  train/          1024×1024 RGB PNGs  — {id:06d}_Ultrasound_PeripheralVascular.png
  train_mask/     1024×1024 binary    — {id:06d}_Ultrasound_PeripheralVascular_fascia+layer.png
                                        {id:06d}_Ultrasound_PeripheralVascular_vein.png
  test/           (same naming)
  test_mask/      (same naming)

Test split strategy:
  Fascia — video "Knee-Ankle-Seg1" held out (79 frames, anatomically different site)
  Vein   — last 15% of each video held out (to match test video distribution)
"""

import csv
import os
import sys
import shutil
from pathlib import Path
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
TASK3_BASE = Path(r"C:\Users\Krish\Downloads\Cygnus_Med_Demo\Task_3_Experimental\vein_detection_task_3_training")
FASCIA_CSV = TASK3_BASE / "output/fascia/metadata.csv"
VEIN_CSV   = TASK3_BASE / "output/vein/metadata.csv"

BIOMEDPARSE = Path(r"C:\Users\Krish\Downloads\Cygnus_Med_Demo\VLM_Fascia_Detection\BiomedParse")
OUT_ROOT = BIOMEDPARSE / "biomedparse_datasets" / "Fascia_Detection"

IMG_SIZE = 1024
FASCIA_TEST_VIDEO = "Knee-Ankle-Seg1"
VEIN_TEST_FRACTION = 0.15

# ── Helpers ──────────────────────────────────────────────────────────────────
def ensure_dirs():
    for d in ["train", "train_mask", "test", "test_mask"]:
        (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)

def load_and_resize(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return img

def load_mask_and_resize(path: Path) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != (IMG_SIZE, IMG_SIZE):
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return mask

def save_frame(img: Image.Image, split: str, idx: int) -> str:
    fname = f"{idx:06d}_Ultrasound_PeripheralVascular.png"
    img.save(OUT_ROOT / split / fname)
    return fname

def save_mask(mask: Image.Image, split: str, idx: int, target_tag: str) -> str:
    fname = f"{idx:06d}_Ultrasound_PeripheralVascular_{target_tag}.png"
    mask.save(OUT_ROOT / f"{split}_mask" / fname)
    return fname

# ── Load CSVs ────────────────────────────────────────────────────────────────
def load_fascia_rows():
    with open(FASCIA_CSV) as f:
        rows = [r for r in csv.DictReader(f)
                if r["has_fascia"] == "True" and r["valid_mask"] == "True"]
    return rows

def load_vein_rows():
    with open(VEIN_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["has_vein"] == "True"]
    return rows

# ── Per-label ID counters (shared so each output file gets a unique global ID)
_global_id = {"train": 0, "test": 0}

# ── Process fascia ────────────────────────────────────────────────────────────
def process_fascia(rows):
    train_rows = [r for r in rows if r["video"] != FASCIA_TEST_VIDEO]
    test_rows  = [r for r in rows if r["video"] == FASCIA_TEST_VIDEO]
    print(f"Fascia — train: {len(train_rows)}, test: {len(test_rows)}")

    # We need a shared frame index that may also receive vein annotations.
    # Return a mapping from (video, frame_path) -> global_id so vein can reuse
    # frames that already exist in the output directory.
    frame_index = {}   # (split, frame_path_str) -> global_id

    for split, split_rows in [("train", train_rows), ("test", test_rows)]:
        for row in split_rows:
            fp  = TASK3_BASE / row["frame_path"]
            mp  = TASK3_BASE / row["mask_path"]
            key = (split, row["frame_path"])

            if key not in frame_index:
                gid = _global_id[split]
                _global_id[split] += 1
                frame_index[key] = gid
                img = load_and_resize(fp)
                save_frame(img, split, gid)
            else:
                gid = frame_index[key]

            mask = load_mask_and_resize(mp)
            save_mask(mask, split, gid, "fascia+layer")

        print(f"  {split}: wrote {sum(1 for k in frame_index if k[0]==split)} frames")

    return frame_index

# ── Process vein ──────────────────────────────────────────────────────────────
def process_vein(rows, fascia_frame_index):
    # Group by video, hold out last VEIN_TEST_FRACTION
    from collections import defaultdict
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)

    train_rows, test_rows = [], []
    for video, vrows in by_video.items():
        vrows.sort(key=lambda r: r["frame_idx"])
        n_test = max(1, int(len(vrows) * VEIN_TEST_FRACTION))
        train_rows.extend(vrows[:-n_test])
        test_rows.extend(vrows[-n_test:])

    print(f"Vein   — train: {len(train_rows)}, test: {len(test_rows)}")

    for split, split_rows in [("train", train_rows), ("test", test_rows)]:
        new_frames = 0
        for row in split_rows:
            fp  = TASK3_BASE / row["frame_path"]
            mp  = TASK3_BASE / row["mask_path"]

            # Vein frames don't overlap with fascia frames (different file naming)
            # so always allocate a new global ID
            gid = _global_id[split]
            _global_id[split] += 1
            img = load_and_resize(fp)
            save_frame(img, split, gid)
            new_frames += 1

            mask = load_mask_and_resize(mp)
            save_mask(mask, split, gid, "vein")

        print(f"  {split}: wrote {new_frames} vein frames")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_dirs()

    print("Loading fascia rows...")
    fascia_rows = load_fascia_rows()
    print(f"  {len(fascia_rows)} valid fascia rows")

    print("Loading vein rows...")
    vein_rows = load_vein_rows()
    print(f"  {len(vein_rows)} valid vein rows")

    print("\nProcessing fascia...")
    fascia_frame_index = process_fascia(fascia_rows)

    print("\nProcessing vein...")
    process_vein(vein_rows, fascia_frame_index)

    total_train = _global_id["train"]
    total_test  = _global_id["test"]
    print(f"\nDone. Output: {OUT_ROOT}")
    print(f"  train/ : {total_train} frames")
    print(f"  test/  : {total_test} frames")
    print(f"\nNext: run create-customer-datasets.py from BiomedParse/biomedparse_datasets/")
    print(f"  python create-customer-datasets.py  (after setting targetpath = 'Fascia_Detection')")
