"""
remove_doppler_frames.py

Scans every .jpg in assets/vein_frames/ and assets/guidance/ and removes
frames that show active Color Doppler.

Color Doppler detection:
  Doppler floods a large rectangular area with saturated red (H<15 or H>165)
  AND saturated blue (H 95-140) simultaneously.  Annotation overlays are
  single-colour contour rings that cover <2 % of the frame each.
  We require BOTH colours to cover ≥ 2.5 % of the image independently,
  which is comfortably above any annotation artefact.

Run from Task_2_App root:
  python backend/remove_doppler_frames.py [--dry-run]
"""

from __future__ import annotations
import os, sys, argparse
import cv2
import numpy as np

ASSETS_BASE = os.path.join(os.path.dirname(__file__), "..", "assets")
SCAN_DIRS = [
    os.path.join(ASSETS_BASE, "vein_frames"),
    os.path.join(ASSETS_BASE, "guidance"),
]

# Fraction of total pixels that must be saturated-red AND saturated-blue
# for the frame to be classified as having active Color Doppler.
MIN_SAT = 70    # permissive saturation floor
MIN_VAL = 40    # permissive brightness floor


def has_doppler(img_bgr: np.ndarray) -> bool:
    """
    Maximum-sensitivity Doppler detection — four independent triggers.
    Prefer false positives (safe to lose a frame) over false negatives.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    active    = (s > MIN_SAT) & (v > MIN_VAL)
    total     = float(img_bgr.shape[0] * img_bgr.shape[1])

    red_mask  = active & ((h < 15) | (h > 165))
    blue_mask = active & (h > 90) & (h < 140)

    red_px   = int(red_mask.sum())
    blue_px  = int(blue_mask.sum())
    red_frac = red_px  / total
    blue_frac= blue_px / total

    # 1. Large single-colour flood (5 % of frame in one Doppler colour)
    if blue_frac >= 0.05 or red_frac >= 0.05:
        return True

    # 2. Both colours present — even a small Doppler box triggers this
    if red_frac >= 0.003 and blue_frac >= 0.003:
        return True

    # 3. Tight spatial proximity: any 20 red+blue pixels within 80 px of each other
    if red_px >= 20 and blue_px >= 20:
        ker = np.ones((80, 80), np.uint8)
        red_dilated = cv2.dilate(red_mask.astype(np.uint8), ker)
        blue_near_red = int((red_dilated.astype(bool) & blue_mask).sum())
        if blue_near_red >= 20:
            return True

    # 4. Overall saturation burden: if >6 % of ALL pixels are highly saturated
    #    (S>120) that's far more than any annotation can produce — must be Doppler
    high_sat = int(((s > 120) & (v > MIN_VAL)).sum())
    if high_sat / total >= 0.06:
        return True

    return False


# Sessions known to be heavily Doppler-contaminated — purge ALL their frames
CONTAMINATED_SESSIONS = {"202401290936_01"}


def scan_dir(root: str, dry_run: bool) -> tuple[int, int]:
    """Walk root recursively. Returns (checked, removed)."""
    checked = removed = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".jpg"):
                continue
            fpath = os.path.join(dirpath, fname)
            checked += 1

            # Purge entire contaminated sessions without pixel analysis
            session_id = fname.split("_obj")[0] if "_obj" in fname else ""
            if session_id in CONTAMINATED_SESSIONS:
                rel = os.path.relpath(fpath, ASSETS_BASE)
                if dry_run:
                    print(f"  [contaminated session] {rel}")
                else:
                    os.remove(fpath)
                    print(f"  [deleted] {rel}")
                removed += 1
                continue

            img = cv2.imread(fpath)
            if img is None:
                continue
            if has_doppler(img):
                rel = os.path.relpath(fpath, ASSETS_BASE)
                if dry_run:
                    print(f"  [doppler] {rel}")
                else:
                    os.remove(fpath)
                    print(f"  [deleted] {rel}")
                removed += 1
    return checked, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without removing anything")
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "DELETING"
    print(f"Color Doppler frame removal — {mode}\n")

    total_checked = total_removed = 0
    for scan_dir_path in SCAN_DIRS:
        if not os.path.isdir(scan_dir_path):
            continue
        print(f"Scanning {scan_dir_path} ...")
        c, r = scan_dir(scan_dir_path, args.dry_run)
        total_checked += c
        total_removed += r

    print(f"\nDone. Checked {total_checked} frames, "
          f"{'would remove' if args.dry_run else 'removed'} {total_removed}.")


if __name__ == "__main__":
    main()
