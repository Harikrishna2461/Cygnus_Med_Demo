#!/usr/bin/env python3
"""
Auto-QC flagging for SAM-2 propagated segmentations.

Instead of scrubbing a whole 15-30 minute overlay video frame by frame, this scans
the saved per-frame masks and flags only the handful of frames where propagation
likely broke, so review time scales with how many things actually went wrong, not
with video length.

Heuristics (per object, per frame, compared against the previous frame it appears in):
  - area_jump      mask area changed too fast frame-to-frame (drift / sudden loss / lock-on
                    to the wrong blob)
  - vanished       mask was present and disappeared
  - reappeared     mask was empty and came back (often paired with a vanished flag earlier)
  - too_small      mask present but implausibly small (near-empty, likely a tracking sliver)
  - too_large      mask covers most of the frame (likely leaked / engulfed the whole crop)
  - touches_border mask touches the crop edge (off by default — see below)
  - never_present  (whole-clip check, not per-frame) the object's mask is empty on
                    EVERY frame of the clip. A completely empty mask has zero
                    frame-to-frame variance, so none of the per-frame checks above
                    ever fire on it — this check exists specifically because that blind
                    spot let a fully-empty clip (Video_0/full_scanner_0, both objects,
                    939/939 frames) look "clean" on the first pass of this tool.

None of this understands the content — it's a triage filter, not a verdict. Treat
flagged frames as "look here first," not "these are definitely wrong," and treat
unflagged stretches as "probably fine," not "guaranteed fine."

`touches_border` is OFF by default: these crops are tight around the tracked object
(crop_scanner center-crops on it), so the object legitimately touches the crop edge
often — this isn't a failure signal here. Turning it on against real annotated clips
produced 300+ flagged frames out of ~400, which defeats the point. Pass
--check-border only if your content actually keeps the target away from the edges.

Usage:
    python qc_flags.py --npz path/to/segmentations/full_scanner_0_1_2_3.npz \
        [--frames-dir path/to/cropped_video_clips_frames/full_scanner_0_1_2_3] \
        [--out review/full_scanner_0_1_2_3_flags.json] \
        [--thumbnails-dir review/full_scanner_0_1_2_3_thumbs]
"""

import argparse
import json
import os

import cv2
import numpy as np


def _load_npz_sorted(npz_path: str):
    """Return (frame_indices, masks) where masks[i] has shape [num_obj, H, W] bool,
    sorted by frame index (npz keys are stored as strings and aren't ordered)."""
    data = np.load(npz_path)
    frame_indices = sorted(int(k) for k in data.files)
    masks = [data[str(idx)] for idx in frame_indices]
    return frame_indices, masks


def find_suspect_frames(
    frame_indices,
    masks,
    area_jump_thresh: float = 0.5,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.7,
    border_margin: int = 2,
    min_abs_area_px: int = 25,
    check_border: bool = False,
):
    """
    frame_indices: list[int], ascending, one per entry in `masks`
    masks: list of arrays, each [num_obj, H, W] bool

    Returns a list of {frame_idx, obj_id, reasons, area, area_frac, prev_area} dicts,
    sorted by frame_idx then obj_id.
    """
    if not masks:
        return []

    num_obj = masks[0].shape[0]
    h, w = masks[0].shape[-2:]
    frame_area = h * w

    flags = []
    prev_area = [None] * num_obj  # last-seen area per object (for jump detection)

    for i, frame_idx in enumerate(frame_indices):
        frame_masks = masks[i]
        for obj_id in range(num_obj):
            m = frame_masks[obj_id]
            area = int(m.sum())
            area_frac = area / frame_area
            reasons = []

            if prev_area[obj_id] is not None:
                pa = prev_area[obj_id]
                denom = max(pa, area, 1)
                rel_change = abs(area - pa) / denom
                if rel_change > area_jump_thresh and abs(area - pa) > min_abs_area_px:
                    if area == 0 and pa > 0:
                        reasons.append("vanished")
                    elif pa == 0 and area > 0:
                        reasons.append("reappeared")
                    else:
                        reasons.append("area_jump")

            if area > 0:
                if area_frac < min_area_frac:
                    reasons.append("too_small")
                if area_frac > max_area_frac:
                    reasons.append("too_large")
                if check_border:
                    touches = (
                        m[:border_margin, :].any()
                        or m[-border_margin:, :].any()
                        or m[:, :border_margin].any()
                        or m[:, -border_margin:].any()
                    )
                    if touches:
                        reasons.append("touches_border")

            if reasons:
                flags.append({
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "reasons": reasons,
                    "area": area,
                    "area_frac": round(area_frac, 4),
                    "prev_area": prev_area[obj_id],
                })

            prev_area[obj_id] = area

    return flags


def find_never_present(frame_indices, masks):
    """
    Whole-clip check: objects whose mask is empty on every single frame.

    This exists because a fully-empty mask has zero frame-to-frame variance, so it
    never trips area_jump/vanished/too_small/etc — those are all *change* detectors.
    An object that was never segmented at all needs a dedicated check.

    Returns a list of {obj_id, total_frames} for any object that never appears.
    """
    if not masks:
        return []
    num_obj = masks[0].shape[0]
    any_present = [False] * num_obj
    for frame_masks in masks:
        for obj_id in range(num_obj):
            if frame_masks[obj_id].any():
                any_present[obj_id] = True
    return [
        {"obj_id": obj_id, "total_frames": len(frame_indices)}
        for obj_id in range(num_obj)
        if not any_present[obj_id]
    ]


def _collapse_to_ranges(flags, gap_tolerance: int = 1):
    """Group consecutive/near-consecutive flagged frame_idx (per obj_id) into ranges,
    so a 40-frame drift shows up as one review item instead of 40."""
    by_obj = {}
    for f in flags:
        by_obj.setdefault(f["obj_id"], []).append(f)

    ranges = []
    for obj_id, items in by_obj.items():
        items.sort(key=lambda f: f["frame_idx"])
        cur = None
        for f in items:
            if cur is not None and f["frame_idx"] - cur["end"] <= gap_tolerance:
                cur["end"] = f["frame_idx"]
                cur["reasons"] |= set(f["reasons"])
                cur["count"] += 1
            else:
                if cur is not None:
                    ranges.append(cur)
                cur = {
                    "obj_id": obj_id,
                    "start": f["frame_idx"],
                    "end": f["frame_idx"],
                    "reasons": set(f["reasons"]),
                    "count": 1,
                }
        if cur is not None:
            ranges.append(cur)

    for r in ranges:
        r["reasons"] = sorted(r["reasons"])
    ranges.sort(key=lambda r: r["start"])
    return ranges


def write_thumbnails(ranges, frames_dir, masks_by_idx, out_dir, colors):
    """Write one overlay JPEG per flagged range (its first frame) for quick eyeballing."""
    os.makedirs(out_dir, exist_ok=True)
    frame_names = sorted(
        (p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg", ".png")),
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    for r in ranges:
        idx = r["start"]
        if idx >= len(frame_names):
            continue
        frame = cv2.imread(os.path.join(frames_dir, frame_names[idx]))
        if frame is None:
            continue
        if idx in masks_by_idx:
            frame_masks = masks_by_idx[idx]
            m = frame_masks[r["obj_id"]]
            color = colors[r["obj_id"] % len(colors)]
            overlay = np.zeros_like(frame)
            overlay[m] = color
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
        tag = "_".join(r["reasons"])
        out_path = os.path.join(out_dir, f"obj{r['obj_id']}_frame{idx:05d}_{tag}.jpg")
        cv2.imwrite(out_path, frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", required=True, help="Path to a saved segmentation .npz "
                        "(as written by SamVideoSegmenter.save_segmentations)")
    parser.add_argument("--frames-dir", default=None, help="Frames directory for this clip "
                        "(cropped_video_clips_frames/<clip_name>) — needed only for --thumbnails-dir")
    parser.add_argument("--out", default=None, help="Where to write the JSON flag report "
                        "(default: alongside the npz, same basename + _flags.json)")
    parser.add_argument("--thumbnails-dir", default=None, help="If set, writes one overlay "
                        "JPEG per flagged range here for fast visual triage")
    parser.add_argument("--area-jump-thresh", type=float, default=0.5)
    parser.add_argument("--min-area-frac", type=float, default=0.002)
    parser.add_argument("--max-area-frac", type=float, default=0.7)
    parser.add_argument("--gap-tolerance", type=int, default=1,
                        help="Merge flagged frames into one range if this close together")
    parser.add_argument("--check-border", action="store_true",
                        help="Also flag masks touching the crop edge. OFF by default — "
                        "see module docstring for why.")
    args = parser.parse_args()

    frame_indices, masks = _load_npz_sorted(args.npz)
    print(f"[qc_flags] Loaded {len(frame_indices)} frames, "
          f"{masks[0].shape[0] if masks else 0} object(s), from {args.npz}")

    never_present = find_never_present(frame_indices, masks)
    for np_ in never_present:
        print(f"    [!!] obj {np_['obj_id']}: NEVER PRESENT — mask is empty on all "
              f"{np_['total_frames']} frames. This clip needs re-annotation for this object.")

    flags = find_suspect_frames(
        frame_indices, masks,
        area_jump_thresh=args.area_jump_thresh,
        min_area_frac=args.min_area_frac,
        max_area_frac=args.max_area_frac,
        check_border=args.check_border,
    )
    ranges = _collapse_to_ranges(flags, gap_tolerance=args.gap_tolerance)

    print(f"[qc_flags] {len(flags)} flagged frame-instances -> {len(ranges)} review range(s)")
    for r in ranges:
        span = f"{r['start']}" if r['start'] == r['end'] else f"{r['start']}-{r['end']}"
        print(f"    obj {r['obj_id']}  frames {span:>15}  ({r['count']} flagged)  {', '.join(r['reasons'])}")

    out_path = args.out or (os.path.splitext(args.npz)[0] + "_flags.json")
    with open(out_path, "w") as f:
        json.dump({
            "npz": args.npz,
            "total_frames": len(frame_indices),
            "num_objects": masks[0].shape[0] if masks else 0,
            "ranges": ranges,
        }, f, indent=2)
    print(f"[qc_flags] Report -> {out_path}")

    if args.thumbnails_dir:
        if not args.frames_dir:
            print("[qc_flags] --thumbnails-dir given without --frames-dir, skipping thumbnails")
        else:
            masks_by_idx = dict(zip(frame_indices, masks))
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]
            write_thumbnails(ranges, args.frames_dir, masks_by_idx, args.thumbnails_dir, colors)
            print(f"[qc_flags] Thumbnails -> {args.thumbnails_dir}")


if __name__ == "__main__":
    main()
