# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
Batch auto-annotation + SAM-2 segmentation.

Processes full_scanner_24 through full_scanner_32 from the crop_scanner/result
folder, automatically annotates body (id=0) and tail (id=1) on a handful of
reference frames, runs SAM-2 propagation across the full video, and writes
overlay .mp4 files to automated_annotated_videos/.

Run from the scanner_annotator/ directory:
    python batch_annotate.py

Optional flags:
    --start 24      first video number (default 24)
    --end   32      last video number inclusive (default 32)
    --model TINY    SAM model: TINY SMALL BASE_PLUS LARGE (default TINY)
    --n-ref 4       reference frames per video to annotate (default 4)
"""

import argparse
import json
import os
import sys

# ── Working-dir guard (relative paths inside sam_segmentation.py) ────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import cv2
import numpy as np
import torch

from segmentation.sam_segmentation import SamVideoSegmenter
from segmentation.sam_models import SamModel

# ── Paths ────────────────────────────────────────────────────────────────────
RESULT_DIR  = r"C:\Users\Krish\Desktop\scanner-annotation\crop_scanner\result"
OUT_DIR     = r"C:\Users\Krish\Desktop\scanner-annotation\automated_annotated_videos"


# ═══════════════════════════════════════════════════════════════════════════════
#  Detection helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_mask(mask: np.ndarray, n: int,
                 dy: int = 0, dx: int = 0) -> list:
    """Return up to n evenly-spaced [x, y] points from a binary mask."""
    pts = np.argwhere(mask > 0)          # rows=y, cols=x
    if len(pts) == 0:
        return []
    step = max(1, len(pts) // n)
    return [[int(p[1]) + dx, int(p[0]) + dy] for p in pts[::step][:n]]


def _contour_mask(contour, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    return m


def _elongation(c) -> float:
    x, y, bw, bh = cv2.boundingRect(c)
    return max(bw, bh) / (min(bw, bh) + 1e-6)


def find_body_tail_points(frame: np.ndarray,
                          n_body: int = 8,
                          n_tail: int = 5) -> tuple:
    """
    Returns (body_pts, tail_pts, neg_pts) each as list of [x, y].

    Strategy
    --------
    1. YCrCb skin detection  -> find non-skin blobs.
    2. Also detect low-saturation gray/dark regions (probe + cable).
    3. Combine.  Largest blob ->body;  most-elongated remaining blob ->tail.
    4. If <2 blobs: split the body blob at centroid for tail.
    5. Hard fallback: centre of frame for body, edge region for tail.
    """
    h, w = frame.shape[:2]
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Non-skin
    skin       = cv2.inRange(ycrcb, (0, 130, 75), (255, 178, 135))
    non_skin   = cv2.bitwise_not(skin)

    # Low-saturation non-white (probe & cable are grayish)
    low_sat    = cv2.inRange(hsv, (0, 0, 20), (180, 70, 210))

    combined   = cv2.bitwise_or(non_skin, low_sat)

    kernel     = np.ones((5, 5), np.uint8)
    combined   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined   = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    min_area    = h * w * 0.008          # ≥ 0.8 % of frame
    sig         = sorted([c for c in contours if cv2.contourArea(c) > min_area],
                         key=cv2.contourArea, reverse=True)

    body_pts: list = []
    tail_pts:  list = []

    if sig:
        # ── Body: largest blob ──────────────────────────────────────────────
        bm = _contour_mask(sig[0], h, w)
        body_pts = _sample_mask(bm, n_body)

        # ── Tail: most-elongated of remaining blobs ─────────────────────────
        if len(sig) > 1:
            tail_cnt = max(sig[1:4], key=_elongation)
            tm = _contour_mask(tail_cnt, h, w)
            tail_pts = _sample_mask(tm, n_tail)

    # ── Split-body fallback ─────────────────────────────────────────────────
    if not tail_pts and sig:
        M = cv2.moments(sig[0])
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bm = _contour_mask(sig[0], h, w)
            x, y, bw, bh = cv2.boundingRect(sig[0])
            if bh >= bw:                 # tall ->split horizontally
                half_a, half_b = bm[:cy, :], bm[cy:, :]
                dy_b = cy
            else:                        # wide ->split vertically
                half_a, half_b = bm[:, :cx], bm[:, cx:]
                dy_b = 0
            large, small = (half_a, half_b) if np.sum(half_a) >= np.sum(half_b) \
                           else (half_b, half_a)
            large_dy = cy if (large is half_b and bh >= bw) else 0
            small_dy = dy_b if (small is half_b and bh >= bw) else 0
            body_pts = _sample_mask(large, n_body, dy=large_dy)
            tail_pts  = _sample_mask(small, n_tail,  dy=small_dy)

    # ── Hard fallback: centre for body, bottom-right for tail ───────────────
    if not body_pts:
        body_pts = [
            [w // 2,       h // 2],
            [w // 2 - 15,  h // 2 - 15],
            [w // 2 + 15,  h // 2 + 15],
            [w // 2,       h // 2 + 20],
            [w // 2,       h // 2 - 20],
        ]
    if not tail_pts:
        tail_pts = [
            [w - 20, h // 2],
            [w - 20, h // 2 - 15],
            [w - 20, h // 2 + 15],
        ]

    # Negative points: corners (definitely background / skin)
    margin   = 10
    neg_pts  = [
        [margin,    margin],
        [w - margin, margin],
        [margin,    h - margin],
        [w - margin, h - margin],
        [w // 2,    margin],
    ]

    return body_pts, tail_pts, neg_pts


# ═══════════════════════════════════════════════════════════════════════════════
#  Annotation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def choose_ref_frames(total: int, n: int) -> list:
    """Return n frame indices spread across [early … late]."""
    if total <= n:
        return list(range(total))
    # Avoid very first / last frames (may be blank or motion-blurred)
    start = max(3, int(total * 0.03))
    end   = min(total - 4, int(total * 0.97))
    step  = max(1, (end - start) // (n - 1))
    idxs  = [start + i * step for i in range(n)]
    return idxs[:n]


def annotate_frame(segmenter: SamVideoSegmenter,
                   frame_path: str,
                   frame_idx: int):
    """Detect body/tail in frame_path and feed points into the segmenter."""
    frame = cv2.imread(frame_path)
    body_pts, tail_pts, neg_pts = find_body_tail_points(frame)

    # BODY (obj_id=0): positive body + negative corners
    all_body_pts = np.array(body_pts + neg_pts,  dtype=float)
    all_body_lbs = np.array([1] * len(body_pts) + [0] * len(neg_pts), dtype=int)
    segmenter.add_points(frame_idx, obj_id=0, points=all_body_pts, labels=all_body_lbs)

    # TAIL (obj_id=1): positive tail + negative corners
    if tail_pts:
        all_tail_pts = np.array(tail_pts + neg_pts, dtype=float)
        all_tail_lbs = np.array([1] * len(tail_pts) + [0] * len(neg_pts), dtype=int)
        segmenter.add_points(frame_idx, obj_id=1, points=all_tail_pts, labels=all_tail_lbs)

    return len(body_pts), len(tail_pts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Video writing
# ═══════════════════════════════════════════════════════════════════════════════

def write_output_video(segmenter: SamVideoSegmenter,
                       frame_count: int,
                       out_path: str,
                       fps: float = 30.0):
    """Write a side-by-side (original | segmented) .mp4 to out_path."""
    sample_orig = segmenter.get_frame(0)     # RGB numpy
    h, w = sample_orig.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

    for idx in range(frame_count):
        orig_bgr = cv2.cvtColor(segmenter.get_frame(idx), cv2.COLOR_RGB2BGR)
        masked   = segmenter.get_masked_frame(idx)
        if masked is None:
            masked = orig_bgr

        side_by_side = np.concatenate([orig_bgr, masked], axis=1)
        writer.write(side_by_side)

    writer.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(video_path: str,
                  segmenter: SamVideoSegmenter,
                  n_ref: int,
                  out_dir: str) -> bool:
    """
    Full pipeline for one video.
    Returns True on success.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video  = os.path.join(out_dir, f"{video_name}_segmented.mp4")
    out_json   = os.path.join(out_dir, f"{video_name}_annotations.json")
    print(f"\n{'='*60}")
    print(f"  Processing: {video_name}")
    print(f"{'='*60}")

    # ── 1. Update segmenter for this video ───────────────────────────────────
    segmenter.video_path  = video_path
    segmenter.video_name  = video_name
    segmenter.frames_dir  = None
    segmenter.frame_names = []
    segmenter.video_segments = {}
    segmenter.obj_ids     = set()
    segmenter.inference_state = None

    # ── 2. Extract frames ────────────────────────────────────────────────────
    print("  [1/4] Extracting frames …")
    segmenter.extract_video_frames()
    total_frames = len(segmenter.frame_names)
    print(f"        {total_frames} frames extracted to {segmenter.frames_dir}")

    # ── 3. Init SAM inference state ──────────────────────────────────────────
    print("  [2/4] Initialising SAM inference state …")
    segmenter.initialize_inference()

    # ── 4. Auto-annotate reference frames ────────────────────────────────────
    ref_frames   = choose_ref_frames(total_frames, n_ref)
    annotations  = {}                    # frame_idx ->{body:[], tail:[], neg:[]}
    has_frame0   = 0 in ref_frames

    print(f"  [3/4] Annotating {n_ref} reference frames: {ref_frames}")
    for fidx in ref_frames:
        frame_path = os.path.join(segmenter.frames_dir,
                                  segmenter.frame_names[fidx])
        nb, nt = annotate_frame(segmenter, frame_path, fidx)
        print(f"        frame {fidx:4d}: body={nb} pts  tail={nt} pts")

        # Record for JSON log
        frame_img  = cv2.imread(frame_path)
        bp, tp, np_ = find_body_tail_points(frame_img)
        annotations[fidx] = {"body": bp, "tail": tp, "neg": np_}

    # Hidden background points on frame 0 if not annotated
    # (required by SAM-2 for objects that start mid-video)
    if not has_frame0:
        for oid in [0, 1]:
            segmenter.add_points(0, obj_id=oid,
                                 points=np.array([[0, 0]], dtype=float),
                                 labels=np.array([0]))
        print("        frame    0: hidden BG sentinel added")

    # ── 5. Propagate ─────────────────────────────────────────────────────────
    print("  [4/4] Running SAM-2 propagation …")
    segmenter.propagate()
    print(f"        Propagated {len(segmenter.video_segments)} frames")

    # ── 6. Write output video ────────────────────────────────────────────────
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print(f"  Writing ->{out_video}")
    write_output_video(segmenter, total_frames, out_video, fps)

    # ── 7. Save annotation JSON ──────────────────────────────────────────────
    with open(out_json, "w") as f:
        json.dump({"ref_frames": ref_frames, "frame_data": {
            str(k): v for k, v in annotations.items()
        }}, f, indent=2)
    print(f"  Annotations ->{out_json}")

    # ── 8. Reset for next video ──────────────────────────────────────────────
    segmenter.reset()
    torch.cuda.empty_cache()

    print(f"  [OK] {video_name} complete")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  type=int, default=24)
    parser.add_argument("--end",    type=int, default=32)
    parser.add_argument("--model",  default="TINY",
                        choices=[m.name for m in SamModel])
    parser.add_argument("--n-ref",  type=int, default=4,
                        help="Reference frames per video (default 4)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = SamModel[args.model]

    video_nums = range(args.start, args.end + 1)
    video_paths = []
    for n in video_nums:
        p = os.path.join(RESULT_DIR, f"full_scanner_{n}.mp4")
        if os.path.exists(p):
            video_paths.append(p)
        else:
            print(f"[WARN] Not found, skipping: {p}")

    if not video_paths:
        print("[ERROR] No videos found.")
        sys.exit(1)

    print(f"\nFound {len(video_paths)} videos  |  model={model.name}  |  ref_frames={args.n_ref}")
    print(f"Output ->{OUT_DIR}\n")

    # Load SAM predictor ONCE and reuse across all videos
    print("Loading SAM model (one-time) …")
    first_path = video_paths[0]
    segmenter  = SamVideoSegmenter(first_path, model_size=model)
    print("SAM model ready.\n")

    failed = []
    for vp in video_paths:
        try:
            process_video(vp, segmenter, args.n_ref, OUT_DIR)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(vp)}: {e}")
            failed.append(vp)
            # Try to recover GPU state
            try:
                segmenter.reset()
            except Exception:
                pass
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Done.  {len(video_paths) - len(failed)}/{len(video_paths)} videos succeeded.")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
