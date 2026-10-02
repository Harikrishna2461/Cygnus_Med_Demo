#!/usr/bin/env python3
"""
Auto-annotation script for ultrasound vein segmentation.

Analyzes video frames to detect vein-like regions and generates annotation
JSON that the GUI can import. The GUI watches data/auto_annotations.json for
changes and loads it automatically.

Usage:
  python auto_annotate.py --video-name full_scanner_23 [--frames 0 100 200] [--data-dir ./data]

When called with --manual-points, skips image analysis and uses the provided
coordinate JSON directly (useful when Claude specifies exact coordinates).
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np


# ──────────────────────────── image analysis ────────────────────────────────

def detect_vein_contours(frame: np.ndarray):
    """
    Return significant contours sorted by area (largest first).
    Tries dark-region detection (anechoic veins in ultrasound) with Otsu
    thresholding, cleaned up with morphological ops.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Dark (anechoic) regions
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * 0.003  # must be at least 0.3 % of frame area
    significant = sorted(
        [c for c in contours if cv2.contourArea(c) > min_area],
        key=cv2.contourArea,
        reverse=True,
    )
    return significant, (h, w)


def sample_from_mask(mask: np.ndarray, n: int, offset_y: int = 0, offset_x: int = 0) -> list:
    """Return up to n evenly-spaced [x, y] points from a binary mask."""
    pts = np.argwhere(mask > 0)  # rows=y, cols=x
    if len(pts) == 0:
        return []
    step = max(1, len(pts) // n)
    sampled = pts[::step][:n]
    return [[int(p[1]) + offset_x, int(p[0]) + offset_y] for p in sampled]


def draw_contour_mask(contour, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def get_points_for_frame(frame_path: str, n_body: int = 8, n_tail: int = 5) -> tuple:
    """
    Analyse one frame.
    Returns (body_points, tail_points, neg_points) as lists of [x, y].
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Cannot read: {frame_path}")

    h, w = frame.shape[:2]
    contours, (fh, fw) = detect_vein_contours(frame)

    body_points: list = []
    tail_points: list = []

    if len(contours) >= 1:
        body_mask = draw_contour_mask(contours[0], fh, fw)
        body_points = sample_from_mask(body_mask, n_body)

    if len(contours) >= 2:
        tail_mask = draw_contour_mask(contours[1], fh, fw)
        tail_points = sample_from_mask(tail_mask, n_tail)
    elif len(contours) == 1 and body_points:
        # Single contour — split it at centroid to get an approximate tail region
        M = cv2.moments(contours[0])
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            full_mask = draw_contour_mask(contours[0], fh, fw)
            x, y, bw, bh = cv2.boundingRect(contours[0])
            if bh >= bw:  # taller → split horizontally at cy
                h1 = full_mask[:cy, :]
                h2 = full_mask[cy:, :]
                if np.sum(h1) >= np.sum(h2):
                    body_points = sample_from_mask(h1, n_body)
                    tail_points = sample_from_mask(h2, n_tail, offset_y=cy)
                else:
                    body_points = sample_from_mask(h2, n_body, offset_y=cy)
                    tail_points = sample_from_mask(h1, n_tail)
            else:  # wider → split vertically at cx
                h1 = full_mask[:, :cx]
                h2 = full_mask[:, cx:]
                if np.sum(h1) >= np.sum(h2):
                    body_points = sample_from_mask(h1, n_body)
                    tail_points = sample_from_mask(h2, n_tail, offset_x=cx)
                else:
                    body_points = sample_from_mask(h2, n_body, offset_x=cx)
                    tail_points = sample_from_mask(h1, n_tail)

    margin = 15
    neg_points = [
        [margin, margin],
        [w - margin, margin],
        [margin, h - margin],
        [w - margin, h - margin],
    ]

    return body_points, tail_points, neg_points


# ─────────────────────────── JSON building ──────────────────────────────────

def build_annotation_dict(frame_data: dict) -> dict:
    """
    frame_data: { frame_idx: {"body": [[x,y],...], "tail": [[x,y],...], "neg": [[x,y],...]} }
    Returns the annotation dict expected by UltrasoundApp.inject_annotations().
    """
    annotations = {
        "0": {"points": [], "labels": [], "frames": []},  # BODY
        "1": {"points": [], "labels": [], "frames": []},  # TAIL
    }
    annotated_frames = []

    for frame_idx, data in frame_data.items():
        frame_idx = int(frame_idx)
        if frame_idx not in annotated_frames:
            annotated_frames.append(frame_idx)

        for pt in data.get("body", []):
            annotations["0"]["points"].append(pt)
            annotations["0"]["labels"].append(1)   # positive for body
            annotations["0"]["frames"].append(frame_idx)

        for pt in data.get("tail", []):
            annotations["1"]["points"].append(pt)
            annotations["1"]["labels"].append(1)   # positive for tail
            annotations["1"]["frames"].append(frame_idx)

        # Negative points go to both objects (they are background for both)
        for pt in data.get("neg", []):
            annotations["0"]["points"].append(pt)
            annotations["0"]["labels"].append(0)
            annotations["0"]["frames"].append(frame_idx)
            annotations["1"]["points"].append(pt)
            annotations["1"]["labels"].append(0)
            annotations["1"]["frames"].append(frame_idx)

    return {"annotations": annotations, "annotated_frames": annotated_frames}


# ────────────────────────────── entry point ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-annotate ultrasound vein video frames")
    parser.add_argument("--video-name", required=True,
                        help="Video name (without extension), e.g. full_scanner_23")
    parser.add_argument("--frames", nargs="+", type=int, default=[0],
                        help="Frame indices to annotate (default: 0)")
    parser.add_argument("--data-dir", default="./data",
                        help="Base data directory (default: ./data)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <data-dir>/auto_annotations.json)")
    parser.add_argument("--n-body", type=int, default=8,
                        help="Number of positive body points per frame")
    parser.add_argument("--n-tail", type=int, default=5,
                        help="Number of positive tail points per frame")
    parser.add_argument("--manual-points", default=None,
                        help="Path to a JSON file with manually specified points. "
                             "Format: {frame_idx: {body:[[x,y],...], tail:[[x,y],...], neg:[[x,y],...]}}. "
                             "Skips image analysis when provided.")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.data_dir, "auto_annotations.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── manual override mode ──
    if args.manual_points:
        with open(args.manual_points) as f:
            frame_data = {int(k): v for k, v in json.load(f).items()}
        result = build_annotation_dict(frame_data)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[auto_annotate] Manual annotations written → {output_path}")
        return

    # ── image-analysis mode ──
    frames_dir = os.path.join(args.data_dir, "frames", args.video_name)
    if not os.path.isdir(frames_dir):
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        print("Make sure you have loaded the video and SAM in the GUI first.")
        sys.exit(1)

    frame_files = sorted(
        [p for p in os.listdir(frames_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    if not frame_files:
        print(f"[ERROR] No frame images found in {frames_dir}")
        sys.exit(1)

    total_frames = len(frame_files)
    frame_data = {}

    for idx in args.frames:
        if idx >= total_frames:
            print(f"[WARN] Frame {idx} out of range (total: {total_frames}), skipping")
            continue
        frame_path = os.path.join(frames_dir, frame_files[idx])
        print(f"[auto_annotate] Analysing frame {idx}: {frame_path}")
        body_pts, tail_pts, neg_pts = get_points_for_frame(
            frame_path, n_body=args.n_body, n_tail=args.n_tail
        )
        print(f"  body: {len(body_pts)} pts, tail: {len(tail_pts)} pts, neg: {len(neg_pts)} pts")
        frame_data[idx] = {"body": body_pts, "tail": tail_pts, "neg": neg_pts}

    if not frame_data:
        print("[ERROR] No frames were successfully analysed.")
        sys.exit(1)

    result = build_annotation_dict(frame_data)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[auto_annotate] Done → {output_path}")
    print("The GUI will auto-load this file if it is open (file watcher active).")


if __name__ == "__main__":
    main()
