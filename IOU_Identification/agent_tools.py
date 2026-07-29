"""
LangChain tools that convert raw video / image pixel data into structured
numerical descriptions the text-only Qwen LLM can reason about.

Key metric: texture_density (fraction of pixels with local texture > threshold).
  - Scan speckle  → texture_density HIGH (0.35–0.70): noise is EVERYWHERE
  - UI text panel → texture_density LOW  (0.02–0.15): texture only at char edges
  - Pure background → texture_density ~0.0
This cleanly separates scan content from UI chrome without a vision model.
"""

import json
import cv2
import numpy as np
from langchain_core.tools import tool

from frame_sampler import sample_frames, get_video_info
from cv_ensemble import detect_roi_cv


# ── core stats ────────────────────────────────────────────────────────────────

def _compute_stats(crop: np.ndarray) -> dict:
    """Compute brightness, texture_density, and grayscale purity for a crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_br = float(gray.mean())

    # texture_density: fraction of pixels whose |Laplacian| exceeds a threshold.
    # Threshold 6 separates speckle-noisy scan pixels from smooth background pixels.
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    texture_density = float((lap > 6).mean())

    b = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    r = crop[:, :, 2].astype(np.int16)
    dev = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    gs_pct = float((dev < 20).mean() * 100)

    return {
        "mean_brightness":  round(mean_br, 1),
        "texture_density":  round(texture_density, 3),   # 0–1
        "grayscale_pct":    round(gs_pct, 1),
    }


def _classify(mean_br: float, tex_density: float, gs_pct: float) -> str:
    if mean_br < 15:
        return "BACKGROUND"
    if tex_density > 0.25 and gs_pct > 80:
        return "SCAN CONTENT"          # spatially distributed speckle
    if tex_density < 0.15 and gs_pct > 85:
        return "UI PANEL"              # text on dark bg: texture only at edges
    if gs_pct < 70:
        return "COLOURED UI CHROME"
    return "MIXED"


def _avg_stats(stats_list: list[dict]) -> dict:
    return {
        "mean_brightness": round(np.mean([s["mean_brightness"] for s in stats_list]), 1),
        "texture_density": round(np.mean([s["texture_density"] for s in stats_list]), 3),
        "grayscale_pct":   round(np.mean([s["grayscale_pct"]   for s in stats_list]), 1),
    }


def _load_frames(video_path: str, n: int = 3) -> list[np.ndarray]:
    return [f for _, f in sample_frames(video_path, n=n)]


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_video_metadata(video_path: str) -> str:
    """
    Return basic metadata for a video file: width, height, fps, total_frames.
    Always call this first to know the valid coordinate range.
    """
    try:
        return json.dumps(get_video_info(video_path))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_cv_detection(video_path: str) -> str:
    """
    Run the computer-vision ROI ensemble and return a bounding box candidate.
    This is a fast starting estimate — verify and refine it with analyze_region.
    Output: {"cv_roi": [x1, y1, x2, y2]}
    """
    try:
        frames = _load_frames(video_path, n=5)
        roi = detect_roi_cv(frames)
        return json.dumps({"cv_roi": list(roi) if roi else None})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def analyze_region(video_path: str, x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Analyse a rectangular region and return image statistics averaged over
    3 sampled frames.

    Returns:
      mean_brightness  — 0 (black) to 255 (white)
      texture_density  — fraction of pixels with significant local texture (0–1).
                         SCAN CONTENT has density > 0.25 (speckle everywhere).
                         UI text panels have density < 0.15 (edges only).
      grayscale_pct    — % of pixels that are near-grayscale (R≈G≈B)
      classification   — SCAN CONTENT | UI PANEL | BACKGROUND | COLOURED UI CHROME | MIXED

    Use this to:
    - Confirm the CV proposed box is actually scan content
    - Check a narrow strip at a boundary to see which side the scan area is on
    - Compare the right/left/top/bottom edges to find exact cut-points
    """
    try:
        frames = _load_frames(video_path, n=3)
        stats_list = []
        for frame in frames:
            h, w = frame.shape[:2]
            fx1, fy1 = max(0, x1), max(0, y1)
            fx2, fy2 = min(w, x2), min(h, y2)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            stats_list.append(_compute_stats(frame[fy1:fy2, fx1:fx2]))

        if not stats_list:
            return json.dumps({"error": "Region outside frame bounds"})

        avg = _avg_stats(stats_list)
        avg["classification"] = _classify(
            avg["mean_brightness"], avg["texture_density"], avg["grayscale_pct"]
        )
        avg["region"] = [x1, y1, x2, y2]
        return json.dumps(avg)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_spatial_profile(video_path: str) -> str:
    """
    Divide the frame into 20 vertical column-slices and 16 horizontal row-slices.
    For each slice report: avg_brightness, texture_density, classification.

    Use this as a map of the frame to quickly identify:
    - Where scan content is concentrated (high texture_density)
    - Where UI panels / sidebars are (low texture_density, not dark)
    - Where pure background is (very low brightness)

    Also returns suggested_roi computed purely from this profile.
    Use suggested_roi alongside cv_roi to triangulate the true boundary.
    """
    try:
        frames = _load_frames(video_path, n=3)
        h, w = frames[0].shape[:2]

        def slices(n: int, axis_len: int, perp_len: int, axis: str) -> list[dict]:
            seg = max(1, axis_len // n)
            result = []
            for i in range(n):
                s = i * seg
                e = min(axis_len, (i + 1) * seg)
                stats_list = []
                for frame in frames:
                    crop = frame[:, s:e, :] if axis == "col" else frame[s:e, :, :]
                    stats_list.append(_compute_stats(crop))
                avg = _avg_stats(stats_list)
                cls = _classify(avg["mean_brightness"], avg["texture_density"], avg["grayscale_pct"])
                result.append({
                    "range": f"{s}-{e-1}",
                    "brightness": avg["mean_brightness"],
                    "texture_density": avg["texture_density"],
                    "type": cls,
                })
            return result

        col_prof = slices(20, w, h, "col")
        row_prof = slices(16, h, w, "row")

        # Derive suggested roi: largest contiguous SCAN CONTENT or MIXED run
        # MIXED = boundary zone between scan and UI — include it so suggested_roi
        # covers the full scan area without the agent needing to binary-search edges.
        def best_range(prof: list[dict]) -> tuple[int, int] | None:
            best, best_len, start = None, 0, None
            for seg in prof:
                is_content = seg["type"] in ("SCAN CONTENT", "MIXED")
                lo = int(seg["range"].split("-")[0])
                hi = int(seg["range"].split("-")[1]) + 1
                if is_content and start is None:
                    start = lo
                elif not is_content and start is not None:
                    span = lo - start
                    if span > best_len:
                        best_len, best = span, (start, lo - 1)
                    start = None
            if start is not None:
                hi = int(prof[-1]["range"].split("-")[1])
                span = hi - start + 1
                if span > best_len:
                    best = (start, hi)
            return best

        cx = best_range(col_prof)
        ry = best_range(row_prof)
        suggested = [cx[0], ry[0], cx[1], ry[1]] if cx and ry else None

        return json.dumps({
            "frame_size":     f"{w}x{h}",
            "col_profile":    col_prof,
            "row_profile":    row_prof,
            "suggested_roi":  suggested,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def scan_boundary(
    video_path: str,
    axis: str,
    scan_from: int,
    scan_to: int,
    perp_from: int,
    perp_to: int,
    step: int = 15,
) -> str:
    """
    Scan along one axis in fine steps to locate the exact edge of the scan area.

    axis:       'x' — scan horizontally (vary x-coordinate)
                'y' — scan vertically   (vary y-coordinate)
    scan_from, scan_to: coordinate range to scan along the chosen axis.
    perp_from, perp_to: the perpendicular range to average over (a representative band).
    step:       pixel step (default 15).

    Returns a table of (position, brightness, texture_density, classification).
    Use this to pin-point exactly where SCAN CONTENT transitions to UI PANEL or BACKGROUND.
    """
    try:
        frames = _load_frames(video_path, n=2)
        h, w = frames[0].shape[:2]
        rows = []
        for pos in range(scan_from, scan_to, step):
            pos_end = min(pos + step, scan_to)
            stats_list = []
            for frame in frames:
                if axis == "x":
                    crop = frame[perp_from:perp_to, pos:pos_end, :]
                else:
                    crop = frame[pos:pos_end, perp_from:perp_to, :]
                if crop.size == 0:
                    continue
                stats_list.append(_compute_stats(crop))
            if not stats_list:
                continue
            avg = _avg_stats(stats_list)
            rows.append({
                "pos":              pos,
                "brightness":       avg["mean_brightness"],
                "texture_density":  avg["texture_density"],
                "type":             _classify(avg["mean_brightness"], avg["texture_density"], avg["grayscale_pct"]),
            })
        return json.dumps({"axis": axis, "scan": rows})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Exported list for the agent
ALL_TOOLS = [
    get_video_metadata,
    run_cv_detection,
    analyze_region,
    get_spatial_profile,
    scan_boundary,
]
