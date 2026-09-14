"""
Scan-area ROI detection — crops out ultrasound-machine UI chrome (icons,
battery/wifi indicators, black letterboxing) before segmentation.

Without this, BiomedParse sees the *entire device screen*, which looks
nothing like the ROI-cropped frames it was trained/finetuned on, and the
"anechoic dark void" vein prompt fires on ANY dark region — including the
black borders and UI icon corners. Cropping to just the scan area first
(then mapping the resulting mask back to full-frame coordinates) removes
that whole failure mode.

Vendored/adapted from ROI_Identification/pipeline.py + cv_ensemble.py +
frame_sampler.py (see memory note `vein_name_classification_system`, which
hit and fixed this exact problem for a different pipeline). CV-only here
(no LangGraph/Groq agent) — keeps this app dependency-light, fast, and
free of any API key.
"""
import cv2
import numpy as np

import cv_ensemble
import frame_sampler

_MAX_SCAN = 60  # how far from each edge to look for a dark->bright jump


def _trim_dark_borders(frames: list, roi: tuple) -> tuple:
    """Adaptive border trim: shrink roi to the sharpest brightness jump near
    each edge, so a coarse ROI box gets tightened onto the true scan content
    without any hardcoded absolute-brightness threshold."""
    x1, y1, x2, y2 = roi
    cropped = []
    for f in frames:
        h, w = f.shape[:2]
        c = f[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if c.size > 0:
            cropped.append(c)
    if not cropped:
        return roi

    avg = np.mean([f.astype(np.float32) for f in cropped], axis=0).astype(np.uint8)
    gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)

    def _jump_trim(means):
        strip = means[:_MAX_SCAN].astype(float)
        if len(strip) < 2:
            return 0
        diffs = np.diff(strip)
        peak = diffs.max()
        if peak < 15:
            return 0
        return int(np.argmax(diffs)) + 1

    l = _jump_trim(col_means)
    r = _jump_trim(col_means[::-1])
    t = _jump_trim(row_means)
    b = _jump_trim(row_means[::-1])

    nx1, ny1, nx2, ny2 = x1 + l, y1 + t, x2 - r, y2 - b
    if nx2 > nx1 + 20 and ny2 > ny1 + 20:
        return (nx1, ny1, nx2, ny2)
    return roi


def detect_roi(video_path: str, width: int, height: int) -> tuple:
    """
    Detect the ultrasound scan-area ROI for an entire video (one box, reused
    for every frame — the UI chrome layout doesn't move within a recording).

    Returns (x1, y1, x2, y2) in original frame pixel coordinates. Falls back
    to the full frame on any detection failure (best-effort, never crashes
    the job).
    """
    full_frame_roi = (0, 0, width, height)
    try:
        sampled = frame_sampler.sample_frames(video_path, n=5)
        frames = [f for _, f in sampled]
        if not frames:
            return full_frame_roi

        box = cv_ensemble.detect_roi_cv(frames)
        if box is None:
            return full_frame_roi

        box = _trim_dark_borders(frames, box)
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 32 or y2 - y1 < 32:  # sanity floor — too small to be real content
            return full_frame_roi
        return (x1, y1, x2, y2)
    except Exception as e:
        print(f"[roi] detection failed ({e}); using full frame")
        return full_frame_roi
