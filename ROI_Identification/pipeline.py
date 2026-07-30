"""
Full ROI-crop pipeline orchestrator.

Stages:
  1. Sample frames from video + get metadata
  2a. Check machine registry (cache hit -> skip to Stage 3)
  2b. Run LangGraph agent (Qwen via Groq) which uses CV + image-analysis tools
      to reason about and determine the correct ROI
  2c. Fall back to CV-only if agent fails
  3.  Crop video frame-by-frame and re-encode
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import frame_sampler
import cv_ensemble
import machine_registry
from video_cropper import crop_video


def _trim_dark_borders(frames: list, roi: tuple) -> tuple:
    """
    Trim black border strips from the ROI.
    Left/right use aggressive thresholds (true black columns + UI markers like 'Q').
    Top/bottom use conservative thresholds (avoid cutting naturally-dark deep scan content).
    """
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

    def _leading(arr, thresh, frac):
        for i, v in enumerate((arr < thresh).mean(axis=-1) if arr.ndim > 1 else arr):
            if v < frac:
                return i
        return 0

    # Left/right: threshold=28, frac=0.65 — catches true black strips and corner markers
    col_dark_lr = (gray < 28).mean(axis=0)
    l = _leading(col_dark_lr, 28, 0.65)
    r = _leading(col_dark_lr[::-1], 28, 0.65)

    # Top/bottom: threshold=18, frac=0.88 — only trim near-black rows, not attenuated scan
    row_dark_tb = (gray < 18).mean(axis=1)
    t = _leading(row_dark_tb, 18, 0.88)
    b = _leading(row_dark_tb[::-1], 18, 0.88)

    nx1, ny1, nx2, ny2 = x1 + l, y1 + t, x2 - r, y2 - b
    if nx2 > nx1 + 20 and ny2 > ny1 + 20:
        return (nx1, ny1, nx2, ny2)
    return roi


def _crop_frames(frames: list, roi: tuple) -> list:
    """Crop a list of numpy frames to the given ROI for VLM view classification."""
    import numpy as np
    x1, y1, x2, y2 = roi
    out = []
    for f in frames:
        h, w = f.shape[:2]
        c = f[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if c.size > 0:
            out.append(c)
    return out


def _classify_view(frames: list, roi: tuple, status_fn) -> str:
    try:
        from vlm_agent import classify_view
        cropped = _crop_frames(frames, roi)
        if not cropped:
            return "UNKNOWN"
        status_fn("Classifying view type (transverse/longitudinal)…")
        view = classify_view(cropped)
        status_fn(f"View type: {view}")
        return view
    except Exception as e:
        status_fn(f"View classification failed: {e}")
        return "UNKNOWN"


def run_pipeline(
    input_path: str,
    output_dir: str,
    use_registry: bool = True,
    use_agent: bool = True,
    on_progress: Optional[Callable[[float], None]] = None,
    audit_log_path: str = "audit_log.json",
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Process one video end-to-end.
    Returns {"output_path", "roi", "method", "confidence"}.
    """
    def _status(msg: str):
        print(f"[Pipeline] {msg}")
        if status_callback:
            status_callback(msg)

    input_path = str(input_path)
    os.makedirs(output_dir, exist_ok=True)

    video_stem  = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{video_stem}_roi.mp4")

    _status(f"Processing: {Path(input_path).name}")

    # Stage 1 ----------------------------------------------------------------
    info = frame_sampler.get_video_info(input_path)
    _status(
        f"Video: {info['width']}x{info['height']}, "
        f"{info['fps']:.1f} fps, {info['total_frames']} frames"
    )

    sampled = frame_sampler.sample_frames(input_path, n=5)
    frames  = [f for _, f in sampled]
    if not frames:
        raise RuntimeError("Could not sample any frames from the video.")
    _status(f"Sampled {len(frames)} representative frames")

    first_frame = frames[0]

    # Stage 2a: registry cache ------------------------------------------------
    if use_registry:
        cached = machine_registry.lookup(first_frame, info["width"], info["height"])
        if cached:
            _status(f"Registry cache hit -> ROI {cached}")
            cached = _trim_dark_borders(frames, cached)
            _status(f"After border trim -> ROI {cached}")
            crop_video(input_path, output_path, cached, on_progress=on_progress)
            view_type = _classify_view(frames, cached, _status)
            return {
                "output_path": output_path,
                "roi":         cached,
                "method":      "registry_cache",
                "confidence":  1.0,
                "view_type":   view_type,
            }

    # Stage 2b: LangGraph agent -----------------------------------------------
    final_roi  = None
    method     = "cv_only"
    confidence = 0.75

    if use_agent:
        _status("Launching ROI detection agent...")
        try:
            from roi_agent import detect_roi
            agent_roi = detect_roi(input_path, status_callback=_status)
            if agent_roi:
                final_roi  = agent_roi
                method     = "agent"
                confidence = 0.92
                _status(f"Agent ROI: {final_roi}")
        except Exception as e:
            _status(f"Agent error ({e}) - falling back to CV-only")

    # Stage 2c: CV fallback ---------------------------------------------------
    if final_roi is None:
        _status("Running CV ensemble fallback...")
        cv_box = cv_ensemble.detect_roi_cv(frames)
        if cv_box:
            final_roi  = cv_box
            method     = "cv_only"
            confidence = 0.75
            _status(f"CV ROI: {final_roi}")
        else:
            raise RuntimeError(
                "Both agent and CV ensemble failed to detect an ROI."
            )

    _status(f"Final ROI: {final_roi} | method: {method} | confidence: {confidence:.2f}")

    # Trim dark border strips and corner UI markers
    final_roi = _trim_dark_borders(frames, final_roi)
    _status(f"After border trim -> ROI {final_roi}")

    # Register for future cache hits (only when confidence is high)
    if use_registry and confidence >= 0.88:
        machine_registry.register(
            first_frame, info["width"], info["height"],
            final_roi, method, confidence, video_stem,
        )

    # Stage 3: crop video -----------------------------------------------------
    _status("Cropping video...")
    crop_video(input_path, output_path, final_roi, on_progress=on_progress)
    _status(f"Done -> {output_path}")

    # Stage 4: VLM view classification ----------------------------------------
    view_type = _classify_view(frames, final_roi, _status)

    return {
        "output_path": output_path,
        "roi":         final_roi,
        "method":      method,
        "confidence":  confidence,
        "view_type":   view_type,
    }


# CLI -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop ultrasound machine UI video to scan-area ROI"
    )
    parser.add_argument("input",          help="Path to raw machine UI video")
    parser.add_argument("--output-dir",   default="outputs", help="Output directory")
    parser.add_argument("--no-agent",     action="store_true", help="CV-only mode (no LLM)")
    parser.add_argument("--no-registry",  action="store_true", help="Skip machine registry cache")
    args = parser.parse_args()

    try:
        result = run_pipeline(
            args.input,
            args.output_dir,
            use_registry=not args.no_registry,
            use_agent=not args.no_agent,
        )
        print(json.dumps(
            {**result, "roi": list(result["roi"])},
            indent=2
        ))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
