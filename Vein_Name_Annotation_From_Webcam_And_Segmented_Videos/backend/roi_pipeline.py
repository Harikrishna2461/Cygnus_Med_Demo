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

# IoU above which the agent's box and the CV ensemble's independent box are treated as
# "the same region" (and averaged) rather than a disagreement (agent trusted alone).
AGREEMENT_IOU_THRESHOLD = 0.5


def _trim_dark_borders(frames: list, roi: tuple) -> tuple:
    """
    Adaptive border trim: finds where real tissue speckle texture starts near each
    edge, using a threshold relative to that column/row's own peak texture (not a
    hardcoded absolute value) and requiring a sustained run of high-texture
    columns/rows (not a single noisy pixel) before declaring "content starts here".

    Originally (ROI_Identification/pipeline.py) this looked for the single sharpest
    brightness jump within 60px of each edge. Confirmed on real reference footage
    that this misses a common real case: the scan cone's edge often fades to black
    *gradually* (falloff, not a step), which has no single sharp jump for that
    approach to find regardless of window size — texture density (present throughout
    real tissue, absent in background) catches it correctly instead.
    """
    MAX_SCAN = 180
    MIN_RUN = 5   # consecutive high-texture columns/rows required to count as content

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
    gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)   # keep uint8 — cv2.Laplacian needs it
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

    col_texture = texture.mean(axis=0)
    row_texture = texture.mean(axis=1)

    def _texture_trim(profile):
        strip = profile[:MAX_SCAN]
        if len(strip) < MIN_RUN:
            return 0
        peak = strip.max()
        if peak < 1e-6:
            return 0
        thr = peak * 0.25
        run = 0
        for i, v in enumerate(strip):
            if v >= thr:
                run += 1
                if run >= MIN_RUN:
                    return max(0, i - (MIN_RUN - 1))
            else:
                run = 0
        return 0

    l = _texture_trim(col_texture)
    r = _texture_trim(col_texture[::-1])
    t = _texture_trim(row_texture)
    b = _texture_trim(row_texture[::-1])

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

    # CV ensemble is cheap/deterministic (no LLM call) — compute it unconditionally so it
    # can both cross-validate a registry cache hit and combine with the agent's estimate.
    # Confirmed empirically on real reference footage: a single LangGraph agent call has
    # real run-to-run variance (383px-604px width swings for the *same* physical machine/
    # crop region across different runs) — averaging with CV's independent estimate when
    # they broadly agree corrects for that variance instead of trusting one LLM call.
    cv_roi = cv_ensemble.detect_roi_cv(frames)

    # Stage 2a: registry cache, cross-validated against a fresh CV estimate -------------
    if use_registry:
        cached = machine_registry.lookup(first_frame, info["width"], info["height"])
        if cached:
            agreement = cv_ensemble.iou(cached, cv_roi) if cv_roi else 0.0
            if agreement >= AGREEMENT_IOU_THRESHOLD:
                _status(f"Registry cache hit -> ROI {cached} (CV agrees, IoU={agreement:.2f})")
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
            _status(f"Registry cache hit -> ROI {cached}, but fresh CV estimate disagrees "
                     f"(IoU={agreement:.2f}, cv={cv_roi}) — distrusting cache, re-detecting.")

    # Stage 2b: LangGraph agent, combined with the CV estimate ------------------------
    final_roi  = None
    method     = "cv_only"
    confidence = 0.75
    agent_roi  = None

    if use_agent:
        _status("Launching ROI detection agent...")
        try:
            from roi_agent import detect_roi
            agent_roi = detect_roi(input_path, status_callback=_status)
            if agent_roi:
                _status(f"Agent ROI: {agent_roi}")
        except Exception as e:
            _status(f"Agent error ({e}) - falling back to CV-only")

    if agent_roi and cv_roi:
        agreement = cv_ensemble.iou(agent_roi, cv_roi)
        if agreement >= AGREEMENT_IOU_THRESHOLD:
            final_roi = tuple(round((a + c) / 2) for a, c in zip(agent_roi, cv_roi))
            method = "agent+cv_averaged"
            confidence = 0.95
            _status(f"Agent/CV agree (IoU={agreement:.2f}) -> averaged ROI {final_roi}")
        else:
            # Disagreement: trust the agent (it reasons explicitly about UI-vs-content,
            # CV is purely statistical) but flag it clearly rather than silently picking.
            final_roi = agent_roi
            method = "agent"
            confidence = 0.92
            _status(f"Agent/CV disagree (IoU={agreement:.2f}, cv={cv_roi}) -> trusting agent: {agent_roi}")
    elif agent_roi:
        final_roi = agent_roi
        method = "agent"
        confidence = 0.92

    # Stage 2c: CV-only fallback (agent unavailable/failed) ----------------------------
    if final_roi is None:
        if cv_roi:
            final_roi  = cv_roi
            method     = "cv_only"
            confidence = 0.75
            _status(f"CV ROI: {final_roi}")
        else:
            raise RuntimeError(
                "Both agent and CV ensemble failed to detect an ROI."
            )

    _status(f"Final ROI: {final_roi} | method: {method} | confidence: {confidence:.2f}")

    # Adaptive border trim: detect dark->bright jump at each edge
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
