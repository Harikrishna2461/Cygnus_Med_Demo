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

import frame_sampler
import cv_ensemble
import machine_registry
from video_cropper import crop_video


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
            crop_video(input_path, output_path, cached, on_progress=on_progress)
            return {
                "output_path": output_path,
                "roi":         cached,
                "method":      "registry_cache",
                "confidence":  1.0,
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

    return {
        "output_path": output_path,
        "roi":         final_roi,
        "method":      method,
        "confidence":  confidence,
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
