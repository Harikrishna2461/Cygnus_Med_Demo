"""
Real annotated ultrasound video frame extraction.

Sources (Task_3 annotated videos — yellow fascia lines, green vessel contours):
    Moving   → GSV-THI  (probe scanning along GSV in thigh, N2 dilated)
    sample   → SFJ      (N2 + N3 junction / tributary escape point)
    Knee-Ankle → GSV-CAL (GSV N2 at knee-ankle segment)
    Perf     → SPJ/SSV  (posterior fascia/perforator — best available proxy)
    Long2    → EXCLUDED  (longitudinal view, not useful for cross-section frames)

Per-region concatenation order (excluding longitudinal):
    SFJ      : sample_data
    GSV-THI  : Moving + Perf  (concatenated for longer clip)
    GSV-CAL  : Knee-Ankle-Seg1
    SPJ      : Perf
    SSV      : Perf
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2  # type: ignore

logger = logging.getLogger(__name__)

# ── source video paths ─────────────────────────────────────────────────────────
_T3 = Path(r"C:\Users\Krish\Downloads\Task_3\Data\2 - Annotated videos")

_V = {
    "moving":     str(_T3 / "202207191643_00-Moving.mp4"),     # GSV thigh, probe moving
    "sample":     str(_T3 / "sample_data.mp4"),                # SFJ area, N2+N3
    "knee_ankle": str(_T3 / "Knee-Ankle-Seg1.mp4"),            # GSV calf, knee-ankle
    "perf":       str(_T3 / "202207111318_38-Perf.mp4"),       # fascia/perf, proxy for posterior
    # Long2 excluded — longitudinal view
}

# Region → ordered list of source video keys to concatenate
_REGION_VIDEOS: dict[str, list[str]] = {
    "SFJ":     ["sample"],
    "GSV-THI": ["moving", "perf"],
    "GSV-CAL": ["knee_ankle"],
    "SPJ":     ["perf"],
    "SSV":     ["perf"],
}


def _existing(keys: list[str]) -> list[str]:
    """Return paths for the keys whose files actually exist on disk."""
    paths = []
    for k in keys:
        p = _V.get(k, "")
        if p and os.path.isfile(p):
            paths.append(p)
        else:
            logger.warning("Video not found: %s -> %s", k, p)
    return paths


def available_videos(region: str) -> list[str]:
    """Return existing video paths for a region (in concatenation order)."""
    return _existing(_REGION_VIDEOS.get(region, []))


def has_videos(region: str) -> bool:
    return bool(available_videos(region))


# ── frame extraction ───────────────────────────────────────────────────────────

def extract_frames(region: str, n_frames: int = 6) -> list[bytes]:
    """
    Concatenate all videos for a region in memory and return n_frames JPEG
    frames evenly distributed across the full concatenated duration.
    Returns [] if no videos exist.
    """
    paths = available_videos(region)
    if not paths:
        return []

    all_frames: list = []
    for vpath in paths:
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            logger.warning("Cannot open: %s", vpath)
            continue
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)
        cap.release()

    if not all_frames:
        return []

    total = len(all_frames)
    indices = [int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
    result = []
    for idx in indices:
        frame = all_frames[min(idx, total - 1)]
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        result.append(buf.tobytes())
    return result


def extract_frames_b64(region: str, n_frames: int = 6) -> list[str]:
    """Return frames as base64 strings."""
    import base64
    return [base64.b64encode(f).decode() for f in extract_frames(region, n_frames)]


def extract_frames_at_positions(region: str, positions: list[float]) -> list[str]:
    """
    Extract frames at specific relative positions (0.0 = start, 1.0 = end)
    across the concatenated video. Returns base64 JPEG strings.
    Positions outside [0,1] are clamped.
    """
    import base64
    paths = available_videos(region)
    if not paths:
        return [""] * len(positions)

    all_frames: list = []
    for vpath in paths:
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            continue
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)
        cap.release()

    if not all_frames:
        return [""] * len(positions)

    total = len(all_frames)
    result = []
    for pos in positions:
        pos = max(0.0, min(1.0, pos))
        idx = int(pos * (total - 1))
        frame = all_frames[idx]
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        result.append(base64.b64encode(buf.tobytes()).decode())
    return result


def save_concat(region: str, output_path: str) -> bool:
    """
    Concatenate all videos for a region and save to output_path (.mp4).
    Used to materialise the concatenated clips to disk.
    """
    paths = available_videos(region)
    if not paths:
        logger.error("No videos for region %s", region)
        return False
    try:
        cap0 = cv2.VideoCapture(paths[0])
        fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap0.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for vpath in paths:
            cap = cv2.VideoCapture(vpath)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
            cap.release()

        writer.release()
        logger.info("Saved: %s", output_path)
        return True
    except Exception as exc:
        logger.error("save_concat failed: %s", exc)
        return False


def frames_for_scenario_events(
    region: str,
    n_events: int,
    fallback_leg: str = "right",
    fallback_doppler: bool = False,
    fallback_reflux: bool = False,
) -> list[str]:
    """
    Return n_events base64 frames evenly distributed across the region video.
    Returns [] if no video exists — no synthetic fallback.
    """
    return extract_frames_b64(region, n_events)
