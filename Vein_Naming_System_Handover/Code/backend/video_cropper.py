"""
Frame-by-frame video cropper with adaptive ROI re-detection.

Writes to a temporary .avi file first (OpenCV VideoWriter is most reliable
with XVID -> .avi), then remuxes to .mp4 using ffmpeg if available, otherwise
re-encodes with mp4v directly.  Either way the final output is .mp4.
"""

import os
import subprocess
import tempfile
from typing import Callable, Optional

import cv2
import numpy as np

import cv_ensemble

# Re-run the (fast, local, no-LLM) CV ensemble this often to catch the scan-content
# boundary shifting mid-video. Confirmed on real reference footage: the true content
# region can change meaningfully (435x600 -> 554x598) within an 11-second window of the
# SAME recording (same preset, same machine) -- a single ROI computed once from a
# handful of sampled frames and applied statically to the whole video is not always
# valid, it's a real, observed failure mode, not a hypothetical edge case.
REDETECT_INTERVAL_SEC = 2.0
# IoU below which a fresh re-detection is treated as a genuine boundary change (adopted)
# rather than ordinary per-frame ensemble jitter (ignored, keeps the current window).
# Set high (not the conventional ~0.5 "same object" bar) because the real drift this
# exists to catch is itself fairly subtle in IoU terms: confirmed on real footage that a
# genuine, meaningful boundary shift (435x600 -> 554x598, ~27% width difference — one
# edge holding steady while the other moves) still scores IoU~0.78 against the original
# box, since the boxes still overlap heavily. A lower threshold would never trigger on
# real drift like this. Wildly wrong readings (e.g. a transient UI overlay) are caught
# separately by the size-sanity floor above, not by this threshold, so raising this does
# not reopen that failure mode.
REDETECT_IOU_THRESHOLD = 0.85


# -- Drift detection ----------------------------------------------------------

def _has_scan_content(crop: np.ndarray, min_var: float = 30.0) -> bool:
    """
    Rough check: does this crop look like live ultrasound content?
    A frozen / blank / solid-colour frame has very low grayscale variance.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.var()) >= min_var


# -- ffmpeg helper -------------------------------------------------------------

def _get_ffmpeg_exe() -> str | None:
    """Return path to an ffmpeg executable, preferring the system one."""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffmpeg_remux(src: str, dst: str) -> bool:
    """
    Try to remux/re-encode src -> dst (H.264 mp4) using ffmpeg.
    Returns True on success, False if ffmpeg is not available.
    """
    exe = _get_ffmpeg_exe()
    if exe is None:
        return False
    try:
        subprocess.run(
            [exe, "-y", "-i", src,
             "-c:v", "libx264", "-crf", "23",
             "-movflags", "+faststart",
             "-an", dst],
            check=True,
            capture_output=True,
            timeout=600,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# -- Main crop function -------------------------------------------------------

def crop_video(
    input_path: str,
    output_path: str,
    roi: tuple[int, int, int, int],
    drift_check_interval: int = 150,
    on_progress: Optional[Callable[[float], None]] = None,
    dynamic: bool = True,
) -> str:
    """
    Crop every frame of input_path to roi=(x1,y1,x2,y2) (used as the starting/seed
    window and as the OUTPUT video's fixed frame size).
    Writes to output_path (.mp4).
    Calls on_progress(0.0-1.0) periodically if provided.
    Returns output_path.

    If dynamic=True (default): every REDETECT_INTERVAL_SEC, re-runs the CV ensemble on
    the current frame and adopts its box as the new active crop window if it disagrees
    enough with the current one (IoU < REDETECT_IOU_THRESHOLD) to indicate a real content
    boundary shift rather than ordinary per-frame jitter. The active window's *position
    and size* can change frame to frame, but each crop is resized to the fixed
    (crop_w, crop_h) before writing, so the output video itself keeps one resolution
    throughout — required for a valid video file, and harmless downstream since
    BioMedParse's own inference resizes its input again anyway.
    """
    x1, y1, x2, y2 = roi
    crop_w = x2 - x1
    crop_h = y2 - y1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open input video: {input_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write to a temp .avi first; remux to mp4 afterwards
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".avi")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        # XVID not available -- fall back to mp4v directly to output
        os.remove(tmp_path)
        tmp_path = output_path
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (crop_w, crop_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open VideoWriter with XVID or mp4v codec.")
        use_remux = False
    else:
        use_remux = True

    frame_idx      = 0
    drift_warnings = 0
    consec_bad     = 0
    active_roi     = (x1, y1, x2, y2)
    redetect_every_n = max(1, int(REDETECT_INTERVAL_SEC * fps))
    # Rolling buffer of recently-read frames, fed to detect_roi_cv (which internally
    # takes a temporal median across whatever frames it's given) for each re-detection
    # check — a SINGLE frame's reading is measurably noisy (confirmed on real footage: a
    # one-off degenerate 82px-tall box vs. the ~600px norm), the same reason the initial
    # seed detection already uses 5 frames rather than 1. Using recently-read frames
    # (rather than seeking ahead) needs no extra video I/O.
    RECENT_BUFFER_SIZE = 5
    recent_frames: list = []
    # Sanity floor against the SEED box (the one robustly detected via agent+CV / cache,
    # not any later re-detection): confirmed on real footage that a brief on-screen
    # transient (a UI overlay/transition, not a real content-boundary shift) can produce
    # a consistently anomalous reading across the entire recent-frame buffer — e.g. width
    # stayed plausible (518px) but height collapsed to 82px vs. the ~600px norm, an order
    # of magnitude beyond anything seen in genuine drift (435x600 -> 554x598). Rejecting
    # candidates far smaller than the seed on either axis catches this without needing to
    # guess what the transient actually was.
    seed_w, seed_h = crop_w, crop_h
    MIN_DIM_FRAC_OF_SEED = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        recent_frames.append(frame)
        if len(recent_frames) > RECENT_BUFFER_SIZE:
            recent_frames.pop(0)

        if dynamic and frame_idx > 0 and frame_idx % redetect_every_n == 0:
            fresh = cv_ensemble.detect_roi_cv(recent_frames)
            if fresh:
                fresh_w, fresh_h = fresh[2] - fresh[0], fresh[3] - fresh[1]
                plausible = (fresh_w >= MIN_DIM_FRAC_OF_SEED * seed_w
                             and fresh_h >= MIN_DIM_FRAC_OF_SEED * seed_h)
                agreement = cv_ensemble.iou(active_roi, fresh)
                if not plausible:
                    print(f"[Cropper] Ignoring implausible re-detection at frame {frame_idx}: "
                          f"{fresh} ({fresh_w}x{fresh_h}, seed was {seed_w}x{seed_h})")
                elif agreement < REDETECT_IOU_THRESHOLD:
                    print(f"[Cropper] ROI shift at frame {frame_idx}: {active_roi} -> "
                          f"{fresh} (IoU={agreement:.2f})")
                    active_roi = fresh

        ax1, ay1, ax2, ay2 = active_roi
        fh, fw = frame.shape[:2]
        fx1 = max(0, ax1); fy1 = max(0, ay1)
        fx2 = min(fw, ax2); fy2 = min(fh, ay2)
        crop = frame[fy1:fy2, fx1:fx2]

        if crop.size == 0:
            crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        elif crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            # Active window's size can differ from the output's fixed (crop_w, crop_h)
            # once it's been re-detected — resize to match rather than pad, so a wider
            # re-detected window doesn't just get truncated back down to the old size.
            crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # Drift detection (legacy passive warning — kept alongside the active
        # re-detection above as a cheap independent signal something looks wrong,
        # e.g. a frozen/blank frame that re-detection alone wouldn't flag)
        if frame_idx > 0 and frame_idx % drift_check_interval == 0:
            if not _has_scan_content(crop):
                consec_bad  += 1
                drift_warnings += 1
                if consec_bad >= 3:
                    print(f"[Cropper] Drift detected at frame {frame_idx} - "
                          "possible mode change mid-video.")
            else:
                consec_bad = 0

        writer.write(crop)
        frame_idx += 1

        if on_progress and total > 0 and frame_idx % 30 == 0:
            on_progress(frame_idx / total)

    cap.release()
    writer.release()

    if drift_warnings:
        print(f"[Cropper] {drift_warnings} drift warning(s) total.")

    # Remux temp .avi -> .mp4
    if use_remux:
        if _ffmpeg_remux(tmp_path, output_path):
            os.remove(tmp_path)
        else:
            # ffmpeg not available -- re-encode with mp4v via OpenCV
            print("[Cropper] ffmpeg not found; re-encoding with OpenCV mp4v codec.")
            cap2   = cv2.VideoCapture(tmp_path)
            fw2    = cv2.VideoWriter_fourcc(*"mp4v")
            out2   = cv2.VideoWriter(output_path, fw2, fps, (crop_w, crop_h))
            while True:
                ok, fr = cap2.read()
                if not ok:
                    break
                out2.write(fr)
            cap2.release()
            out2.release()
            os.remove(tmp_path)

    if on_progress:
        on_progress(1.0)

    return output_path
