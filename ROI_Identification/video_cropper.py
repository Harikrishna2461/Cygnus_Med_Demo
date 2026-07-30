"""
Frame-by-frame video cropper with drift detection.

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
) -> str:
    """
    Crop every frame of input_path to roi=(x1,y1,x2,y2).
    Writes to output_path (.mp4).
    Calls on_progress(0.0-1.0) periodically if provided.
    Returns output_path.
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        fx1 = max(0, x1); fy1 = max(0, y1)
        fx2 = min(fw, x2); fy2 = min(fh, y2)
        crop = frame[fy1:fy2, fx1:fx2]

        if crop.size == 0 or crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            # Pad if needed (shouldn't happen but be safe)
            canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            h_c, w_c = crop.shape[:2]
            canvas[:h_c, :w_c] = crop
            crop = canvas

        # Drift detection
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
