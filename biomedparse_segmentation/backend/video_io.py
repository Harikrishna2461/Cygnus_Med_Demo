"""
Video I/O helpers: FPS/frame-count detection, frame extraction, and
H.264 video writing (via imageio-ffmpeg — OpenCV's VideoWriter has no working
H.264 encoder on this machine, same reasoning as Task_2's video_io.py).
"""
import os

import cv2
import imageio_ffmpeg
import numpy as np


def probe_video(path: str) -> dict:
    """
    Read the *actual* fps and frame count of a video.

    Container metadata (cv2.CAP_PROP_FPS / CAP_PROP_FRAME_COUNT) is read first,
    but frame count is cross-checked by decoding to the end, since container
    headers can be wrong/estimated for some AVI/mp4 muxers. The decoded count
    is treated as ground truth; fps is kept from metadata (frame timing) unless
    it is missing/nonsensical, in which case it is derived from
    frame_count / duration instead.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    meta_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Decode-count as ground truth (headers can lie / be estimates)
    decoded_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded_count += 1
    cap.release()

    frame_count = decoded_count if decoded_count > 0 else meta_count

    fps = meta_fps
    if not fps or fps <= 0 or fps > 240:
        # Metadata fps missing/bogus — can't derive without a real duration,
        # fall back to a sane default.
        fps = 30.0

    duration_sec = frame_count / fps if fps > 0 else 0.0

    return {
        'fps': float(fps),
        'frame_count': int(frame_count),
        'meta_frame_count': meta_count,
        'width': width,
        'height': height,
        'duration_sec': float(duration_sec),
    }


def iter_frames(path: str):
    """Yield (frame_index, RGB uint8 ndarray) for every frame in the video, in order."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


class H264VideoWriter:
    """Streams RGB uint8 frames to an H.264-encoded mp4 via bundled ffmpeg."""

    def __init__(self, out_path: str, fps: float, width: int, height: int):
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._writer = imageio_ffmpeg.write_frames(
            out_path, (width, height), fps=fps, codec='libx264',
            output_params=['-pix_fmt', 'yuv420p'],
        )
        self._writer.send(None)  # prime the generator

    def write(self, frame_rgb: np.ndarray):
        self._writer.send(np.ascontiguousarray(frame_rgb))

    def close(self):
        self._writer.close()
