"""Video reading/writing primitives. No pipeline/orchestration logic lives here."""
import cv2
import imageio_ffmpeg

import config


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": (frame_count / fps) if fps else 0.0,
        "width": w,
        "height": h,
    }


def iter_sample_frames(video_path: str, interval_sec: float):
    """Yields (timestamp_sec, frame_bgr) at ~interval_sec spacing. Reads sequentially
    (never seeks) since sequential decode is far more reliable across codecs than
    repeated random seeks."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(fps * interval_sec))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_interval == 0:
                yield idx / fps, frame
            idx += 1
    finally:
        cap.release()


class TimestampFrameReader:
    """Frame-at-timestamp lookup optimized for the monotonically-increasing access
    pattern this pipeline uses (Pass 2 walks the webcam video forward in step with
    Pass 1's ultrasound timestamps). Falls back to a seek if a timestamp ever goes
    backward, but that path is not expected to be exercised."""

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duration_sec = (self.frame_count / self.fps) if self.fps else 0.0
        self._last_frame = None
        self._last_idx = -1

    def get(self, timestamp_sec: float):
        ceiling = max(self.duration_sec - 1.0 / self.fps, 0.0)
        t = max(0.0, min(timestamp_sec + config.WEBCAM_TIME_OFFSET_SEC, ceiling))
        target_idx = int(t * self.fps)
        if target_idx < self._last_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            self._last_idx = target_idx - 1
        while self._last_idx < target_idx:
            ok, frame = self.cap.read()
            if not ok:
                break
            self._last_frame = frame
            self._last_idx += 1
        return self._last_frame

    def release(self):
        self.cap.release()


class OutputVideoWriter:
    """Writes H.264 mp4 via imageio-ffmpeg's bundled static ffmpeg binary. OpenCV's own
    VideoWriter has no working H.264 encoder on this machine (OpenH264 DLL missing) and
    silently falls back to mp4v, which Chrome/Edge frequently refuse to play via
    <video> — confirmed during build. This path needs no system/admin ffmpeg install."""

    def __init__(self, path: str, fps: float, frame_size: tuple):
        self.path = path
        self.frame_size = frame_size  # (W, H)
        self._gen = imageio_ffmpeg.write_frames(
            path, frame_size, fps=fps, codec=config.OUTPUT_CODEC,
            pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            output_params=["-movflags", "+faststart"],
        )
        self._gen.send(None)  # seed the generator
        self.fourcc_used = f"{config.OUTPUT_CODEC}(ffmpeg)"

    def write(self, frame_bgr):
        self._gen.send(frame_bgr.tobytes())

    def release(self):
        self._gen.close()
