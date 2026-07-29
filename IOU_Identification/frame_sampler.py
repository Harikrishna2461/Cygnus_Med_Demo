import cv2
import numpy as np


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_frames(video_path: str, n: int = 5) -> list[tuple[int, np.ndarray]]:
    """
    Extract n sharp representative frames from a video.
    Target positions: 5%, 15%, 25%, 50%, 75% of duration.
    Returns list of (frame_index, bgr_frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        raise ValueError(f"Video has no frames: {video_path}")

    fractions = [0.05, 0.15, 0.25, 0.50, 0.75][:n]
    results: list[tuple[int, np.ndarray]] = []

    for frac in fractions:
        target = int(frac * total)
        best_frame: np.ndarray | None = None
        best_sharp = -1.0
        best_idx = target

        for offset in range(-10, 11):
            idx = max(0, min(total - 1, target + offset))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            s = _sharpness(frame)
            if s > best_sharp:
                best_sharp = s
                best_frame = frame.copy()
                best_idx = idx

        if best_frame is not None:
            results.append((best_idx, best_frame))

    cap.release()
    return results


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    info = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info
