import cv2
import numpy as np
import subprocess
import shutil
import os
from vlm_classifier import classify_frame

FFMPEG = shutil.which("ffmpeg")

TEXT_BG_COLOR = {
    "N1": (0,   0, 180),
    "N2": (180, 180,  0),
    "N3": (0,  180, 180),
}
TEXT_COLOR = (255, 255, 255)


def detect_green_circles(frame: np.ndarray) -> list[dict]:
    """Detect pre-drawn green annotation circles via HSV colour thresholding."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))

    # Close the ring outlines into solid blobs so minEnclosingCircle is accurate
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        circles.append({"center_x": int(cx), "center_y": int(cy), "radius": max(8, int(r))})
    return circles


def _match_to_cache(detected: list[dict], cached: list[dict]) -> list[dict]:
    """Assign cached classifications to freshly-detected circles by nearest centre."""
    if not cached or not detected:
        return []
    result, used = [], set()
    for c in detected:
        best, best_d = -1, float("inf")
        for i, cv_ in enumerate(cached):
            if i in used:
                continue
            d = ((c["center_x"] - cv_["center_x"]) ** 2 +
                 (c["center_y"] - cv_["center_y"]) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, i
        if best >= 0 and best_d < 200:
            used.add(best)
            result.append({**c, "classification": cached[best]["classification"]})
    return result


def draw_vein_labels(frame: np.ndarray, veins: list) -> np.ndarray:
    out        = frame.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1

    for v in veins:
        cx  = v["center_x"]
        cy  = v["center_y"]
        r   = v["radius"]
        cls = v["classification"]
        bg_col = TEXT_BG_COLOR.get(cls, (160, 160, 160))

        label = cls
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        pad = 3
        lx  = cx - tw // 2 - pad
        ly  = cy - r - 8
        lx  = max(1, min(lx, out.shape[1] - tw - 2 * pad - 2))
        ly  = max(th + pad + 1, min(ly, out.shape[0] - baseline - 2))

        cv2.rectangle(out,
                      (lx, ly - th - pad),
                      (lx + tw + 2 * pad, ly + baseline),
                      bg_col, -1)
        cv2.putText(out, label, (lx + pad, ly),
                    font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)

    return out


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _open_writer(path: str, fps: float, w: int, h: int):
    candidates = [
        ("avc1", path),
        ("mp4v", path),
        ("XVID", path.replace(".mp4", ".avi")),
    ]
    for fourcc_str, out_path in candidates:
        cc = cv2.VideoWriter_fourcc(*fourcc_str)
        wr = cv2.VideoWriter(out_path, cc, fps, (w, h))
        if wr.isOpened():
            print(f"[video] codec={fourcc_str}  path={out_path}")
            return wr, out_path
    raise RuntimeError(
        "VideoWriter could not open with any available codec "
        "(tried avc1 / mp4v / XVID). "
        "Ensure opencv-python-headless is installed correctly."
    )


def _ffmpeg_h264(src: str, dst: str) -> bool:
    if not FFMPEG:
        return False
    try:
        r = subprocess.run(
            [FFMPEG, "-y", "-i", src,
             "-c:v", "libx264", "-preset", "fast",
             "-crf", "22", "-movflags", "+faststart",
             "-pix_fmt", "yuv420p", dst],
            capture_output=True, timeout=600,
        )
        return r.returncode == 0 and os.path.getsize(dst) > 1_000
    except Exception as e:
        print(f"[ffmpeg] {e}")
        return False


def process_video(
    input_path:  str,
    output_path: str,
    sample_rate: int,
    progress_cb=None,
    frame_cb=None,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if progress_cb:
        progress_cb(1, f"Video: {width}x{height} @ {fps:.1f} fps, {total} frames",
                    {"total_frames": total})

    tmp_path = output_path.replace(".mp4", "_raw.mp4")
    writer, tmp_path = _open_writer(tmp_path, fps, width, height)

    cached_veins: list = []
    total_veins:  int  = 0
    vlm_calls:    int  = 0
    frame_idx:    int  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect green circles every frame (fast OpenCV op — gives exact positions)
        circles = detect_green_circles(frame)

        # Call VLM on sampled frames; it only needs to classify, not locate
        if frame_idx % sample_rate == 0 and circles:
            vlm_calls += 1
            result = classify_frame(encode_jpeg(frame, quality=88), circles)
            if result:
                cached_veins = result
                total_veins += len(result)

        # Match cached classifications to freshly-detected circle positions.
        # Fall back to cached positions directly if detection missed this frame.
        if circles:
            veins = _match_to_cache(circles, cached_veins) or cached_veins
        else:
            veins = cached_veins

        annotated = draw_vein_labels(frame, veins)
        writer.write(annotated)

        if frame_cb is not None and frame_idx % 2 == 0:
            frame_cb(encode_jpeg(annotated, quality=65))
            if progress_cb:
                pct = int(frame_idx / max(total, 1) * 88)
                progress_cb(
                    pct,
                    f"Frame {frame_idx}/{total}  |  VLM calls: {vlm_calls}",
                    {
                        "frames_processed":        frame_idx,
                        "veins_found":             total_veins,
                        "current_classifications": [v["classification"] for v in veins],
                    },
                )

        frame_idx += 1

    cap.release()
    writer.release()

    if progress_cb:
        progress_cb(90, "Encoding output video...")

    if _ffmpeg_h264(tmp_path, output_path):
        os.remove(tmp_path)
        msg = "Done! (H.264)"
    else:
        if tmp_path != output_path:
            if tmp_path.endswith(".avi"):
                output_path_final = output_path.replace(".mp4", ".avi")
                os.rename(tmp_path, output_path_final)
                output_path = output_path_final
            else:
                os.rename(tmp_path, output_path)
        msg = "Done!"

    if progress_cb:
        progress_cb(100, msg,
                    {"frames_processed": frame_idx, "veins_found": total_veins,
                     "output_path": output_path})
