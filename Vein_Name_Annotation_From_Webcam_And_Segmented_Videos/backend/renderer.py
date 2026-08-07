"""Draws segmentation results onto frames. Pure rendering — no classification decisions here."""
import cv2
import numpy as np

FASCIA_SUP_COLOR = (0, 230, 230)   # matches Task_4_VLM_Fascia_Vein_Detection's convention
FASCIA_DEEP_COLOR = (0, 160, 230)

N_CLASS_COLORS = {
    "N1": (60, 60, 230),    # deep — reddish
    "N2": (0, 210, 0),      # saphenous trunk — green (matches Task_4's generic vein color)
    "N3": (230, 140, 40),   # superficial tributary — blue-ish
}
DEFAULT_COLOR = (0, 210, 0)


def _put_label(img, text, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_fascia_lines(frame_bgr: np.ndarray, fascia) -> np.ndarray:
    out = frame_bgr.copy()
    if fascia is None:
        return out
    W = out.shape[1]
    for col in range(W):
        sr = fascia.sup_row_at_col[col]
        dr = fascia.deep_row_at_col[col]
        if not np.isnan(sr):
            cv2.circle(out, (col, int(sr)), 3, FASCIA_SUP_COLOR, -1)
        if not np.isnan(dr):
            cv2.circle(out, (col, int(dr)), 3, FASCIA_DEEP_COLOR, -1)
    return out


def _resample_closed_contour(pts: np.ndarray, n_out: int) -> np.ndarray:
    """Evenly resample a closed polygon's vertices along its perimeter (arc length).
    Needed before smoothing because cv2.findContours (CHAIN_APPROX_SIMPLE) spaces raw
    vertices unevenly -- a long straight edge collapses to just its 2 endpoints -- so a
    plain index-based moving average on the raw points would distort the shape instead of
    smoothing it evenly around the boundary."""
    pts = pts.astype(np.float64)
    closed = np.vstack([pts, pts[:1]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        return pts
    targets = np.linspace(0, total, n_out, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg_len) - 1)
    seg_t = np.where(seg_len[idx] > 0, (targets - cum[idx]) / np.where(seg_len[idx] > 0, seg_len[idx], 1), 0.0)
    return closed[idx] + seg_t[:, None] * (closed[idx + 1] - closed[idx])


def _smooth_vein_contour(contour: np.ndarray) -> np.ndarray:
    """Rounds off jagged segmentation noise (sharp corners, short/steep zigzag edges)
    while keeping the contour's actual shape -- deliberately NOT forced to a perfect
    ellipse/circle (an earlier version did that; too aggressive, made every vein look
    identical regardless of its real cross-section -- see user feedback). Resamples to
    evenly-spaced points around the perimeter, then applies a small circular
    (wrap-around) moving-average filter a few times, which blunts sharp corners without
    erasing genuine shape irregularity."""
    if contour is None or len(contour) < 8:
        return contour
    pts = contour.reshape(-1, 2)
    n_out = int(np.clip(len(pts) * 2, 24, 80))
    smoothed = _resample_closed_contour(pts, n_out)
    k = 5  # smoothing window -- small relative to n_out so it rounds corners, not shape
    half = k // 2
    kernel = np.ones(k) / k
    for _ in range(3):
        padded = np.vstack([smoothed[-half:], smoothed, smoothed[:half]])
        xs = np.convolve(padded[:, 0], kernel, mode="valid")
        ys = np.convolve(padded[:, 1], kernel, mode="valid")
        smoothed = np.stack([xs, ys], axis=1)
    return smoothed.astype(np.int32).reshape(-1, 1, 2)


def _render(frame_bgr, blobs, fascia, label_fn):
    out = draw_fascia_lines(frame_bgr, fascia)
    for b in blobs:
        color = N_CLASS_COLORS.get(b.n_class, DEFAULT_COLOR)
        cv2.drawContours(out, [_smooth_vein_contour(b.contour)], -1, color, 2)
        x, y, _, _ = b.bbox
        _put_label(out, label_fn(b), (x, max(15, y - 6)), color=(255, 255, 255))
    return out


def draw_intermediate_frame(frame_bgr: np.ndarray, blobs, fascia) -> np.ndarray:
    """Numbered contours; shows blob_id alone pre-classification, "id:N-class" once Stage 2 has run."""
    return _render(frame_bgr, blobs, fascia,
                    lambda b: str(b.blob_id) if not b.n_class else f"{b.blob_id}:{b.n_class}")


def draw_final_frame(frame_bgr: np.ndarray, blobs, fascia, vein_names: dict) -> np.ndarray:
    """vein_names: {blob_id: "GSV"|...}. Falls back to N-class if a blob wasn't named."""
    def _label(b):
        name = vein_names.get(b.blob_id)
        return f"{b.blob_id}:{name}" if name else f"{b.blob_id}:{b.n_class or '?'}"
    return _render(frame_bgr, blobs, fascia, _label)


_POSITION_LABEL_TEXT = {0: "0 (ABOVE knee)", 1: "1 (AT/BELOW knee)", "uncertain": "uncertain"}
_POSITION_LABEL_COLOR = {0: (60, 60, 230), 1: (0, 210, 0), "uncertain": (150, 150, 150)}


def draw_position_debug_frame(webcam_frame_bgr: np.ndarray, probe_position) -> np.ndarray:
    """Debug/verification video: burns Stage 3a's fast binary probe_position judgment
    (see stage3_webcam_location.read_probe_position) directly onto the webcam frame it
    was based on, at 0 (above knee) / 1 (at-or-below knee) / uncertain — so a human can
    scrub through and visually confirm the upstream binary split (which the narrowed-
    vocabulary leg_level classification in Stage B depends on) is landing correctly,
    independent of what made it into the final annotated ultrasound video."""
    out = webcam_frame_bgr.copy()
    text = _POSITION_LABEL_TEXT.get(probe_position, "uncertain")
    color = _POSITION_LABEL_COLOR.get(probe_position, (150, 150, 150))
    _put_label(out, f"stage A: {text}", (16, 40), color=color, scale=1.1)
    return out
