"""Draws segmentation results onto frames. Pure rendering — no classification decisions here."""
import cv2
import numpy as np

import knee_cv

FASCIA_SUP_COLOR = (0, 230, 230)   # matches Task_4_VLM_Fascia_Vein_Detection's convention
FASCIA_DEEP_COLOR = (0, 160, 230)

N_CLASS_COLORS = {
    "N1": (60, 60, 230),    # deep — reddish
    "N2": (0, 210, 0),      # saphenous trunk — green (matches Task_4's generic vein color)
    "N3": (230, 140, 40),   # superficial tributary — blue-ish
}
DEFAULT_COLOR = (0, 210, 0)


def _put_label(img, text, org, color=(255, 255, 255), scale=0.5, thickness=1):
    # Outline thickness scales with the main thickness (roughly 2x + a fixed margin) so
    # a bigger/bolder label keeps a legible black outline instead of the outline
    # staying fixed-width while the text grows past it.
    outline_thickness = thickness * 2 + 1
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


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
    """vein_names: {blob_id: "GSV"|...}.

    STRICT policy per explicit user direction: this video must show ONLY real vein
    names, nothing else -- no placeholder text, no internal state exposed. Two earlier
    versions both violated this in different ways: (1) falling back to the raw N-class
    ("N2", etc.) when unnamed -- Stage 2's depth vocabulary leaking into what's supposed
    to be the finished-names video; (2) a "naming..." placeholder -- an improvement over
    (1) but still not an actual vein name, and confirmed on real output to read as "wtf
    is this" rather than a clear in-progress indicator. The fix now is to not render an
    unnamed blob AT ALL -- no contour, no label -- rather than show any non-name text.
    It simply appears once naming resolves (which is usually within a tick or two --
    stage3_vein_naming's retry/validation loop rejects off-vocabulary and duplicate-
    trunk answers rather than accepting a wrong one, see that module), instead of ever
    displaying a placeholder in the meantime.

    CONFIRMED REAL GAP in that filter: "uncertain" is a legitimate, non-empty value
    name_veins() can return (the model explicitly saying "I don't know" rather than
    guessing) -- `vein_names.get(b.blob_id)` is truthy for it, so it sailed straight
    through the "only real names get drawn" filter and rendered as "1:uncertain" for an
    entire clip. Per explicit direction, "uncertain" is NOT a vein name and must be
    treated exactly like "no name yet" -- excluded here too, not just missing entries."""
    NON_NAMES = {"uncertain", "unknown", ""}
    blobs = [b for b in blobs
             if (vein_names.get(b.blob_id) or "").strip().lower() not in NON_NAMES]
    return _render(frame_bgr, blobs, fascia, lambda b: f"{b.blob_id}:{vein_names[b.blob_id]}")


_POSITION_LABEL_TEXT = {0: "0 (ABOVE knee)", 1: "1 (AT/BELOW knee)", "uncertain": "uncertain"}
_POSITION_LABEL_COLOR = {0: (60, 60, 230), 1: (0, 210, 0), "uncertain": (150, 150, 150)}


def draw_position_debug_frame(webcam_frame_bgr: np.ndarray, probe_position) -> np.ndarray:
    """Debug/verification video: burns Stage 3a's fast binary probe_position judgment
    (see stage3_webcam_location.read_probe_position) directly onto the webcam frame it
    was based on, at 0 (above knee) / 1 (at-or-below knee) / uncertain — so a human can
    scrub through and visually confirm the upstream binary split (which the narrowed-
    vocabulary leg_level classification in Stage B depends on) is landing correctly,
    independent of what made it into the final annotated ultrasound video.

    Also burns the SAME knee-height line knee_cv.py draws for the VLM itself (see
    read_probe_position's "CV-assisted knee line" docstring) — recomputed here directly
    on this frame (classical CV, zero VLM tokens, cheap enough to call once per debug
    frame) so a human reviewing this video sees exactly what the model saw when it made
    the above/below-knee call, not just the call's answer. When knee_cv can't find a
    confident knee on this particular frame (falls back to text-only for the real VLM
    call too, see read_probe_position), no line is drawn — never a guessed one."""
    out = webcam_frame_bgr.copy()
    knee_y, _bbox = knee_cv.find_knee_y(webcam_frame_bgr)
    if knee_y is not None:
        out = knee_cv.draw_knee_line(out, knee_y)
    text = _POSITION_LABEL_TEXT.get(probe_position, "uncertain")
    color = _POSITION_LABEL_COLOR.get(probe_position, (150, 150, 150))
    # Confirmed illegible on real footage: webcam frames are full 1920x1080, and the old
    # scale=1.1/thickness=1 label was sized for a much smaller frame -- proportionally
    # tiny and thin on the real resolution. Bumped scale and thickness substantially, and
    # moved down a little from the original y=40 per explicit request.
    _put_label(out, f"stage A: {text}", (24, 90), color=color, scale=2.2, thickness=4)
    return out
