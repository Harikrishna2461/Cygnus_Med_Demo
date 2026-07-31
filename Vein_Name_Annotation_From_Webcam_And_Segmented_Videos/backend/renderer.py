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


def _render(frame_bgr, blobs, fascia, label_fn):
    out = draw_fascia_lines(frame_bgr, fascia)
    for b in blobs:
        color = N_CLASS_COLORS.get(b.n_class, DEFAULT_COLOR)
        cv2.drawContours(out, [b.contour], -1, color, 2)
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
