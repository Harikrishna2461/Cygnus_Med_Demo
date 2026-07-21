"""
Fascia two-line visualisation helpers.
Import this in the notebook with:
    exec(open(r'C:/Users/Krish/Downloads/Cygnus_Med_Demo/VLM_Fascia_Detection/fascia_helpers.py').read())
"""
import numpy as np
import cv2


def prob_to_fascia_two_lines(prob, threshold=0.15):
    """
    Returns (superficial_mask, deep_mask).

    Probability-weighted centroid per column, smoothed, then offset by
    HALF_GAP in each direction.  No jump filter — both masks are always
    produced when enough columns have signal.
    """
    H, W = prob.shape
    HALF_GAP  = max(40, int(0.05 * H))   # >= 40 px  →  80 px total gap
    LINE_HALF = 3                          # ±3 px  →  7 px thick lines

    col_max = prob.max(axis=0)             # (W,)
    valid   = col_max > threshold          # bool (W,)

    if valid.sum() < int(0.40 * W):
        return np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)

    # Probability-weighted centroid per column (vectorised)
    rows     = np.arange(H, dtype=np.float64)
    col_sums = np.maximum(prob.sum(axis=0), 1e-9)
    ctr_raw  = (prob * rows[:, None]).sum(axis=0) / col_sums   # (W,)

    # Interpolate invalid columns then smooth
    valid_idx  = np.where(valid)[0]
    ctr_filled = np.interp(np.arange(W), valid_idx, ctr_raw[valid_idx])
    k          = min(63, max(3, W // 16))
    ctr_smooth = np.convolve(ctr_filled, np.ones(k) / k, mode='same')

    sup_rows  = np.clip((ctr_smooth - HALF_GAP).astype(int), 0, H - 1)
    deep_rows = np.clip((ctr_smooth + HALF_GAP).astype(int), 0, H - 1)

    sup_mask  = np.zeros((H, W), dtype=np.uint8)
    deep_mask = np.zeros((H, W), dtype=np.uint8)

    cols = valid_idx
    for dr in range(-LINE_HALF, LINE_HALF + 1):
        sup_mask [np.clip(sup_rows[cols]  + dr, 0, H-1), cols] = 255
        deep_mask[np.clip(deep_rows[cols] + dr, 0, H-1), cols] = 255

    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    sup_mask  = cv2.morphologyEx(sup_mask,  cv2.MORPH_CLOSE, h_close)
    deep_mask = cv2.morphologyEx(deep_mask, cv2.MORPH_CLOSE, h_close)

    return sup_mask, deep_mask


def prob_to_fascia_centreline(prob, threshold=0.15):
    """Single deep-band line — used by IoU evaluation cell."""
    _, deep = prob_to_fascia_two_lines(prob, threshold)
    return deep


def annotate(image_rgb, vein_mask, fascia_prob=None, fascia_mask=None):
    """
    fascia_prob : raw prob map   -> draws BOTH fascia bands (cyan + blue)
    fascia_mask : single mask    -> draws one cyan line (legacy)
    """
    out = image_rgb.copy()
    SUPERFICIAL_COLOR = (0, 230, 230)
    DEEP_COLOR        = (0, 160, 230)
    VEIN_COLOR        = (0, 210, 0)

    if fascia_prob is not None:
        sup_mask, deep_mask = prob_to_fascia_two_lines(fascia_prob)
        if sup_mask.max() > 0:
            out[sup_mask > 0] = SUPERFICIAL_COLOR
        if deep_mask.max() > 0:
            out[deep_mask > 0] = DEEP_COLOR
    elif fascia_mask is not None and fascia_mask.max() > 0:
        out[fascia_mask > 0] = SUPERFICIAL_COLOR

    if vein_mask is not None and vein_mask.max() > 0:
        cnts, _ = cv2.findContours(vein_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, VEIN_COLOR, 3)
    return out
