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
    The model predicts the full fascia zone; top edge = superficial line,
    bottom edge = deep line — both read directly from the blob boundaries.
    """
    H, W = prob.shape
    LINE_HALF = 6                      # 13px thick per line

    col_max = prob.max(axis=0)
    valid   = col_max > threshold

    if valid.sum() < int(0.40 * W):
        return np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)

    above = prob > threshold
    sup_raw  = np.argmax(above, axis=0).astype(np.float64)
    deep_raw = (H - 1 - np.argmax(above[::-1], axis=0)).astype(np.float64)

    valid_idx   = np.where(valid)[0]
    sup_filled  = np.interp(np.arange(W), valid_idx, sup_raw[valid_idx])
    deep_filled = np.interp(np.arange(W), valid_idx, deep_raw[valid_idx])

    # Edge-pad before convolving so boundary doesn't ramp toward zero
    k   = min(63, max(3, W // 16))
    pad = k // 2
    kernel = np.ones(k) / k
    sup_smooth  = np.convolve(np.pad(sup_filled,  pad, mode='edge'), kernel, mode='valid')[:W]
    deep_smooth = np.convolve(np.pad(deep_filled, pad, mode='edge'), kernel, mode='valid')[:W]

    sup_rows  = np.clip(sup_smooth.astype(int),  0, H - 1)
    deep_rows = np.clip(deep_smooth.astype(int), 0, H - 1)

    sup_mask  = np.zeros((H, W), dtype=np.uint8)
    deep_mask = np.zeros((H, W), dtype=np.uint8)
    cols = valid_idx
    for dr in range(-LINE_HALF, LINE_HALF + 1):
        sup_mask [np.clip(sup_rows[cols]  + dr, 0, H-1), cols] = 255
        deep_mask[np.clip(deep_rows[cols] + dr, 0, H-1), cols] = 255

    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    sup_mask  = cv2.morphologyEx(sup_mask,  cv2.MORPH_CLOSE, h_close)
    deep_mask = cv2.morphologyEx(deep_mask, cv2.MORPH_CLOSE, h_close)

    # Clip to valid column range — prevents MORPH_CLOSE edge bleed
    c0, c1 = int(valid_idx[0]), int(valid_idx[-1]) + 1
    if c0 > 0:
        sup_mask[:, :c0] = 0;  deep_mask[:, :c0] = 0
    if c1 < W:
        sup_mask[:, c1:] = 0;  deep_mask[:, c1:] = 0

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
        band = cv2.bitwise_or(sup_mask, deep_mask)
        if band.max() > 0:
            out[band > 0] = SUPERFICIAL_COLOR
    elif fascia_mask is not None and fascia_mask.max() > 0:
        out[fascia_mask > 0] = SUPERFICIAL_COLOR

    if vein_mask is not None and vein_mask.max() > 0:
        cnts, _ = cv2.findContours(vein_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, VEIN_COLOR, 3)
    return out
