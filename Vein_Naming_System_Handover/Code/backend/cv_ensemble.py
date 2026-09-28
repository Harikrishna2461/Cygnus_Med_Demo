"""
General-purpose CV ensemble for ultrasound scan-area ROI detection.
Works on any machine UI without prior knowledge of layout or brand.
Four signals, all derived from physical properties of B-mode ultrasound:
  1. Grayscale content  — scan area has no colour; UI chrome does
  2. Non-dark region    — dark surround separates scan content from chrome
  3. Texture / speckle  — scan area has characteristic high-frequency texture
  4. Contour border     — many machines draw an explicit border around scan area
"""

import cv2
import numpy as np
from typing import Optional


# ── Signal helpers ────────────────────────────────────────────────────────────

def _grayscale_mask(frame: np.ndarray, thr: int = 20) -> np.ndarray:
    """Pixels where all channels are near-equal → near-grayscale content."""
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    dev = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    return (dev < thr).astype(np.uint8) * 255


def _nondark_mask(frame: np.ndarray) -> np.ndarray:
    """
    Remove background surround pixels.
    Rather than a fixed threshold, we use Otsu's method on a darkened version
    of the frame to find the valley between background and content.
    This adapts to machines where the surround is dark-gray, not pure-black.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Otsu threshold on a gamma-darkened image makes the valley more pronounced
    darkened = (gray.astype(np.float32) ** 0.5 * 16).clip(0, 255).astype(np.uint8)
    thr, mask = cv2.threshold(darkened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _texture_mask(frame: np.ndarray, sigma: float = 2.0, thr: int = 3) -> np.ndarray:
    """High-frequency texture (speckle noise) present in scan content.
    Works even in very dark tissue regions where brightness is near-zero,
    because speckle noise creates local variation even at low intensities."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    diff = np.abs(gray - blur)
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def _largest_component_bbox(
    mask: np.ndarray,
    min_fill: float = 0.05,
    max_fill: float = 0.98,
) -> Optional[tuple[int, int, int, int]]:
    """
    Apply large morphological closure (proportional to image size) to fill
    interior dark gaps within the scan area, then return the bounding box of
    the largest connected component.
    """
    h, w = mask.shape[:2]
    total = h * w

    # Kernel size ~1/25 of shortest dimension — large enough to bridge dark
    # tissue gaps inside the scan area without merging separate UI regions.
    k = max(15, min(w, h) // 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n < 2:
        return None

    best_box = None
    best_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        fill = area / total
        if fill < min_fill or fill > max_fill:
            continue
        if area > best_area:
            best_area = area
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            best_box = (x, y, x + bw, y + bh)

    return best_box


def _contour_bbox(
    frame: np.ndarray,
    min_fill: float = 0.10,
    max_fill: float = 0.97,
) -> Optional[tuple[int, int, int, int]]:
    """
    Canny + contour approach: find the largest enclosed rectangular region
    that is neither too small nor full-frame (which would just be the screen edge).
    """
    h, w = frame.shape[:2]
    total = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 60)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_box = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        fill = area / total
        if fill < min_fill or fill > max_fill:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if area > best_area:
            best_area = area
            best_box = (x, y, x + bw, y + bh)

    return best_box


# ── Signal 5: column/row density profile ─────────────────────────────────────

def _find_dense_range(
    profile: np.ndarray, rel_thr: float = 0.30
) -> tuple[int, int] | tuple[None, None]:
    """
    Largest contiguous index range where profile > rel_thr * max(profile).
    Returns (start, end) inclusive, or (None, None) if not found.
    """
    peak = float(profile.max())
    if peak < 1e-6:
        return None, None
    thr = peak * rel_thr
    above = profile > thr
    best_start = best_end = None
    best_len = 0
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            span = i - start
            if span > best_len:
                best_len = span
                best_start, best_end = start, i - 1
            start = None
    if start is not None:
        span = len(profile) - start
        if span > best_len:
            best_start, best_end = start, len(profile) - 1
    return best_start, best_end


def _profile_bbox(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Column/row density profile signal.
    The scan area has HIGH density of nondark+grayscale content per column/row.
    Thin parameter panels and sidebars have LOWER per-column density.
    Works independently of component connectivity, so depth-ruler bridges
    between scan area and right panel do not corrupt this signal.
    """
    h, w = frame.shape[:2]
    gs = _grayscale_mask(frame).astype(np.float32) / 255.0
    nd = _nondark_mask(frame).astype(np.float32) / 255.0
    content = gs * nd                   # bright grayscale pixels

    col_density = content.mean(axis=0)  # (W,) — density per column
    row_density = content.mean(axis=1)  # (H,) — density per row

    x1, x2 = _find_dense_range(col_density, rel_thr=0.30)
    y1, y2 = _find_dense_range(row_density, rel_thr=0.15)

    if x1 is None or y1 is None:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ── Per-frame detection ───────────────────────────────────────────────────────

def _detect_single_frame(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Run all signals on one frame, return list of bounding box proposals."""
    proposals: list[tuple[int, int, int, int]] = []

    gs = _grayscale_mask(frame)
    nd = _nondark_mask(frame)
    tx = _texture_mask(frame)

    # Signal 1+2: grayscale AND (non-dark OR texture)
    nd_or_tx = cv2.bitwise_or(nd, tx)
    combined_12 = cv2.bitwise_and(gs, nd_or_tx)
    b = _largest_component_bbox(combined_12)
    if b:
        proposals.append(b)

    # Signal 3: texture alone — captures speckle in near-pitch-black regions
    b = _largest_component_bbox(tx)
    if b:
        proposals.append(b)

    # Signal 4: contour border
    b = _contour_bbox(frame)
    if b:
        proposals.append(b)

    # Signal 5: column/row density profile (not affected by bridge artifacts)
    b = _profile_bbox(frame)
    if b:
        proposals.append(b)

    return proposals


# ── Ensemble fusion ───────────────────────────────────────────────────────────

def _median_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        int(np.median([b[0] for b in boxes])),
        int(np.median([b[1] for b in boxes])),
        int(np.median([b[2] for b in boxes])),
        int(np.median([b[3] for b in boxes])),
    )


def detect_roi_cv(
    frames: list[np.ndarray],
    inward_padding: int = 4,
) -> Optional[tuple[int, int, int, int]]:
    """
    Run the four-signal ensemble across all supplied frames and return a
    consensus (x1, y1, x2, y2) bounding box, or None on failure.

    Temporal median across frames rejects transient UI overlays (tooltips,
    measurement cursors) that might pollute a single-frame reading.
    """
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    per_frame_medians: list[tuple[int, int, int, int]] = []

    for frame in frames:
        props = _detect_single_frame(frame)
        if props:
            per_frame_medians.append(_median_box(props))

    if not per_frame_medians:
        return None

    x1, y1, x2, y2 = _median_box(per_frame_medians)

    # Apply small inward padding and clamp to frame bounds
    x1 = max(0, x1 + inward_padding)
    y1 = max(0, y1 + inward_padding)
    x2 = min(w, x2 - inward_padding)
    y2 = min(h, y2 - inward_padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0
