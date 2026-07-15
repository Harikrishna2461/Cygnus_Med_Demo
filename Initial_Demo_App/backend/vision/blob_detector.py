"""
Blob detection and tracking for ultrasound vein imaging.
Ported from Cygnus_Assesment_Project with adaptations for CMED.

Detects two blobs (A=large green, B=small magenta) using:
- SimpleBlobDetector with shape/size filters
- KLT optical flow tracking across frames
- Temporal smoothing (EMA)
- NCC refinement for drift prevention
"""

import cv2
import numpy as np


# ROI Detection helper (fallback)
def detect_ultrasound_box(frame_bgr):
    """
    Detect the ultrasound image on the right panel.
    Returns conservative box if uncertain.
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (5, 5), 0.8)

    # Look only on the right half
    xR = int(W * 0.45)
    roi = g[:, xR:]

    _, th = cv2.threshold(roi, 18, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), 2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Generous fallback
        x = int(W * 0.48)
        y = int(H * 0.10)
        w = int(W * 0.50)
        h = int(H * 0.80)
        return (x, y, w, h)

    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x += xR

    if w < W * 0.20 or h < H * 0.20:
        x = int(W * 0.48)
        y = int(H * 0.10)
        w = int(W * 0.50)
        h = int(H * 0.80)

    # Modest padding
    padL = int(0.02 * W)
    padR = int(0.03 * W)
    padT = int(0.02 * H)
    padB = int(0.04 * H)
    x = max(0, min(x - padL, W - 1))
    y = max(0, min(y - padT, H - 1))
    w = max(20, min(w + padL + padR, W - x))
    h = max(20, min(h + padT + padB, H - y))
    return (int(x), int(y), int(w), int(h))


# ========= Tunables (practical, minimal) =========
CLAHE_CLIP = 2.0

# Content window inside the right-plane ROI
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 0.05, 0.12, 0.05, 0.05

# Extra dynamic guard near the right border (fraction of content width)
R_EDGE_GUARD = 0.18

# Blob shape/size
BLOB_MIN_THRESH, BLOB_MAX_THRESH, BLOB_STEPS = 5, 220, 10
BLOB_MIN_AREA_FRAC, BLOB_MAX_AREA_FRAC = 0.00012, 0.08
BLOB_MIN_CIRCULARITY, BLOB_MIN_INERTIA, BLOB_MIN_CONVEXITY = 0.60, 0.40, 0.60

# Pairing rules (A=big green, B=small magenta)
B_RATIO_MIN, B_RATIO_MAX = 0.20, 0.80
B_ALIGN_FRAC = 0.20

# Boxes
BOX_SCALE = 1.10
A_MAX_SIDE, B_MAX_SIDE = 170, 130

# KLT
KLT_WIN, KLT_LEVELS = (19, 19), 3
KLT_TERM = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02)
KLT_MAX_PTS, REDETECT_A, REDETECT_B = 60, 8, 6

# Temporal smoothing / sanity
EMA_ALPHA = 0.35
MAX_JUMP_PX_A, MAX_JUMP_PX_B = 26, 22
NCC_SIZE, NCC_MIN = 32, 0.28
EDGE_PENALTY = 0.35

# =================================================


def _clahe(g):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    return cv2.createCLAHE(CLAHE_CLIP, (8, 8)).apply(g)


def _mk_blob(area):
    """Create SimpleBlobDetector with tuned parameters."""
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold, p.maxThreshold = BLOB_MIN_THRESH, BLOB_MAX_THRESH
    p.thresholdStep = max(5, (BLOB_MAX_THRESH - BLOB_MIN_THRESH) // BLOB_STEPS)
    p.filterByColor = False
    p.filterByArea = True
    p.minArea = max(10.0, area * BLOB_MIN_AREA_FRAC)
    p.maxArea = max(p.minArea + 1.0, area * BLOB_MAX_AREA_FRAC)
    p.filterByCircularity, p.minCircularity = True, BLOB_MIN_CIRCULARITY
    p.filterByInertia, p.minInertiaRatio = True, BLOB_MIN_INERTIA
    p.filterByConvexity, p.minConvexity = True, BLOB_MIN_CONVEXITY
    return cv2.SimpleBlobDetector_create(p)


def _bbox_center_r(cx, cy, r, W, H, cap):
    """Create bbox around blob center with radius constraint."""
    side = int(min(cap, max(16, 2 * r * BOX_SCALE)))
    x = np.clip(cx - side // 2, 0, W - 1)
    y = np.clip(cy - side // 2, 0, H - 1)
    side = min(side, W - x, H - y)
    return (int(x), int(y), int(side), int(side))


def _klt_grid(box, want=KLT_MAX_PTS, step=6):
    """Generate grid of points for KLT tracking."""
    x, y, w, h = box
    pts = []
    for yy in range(y + 3, y + h - 3, step):
        for xx in range(x + 3, x + w - 3, step):
            pts.append([xx, yy])
            if len(pts) >= want:
                break
        if len(pts) >= want:
            break
    return np.float32(pts).reshape(-1, 1, 2) if pts else None


def _center_median(pts):
    """Get median center from tracked points."""
    if pts is None or len(pts) == 0:
        return None
    a = pts.reshape(-1, 2)
    m = np.median(a, axis=0)
    return (int(m[0]), int(m[1]))


def _ring_contrast(img, x, y, r):
    """Dark center vs bright ring (0..1)."""
    r = int(max(6, r))
    H, W = img.shape
    if x - r < 0 or y - r < 0 or x + r > W or y + r > H:
        return 0.0
    patch = img[y - r : y + r, x - r : x + r]
    mask_in = np.zeros_like(patch)
    cv2.circle(mask_in, (r, r), max(2, r // 2), 255, -1)
    ring = cv2.dilate(mask_in, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)), 1)
    ring = cv2.subtract(ring, mask_in)
    mean_in = cv2.mean(patch, mask=mask_in)[0]
    mean_ring = cv2.mean(patch, mask=ring)[0]
    return max(0.0, (mean_ring - mean_in) / max(1.0, mean_ring))


def _valid_mask(content):
    """Mask of allowed pixels (remove margins + bright vertical bands)."""
    H, W = content.shape
    mask = np.zeros_like(content, np.uint8)
    xl = int(M_LEFT * W)
    xr = int((1.0 - M_RIGHT) * W)
    yt = int(M_TOP * H)
    yb = int((1.0 - M_BOTTOM) * H)
    cv2.rectangle(mask, (xl, yt), (xr - 1, yb - 1), 255, -1)

    # Brightness-based vertical band removal
    col = np.mean(content[yt:yb, xl:xr], axis=0)
    thr = np.percentile(col, 96)
    bad_cols = np.where(col >= thr)[0]
    if len(bad_cols) > 0:
        for c in bad_cols:
            xx = xl + int(c)
            cv2.line(mask, (xx, yt), (xx, yb - 1), 0, 3)

    # Fixed right guard
    guard = int(R_EDGE_GUARD * W)
    cv2.rectangle(mask, (W - guard, 0), (W - 1, H - 1), 0, -1)
    return mask


def _detect_all_blobs(content, mask):
    """Detect all blobs in content. Returns list of (x, y, r) tuples in CONTENT coords."""
    H, W = content.shape
    det = _mk_blob(float(H * W))
    kps = det.detect(content)

    def ok(k):
        x, y = int(k.pt[0]), int(k.pt[1])
        return (
            0 <= x < W
            and 0 <= y < H
            and mask[y, x] > 0
            and 6 <= x < W - 6
            and 6 <= y < H - 6
        )

    kps = [k for k in kps if ok(k)]
    if not kps:
        return []
    
    # Sort by size (largest first)
    kps.sort(key=lambda k: k.size, reverse=True)
    
    # Return all blobs as (x, y, radius)
    blobs = []
    for k in kps:
        x, y = int(k.pt[0]), int(k.pt[1])
        r = int(max(5, 0.5 * k.size))
        blobs.append((x, y, r))
    
    return blobs


def _ncc_refine(content, center, tpl):
    """Refine around center using NCC; return (cx,cy,score)."""
    if center is None or tpl is None:
        return None, 0.0
    H, W = content.shape
    cx, cy = center
    r = NCC_SIZE // 2 + 16
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(W, cx + r)
    y1 = min(H, cy + r)
    search = content[y0:y1, x0:x1]
    if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    minv, maxv, minl, maxl = cv2.minMaxLoc(res)
    px, py = maxl
    rcx, rcy = x0 + px + tpl.shape[1] // 2, y0 + py + tpl.shape[0] // 2
    return (int(rcx), int(rcy)), float(maxv)


def _speed_limit(prev, new, max_jump):
    """Clamp movement to max_jump pixels."""
    if prev is None:
        return new
    dx = np.clip(new[0] - prev[0], -max_jump, max_jump)
    dy = np.clip(new[1] - prev[1], -max_jump, max_jump)
    return (int(prev[0] + dx), int(prev[1] + dy))


class _BlobState:
    """State for tracking a single blob."""
    def __init__(self, blob_id):
        self.blob_id = blob_id  # Integer ID
        self.center = None
        self.radius = 0
        self.bbox = None
        self.pts = None  # KLT points
        self.has_track = False
        self.ema = None  # Exponential moving average
        self.tpl = None  # NCC template
        self.frames_since_update = 0
        self.confidence = 0


class _State:
    """Internal state tracker for multi-blob detection."""

    def __init__(self):
        self.prev_content = None
        self.prev_xywh = None
        self.blobs = {}  # {blob_id: _BlobState}
        self.next_blob_id = 0  # Counter for new blobs


class BlobDetector:
    """Detect and track two blobs in ultrasound frames."""

    def __init__(self):
        self.s = _State()

    def reset(self):
        """Reset tracking state."""
        self.s = _State()

    def process_frame(self, frame_bgr):
        """
        Process a single frame for multi-blob detection.
        
        Returns:
            tuple: (annotated_frame, metadata_dict)
        """
        s = self.s

        # 1) Locate ROI & prepare content
        try:
            x0, y0, rw, rh = detect_ultrasound_box(frame_bgr)
        except:
            H, W = frame_bgr.shape[:2]
            x0, y0, rw, rh = int(W * 0.48), int(H * 0.10), int(W * 0.50), int(H * 0.80)

        gray = _clahe(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

        xl = int(M_LEFT * rw)
        xr = int((1.0 - M_RIGHT) * rw)
        yt = int(M_TOP * rh)
        yb = int((1.0 - M_BOTTOM) * rh)
        cx0, cy0 = x0 + xl, y0 + yt
        cw, ch = max(20, xr - xl), max(20, yb - yt)
        content = gray[cy0 : cy0 + ch, cx0 : cx0 + cw]
        H, W = content.shape
        mask = _valid_mask(content)

        def to_full(pt):
            return (int(pt[0] + cx0), int(pt[1] + cy0))

        same = (s.prev_xywh == (cx0, cy0, cw, ch)) and (s.prev_content is not None)

        # 2) KLT tracking for existing blobs
        for blob_id, blob_state in list(s.blobs.items()):
            if not same or not blob_state.has_track or blob_state.pts is None or len(blob_state.pts) == 0:
                blob_state.has_track = False
                blob_state.frames_since_update += 1
                continue
            
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                s.prev_content, content, blob_state.pts, None,
                winSize=KLT_WIN, maxLevel=KLT_LEVELS, criteria=KLT_TERM
            )
            good = (st.reshape(-1) == 1) if st is not None else np.zeros(0, bool)
            arr = (
                nxt.reshape(-1, 2)[good]
                if (nxt is not None and good.size)
                else np.zeros((0, 2))
            )
            
            if arr.size:
                keep = [
                    p
                    for p in arr
                    if 0 <= int(p[0]) < W
                    and 0 <= int(p[1]) < H
                    and mask[int(p[1]), int(p[0])] > 0
                ]
                arr = np.array(keep, dtype=np.float32) if keep else np.zeros((0, 2), np.float32)
            
            if len(arr) >= REDETECT_A:
                blob_state.pts = arr.reshape(-1, 1, 2)
                c = _center_median(blob_state.pts)
                if c is not None:
                    c_full = to_full(c)
                    c_full = _speed_limit(
                        tuple(blob_state.ema.astype(int)) if blob_state.ema is not None else c_full,
                        c_full,
                        MAX_JUMP_PX_A,
                    )
                    if blob_state.ema is None:
                        blob_state.ema = np.array(c_full, np.float32)
                    blob_state.ema = (1 - EMA_ALPHA) * blob_state.ema + EMA_ALPHA * np.array(c_full, np.float32)
                    blob_state.center = (int(blob_state.ema[0]), int(blob_state.ema[1]))
                    blob_state.has_track = True
                    blob_state.frames_since_update = 0
                else:
                    blob_state.has_track = False
                    blob_state.frames_since_update += 1
            else:
                blob_state.has_track = False
                blob_state.frames_since_update += 1
        
        # Remove stale blobs (not tracked for 30 frames)
        for blob_id in list(s.blobs.keys()):
            if s.blobs[blob_id].frames_since_update > 30:
                del s.blobs[blob_id]

        # 3) Detect all blobs
        detected_blobs = _detect_all_blobs(content, mask)

        # 4) Assign detected blobs to existing tracks or create new ones
        if detected_blobs:
            for det_x, det_y, det_r in detected_blobs:
                det_full = to_full((det_x, det_y))
                best_blob_id = None
                best_distance = float('inf')
                
                # Find closest existing blob track
                for blob_id, blob_state in s.blobs.items():
                    if blob_state.center is None:
                        continue
                    dist = np.sqrt((det_full[0] - blob_state.center[0])**2 + 
                                 (det_full[1] - blob_state.center[1])**2)
                    if dist < best_distance and dist < 50:
                        best_distance = dist
                        best_blob_id = blob_id
                
                if best_blob_id is not None:
                    blob_state = s.blobs[best_blob_id]
                    blob_state.center = det_full
                    blob_state.radius = det_r
                    blob_state.frames_since_update = 0
                    blob_state.has_track = True
                    b = _bbox_center_r(det_x, det_y, det_r, W, H, cap=A_MAX_SIDE)
                    blob_state.bbox = (b[0] + cx0, b[1] + cy0, b[2], b[3])
                    blob_state.pts = _klt_grid(b)
                    
                    xa, ya = blob_state.center
                    gx0, gy0 = max(0, xa - NCC_SIZE // 2), max(0, ya - NCC_SIZE // 2)
                    gx1, gy1 = min(gray.shape[1], xa + NCC_SIZE // 2), min(
                        gray.shape[0], ya + NCC_SIZE // 2
                    )
                    blob_state.tpl = gray[gy0:gy1, gx0:gx1].copy()
                else:
                    # Create new blob track
                    blob_id = s.next_blob_id
                    s.next_blob_id += 1
                    blob_state = _BlobState(blob_id)
                    blob_state.center = det_full
                    blob_state.radius = det_r
                    b = _bbox_center_r(det_x, det_y, det_r, W, H, cap=A_MAX_SIDE)
                    blob_state.bbox = (b[0] + cx0, b[1] + cy0, b[2], b[3])
                    blob_state.pts = _klt_grid(b)
                    blob_state.has_track = blob_state.pts is not None and len(blob_state.pts) >= REDETECT_A
                    blob_state.ema = np.array(blob_state.center, np.float32)
                    
                    xa, ya = blob_state.center
                    gx0, gy0 = max(0, xa - NCC_SIZE // 2), max(0, ya - NCC_SIZE // 2)
                    gx1, gy1 = min(gray.shape[1], xa + NCC_SIZE // 2), min(
                        gray.shape[0], ya + NCC_SIZE // 2
                    )
                    blob_state.tpl = gray[gy0:gy1, gx0:gx1].copy()
                    s.blobs[blob_id] = blob_state

        # 5) NCC refinement for all tracked blobs
        for blob_id, blob_state in s.blobs.items():
            if blob_state.center is None or blob_state.tpl is None:
                continue
            cx, cy = blob_state.center[0] - cx0, blob_state.center[1] - cy0
            cand, score = _ncc_refine(content, (cx, cy), blob_state.tpl)
            if cand is not None and score >= NCC_MIN:
                c_full = to_full(cand)
                c_full = _speed_limit(
                    tuple(blob_state.ema.astype(int)) if blob_state.ema is not None else c_full,
                    c_full,
                    MAX_JUMP_PX_A,
                )
                blob_state.center = c_full

        # Save for next KLT
        s.prev_content = content.copy()
        s.prev_xywh = (cx0, cy0, cw, ch)

        # 6) Draw results
        out = frame_bgr.copy()
        targets = []
        
        colors = [
            (0, 255, 0),      # Green
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (0, 165, 255),    # Orange
            (255, 0, 0),      # Blue
            (0, 255, 255),    # Yellow
        ]

        for idx, (blob_id, blob_state) in enumerate(sorted(s.blobs.items())):
            if blob_state.center is None or blob_state.radius <= 0:
                continue

            bx, by = blob_state.center
            color = colors[idx % len(colors)]
            center_color = ((0, 255, 255) if color == (0, 255, 0) else (255, 255, 0))
            
            if blob_state.bbox:
                bbox = blob_state.bbox
                cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
            
            cv2.circle(out, (bx, by), 5, center_color, 2)
            
            conf = int(70 + 30 * min(1.0, (len(blob_state.pts) if blob_state.pts is not None else 0) / KLT_MAX_PTS))
            
            cv2.putText(out, f"Blob {blob_id}", (bx - 20, by - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(out, f"{conf}%", (bx - 10, by + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            targets.append({
                "id": blob_id,
                "bbox": blob_state.bbox,
                "center": (bx, by),
                "conf": conf,
                "mode": "track" if blob_state.has_track else "detect",
            })

        return out, {
            "targets": targets,
            "confidence": int(np.mean([t["conf"] for t in targets])) if targets else 0,
            "blob_count": len(targets),
            "successful": len(targets) > 0,
        }
