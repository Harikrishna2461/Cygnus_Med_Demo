"""
Standalone test for Foundation Pipeline (PipelineB) detection functions.
No Flask, no TF models needed — only cv2, numpy, scipy.
Runs on the existing frames_preview JPGs and saves annotated output.
"""
import cv2, numpy as np, os
from pathlib import Path
from scipy.ndimage import uniform_filter1d

IMG_SIZE = 256
NEAR_PX  = 20
HERE = Path(__file__).parent

# ── copied helpers from app.py (no TF dependency) ────────────────────────────

def get_scan_x_limits(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, bright = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    k      = max(25, int(h * 0.06))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    col_bright = (opened > 0).sum(axis=0)
    xs = np.where(col_bright > h * 0.05)[0]
    if len(xs) < 10:
        return 0, w
    x_left, x_right = int(xs[0]), int(xs[-1] + 1)
    if (x_right - x_left) < w * 0.4:
        return 0, w
    return x_left, x_right

def _refine_fascia_per_column(bgr, approx_y, search_half=25, scan_xl_m=0, scan_xr_m=IMG_SIZE):
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.resize(gray.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    gray_s = cv2.GaussianBlur(gray_r, (7, 7), 0)
    y_lo = int(np.clip(approx_y - search_half, 0, IMG_SIZE - 1))
    y_hi = int(np.clip(approx_y + search_half + 1, 1, IMG_SIZE))
    half = 8
    x_lo = max(0, scan_xl_m)
    x_hi = min(IMG_SIZE, scan_xr_m)
    top_out = np.full(IMG_SIZE, np.nan, np.float32)
    bot_out = np.full(IMG_SIZE, np.nan, np.float32)
    for x in range(x_lo, x_hi):          # only scan columns
        col_strip = gray_s[y_lo:y_hi, x]
        if float(col_strip.max()) < 15:
            continue
        peak = y_lo + int(np.argmax(col_strip))
        top_out[x] = float(max(0,            peak - half))
        bot_out[x] = float(min(IMG_SIZE - 1, peak + half))
    valid = ~np.isnan(top_out)
    if valid.sum() < 20:
        cy = float(np.clip(approx_y, half, IMG_SIZE - 1 - half))
        top_out[x_lo:x_hi] = cy - half
        bot_out[x_lo:x_hi] = cy + half
        return top_out, bot_out
    # IQR outlier rejection: columns where per-column peak deviates wildly (scan-edge artifacts)
    tops = top_out[valid]
    q25, q75 = float(np.percentile(tops, 25)), float(np.percentile(tops, 75))
    iqr = q75 - q25 if q75 > q25 else 8.0
    lo_fence, hi_fence = q25 - 2.0 * iqr, q75 + 2.0 * iqr
    med = float(np.median(tops))
    tops = np.clip(tops, lo_fence, hi_fence)   # replace outliers with fence value
    top_out[valid] = tops
    bots = bot_out[valid]
    bot_out[valid] = np.clip(bots, lo_fence + 16, hi_fence + 16)

    xs  = np.arange(IMG_SIZE, dtype=np.float32)
    xv  = xs[valid]
    scan_xs = np.arange(x_lo, x_hi, dtype=np.float32)
    top_out[x_lo:x_hi] = uniform_filter1d(np.interp(scan_xs, xv, top_out[valid]), 25).astype(np.float32)
    bot_out[x_lo:x_hi] = uniform_filter1d(np.interp(scan_xs, xv, bot_out[valid]), 25).astype(np.float32)
    return top_out, bot_out

def fascia_from_brightness(bgr, scan_xl_m=0, scan_xr_m=IMG_SIZE):
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.resize(gray.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    x_lo, x_hi = max(0, scan_xl_m), min(IMG_SIZE, scan_xr_m)
    # Average ONLY scan columns — side panels would bias the profile
    row_sm = uniform_filter1d(gray_r[:, x_lo:x_hi].mean(axis=1).astype(np.float64), 15)
    # Skip scan border: find where scan content actually starts, add a margin
    scan_y_top, _ = detect_scan_y_limits(gray_r, scan_xl_m, scan_xr_m)
    y0 = scan_y_top + 15            # skip the coupling echo line at scan edge
    y0 = max(y0, int(IMG_SIZE * 0.18))  # at least 18% from top
    y1 = int(IMG_SIZE * 0.75)
    if y0 >= y1:
        y0 = int(IMG_SIZE * 0.15)
    grad = np.gradient(row_sm)
    fascia_cy = y0 + int(np.argmax(grad[y0:y1]))
    return _refine_fascia_per_column(bgr, fascia_cy, scan_xl_m=scan_xl_m, scan_xr_m=scan_xr_m), fascia_cy

def nms(dets, iou_thresh=0.3):
    if len(dets) <= 1:
        return dets
    dets_sorted = sorted(dets, key=lambda d: d['area'], reverse=True)
    kept = []
    for d in dets_sorted:
        b1 = d['bbox']
        def iou(b2):
            x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
            ix = max(0, min(x1+w1,x2+w2) - max(x1,x2))
            iy = max(0, min(y1+h1,y2+h2) - max(y1,y2))
            return (ix*iy) / (w1*h1 + w2*h2 - ix*iy + 1e-6)
        if all(iou(k['bbox']) < iou_thresh for k in kept):
            kept.append(d)
    return kept

def detect_scan_y_limits(gray_r, scan_xl_m, scan_xr_m):
    x_lo = max(0, scan_xl_m)
    x_hi = min(IMG_SIZE, scan_xr_m)
    row_mean = gray_r[:, x_lo:x_hi].mean(axis=1).astype(np.float64)
    # Use positive gradient to find the scan top border (dark header → bright scan edge = large jump)
    grad = np.gradient(row_mean)
    top_candidates = np.where(grad[:IMG_SIZE // 2] > 3.0)[0]
    y_top = int(top_candidates[0]) + 3 if len(top_candidates) else int(IMG_SIZE * 0.10)
    # Scan bottom: last row in lower half with enough brightness
    thresh_bot = max(12.0, float(row_mean.max()) * 0.08)
    y_bot = next((y for y in range(IMG_SIZE-1, IMG_SIZE//2, -1) if row_mean[y] > thresh_bot), IMG_SIZE-1) + 1
    return y_top, y_bot

def blob_detect(bgr, min_area=80, scan_xl_m=0, scan_xr_m=IMG_SIZE):
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    # 5×5 blur preserves small dark vein interiors; 9×9 was blurring them above threshold
    blurred = cv2.GaussianBlur(gray_r, (5, 5), 0)

    x_lo, x_hi = max(0, scan_xl_m), min(IMG_SIZE, scan_xr_m)
    # Inset 2px from each scan edge to avoid border-line artifacts
    xi_lo, xi_hi = x_lo + 2, x_hi - 2
    scan_y_top, scan_y_bot = detect_scan_y_limits(gray_r, scan_xl_m, scan_xr_m)
    scan_mean = float(blurred[scan_y_top:scan_y_bot, xi_lo:xi_hi].mean())

    # Lower multiplier (0.45) so dimmer-than-average anechoic veins aren't missed
    thresh_val = max(20, int(scan_mean * 0.45))
    _, binary  = cv2.threshold(blurred.astype(np.uint8), thresh_val, 255, cv2.THRESH_BINARY_INV)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    # Blank everything outside the inset scan area
    binary[:, :xi_lo]      = 0
    binary[:, xi_hi:]      = 0
    binary[:scan_y_top, :] = 0
    binary[scan_y_bot:, :] = 0
    scan_w = max(1, xi_hi - xi_lo)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dets = []
    for i in range(1, n):
        a  = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if a < min_area:
            continue
        # Reject blobs spanning >55% of scan width — deep tissue bands, not veins
        if bw > scan_w * 0.55:
            continue
        if min(bw, bh) == 0 or max(bw, bh) / min(bw, bh) > 3.5:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perim = cv2.arcLength(contours[0], True)
        if perim > 0 and (4 * np.pi * a / perim ** 2) < 0.20:
            continue
        dets.append({
            'bbox':     (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], bw, bh),
            'centroid': (float(centroids[i][0]), float(centroids[i][1])),
            'area':     int(a),
        })
    return nms(dets)

def draw_annotations(bgr, top_y, bot_y, dets, scan_xl, scan_xr, fascia_cy_raw=None):
    vis = bgr.copy()
    h0, w0 = bgr.shape[:2]
    sx, sy = w0 / IMG_SIZE, h0 / IMG_SIZE

    # Scan area border
    cv2.rectangle(vis, (scan_xl, 0), (scan_xr, h0), (50, 50, 80), 1)

    # Fascia band
    xs_valid = [x for x in range(IMG_SIZE) if not np.isnan(top_y[x])]
    if len(xs_valid) > 1:
        pts_top = np.array([[int(x*sx), int(top_y[x]*sy)] for x in xs_valid], np.int32)
        pts_bot = np.array([[int(x*sx), int(bot_y[x]*sy)] for x in xs_valid], np.int32)
        ov = vis.copy()
        cv2.fillPoly(ov, [np.vstack([pts_top, pts_bot[::-1]])], (0, 180, 180))
        cv2.addWeighted(ov, 0.20, vis, 0.80, 0, vis)
        cv2.polylines(vis, [pts_top], False, (0, 255, 255), 2)
        cv2.polylines(vis, [pts_bot], False, (0, 255, 255), 2)

    if fascia_cy_raw is not None:
        flat_y = int(fascia_cy_raw * sy)
        cv2.line(vis, (0, flat_y), (w0, flat_y), (0, 80, 200), 1)

    # Vein detections
    for i, d in enumerate(dets):
        x, y, bw, bh = d['bbox']
        cx, cy = d['centroid']
        # classify by fascia position
        xi = int(np.clip(cx, 0, IMG_SIZE-1))
        if not np.isnan(top_y[xi]):
            if cy < top_y[xi] - NEAR_PX:
                col = (0, 140, 255)   # orange = N3
                lbl = "N3"
            elif cy > bot_y[xi] + NEAR_PX:
                col = (50, 50, 220)   # red = N1
                lbl = "N1"
            else:
                col = (50, 200, 50)   # green = N2
                lbl = "N2"
        else:
            col = (150, 150, 150)
            lbl = "?"
        rx, ry = int(x*sx), int(y*sy)
        rw, rh = int(bw*sx), int(bh*sy)
        cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), col, 2)
        cv2.putText(vis, f"{i} {lbl}", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return vis

# ── Run tests ─────────────────────────────────────────────────────────────────

FRAMES_DIR = HERE / 'frames_preview'
OUT_DIR    = HERE / 'test_output'
OUT_DIR.mkdir(exist_ok=True)

test_frames = [
    'sample_data_raw_f101.jpg',
    'sample_data_raw_f506.jpg',
    'sample_data_raw_f809.jpg',
    '202207111318_38-Perf_raw_f48.jpg',
    '202207111318_38-Perf_raw_f241.jpg',
    '202207191643_00-Moving_raw_f146.jpg',
    '202207191643_00-Moving_raw_f733.jpg',
    'Knee-Ankle-Seg1_raw_f84.jpg',
]

for fname in test_frames:
    p = FRAMES_DIR / fname
    if not p.exists():
        print(f"SKIP {fname}")
        continue

    bgr = cv2.imread(str(p))
    if bgr is None:
        print(f"FAIL load {fname}")
        continue

    scan_xl, scan_xr = get_scan_x_limits(bgr)
    h0, w0 = bgr.shape[:2]
    scan_xl_m = max(0, round(scan_xl * IMG_SIZE / w0))
    scan_xr_m = min(IMG_SIZE, round(scan_xr * IMG_SIZE / w0))

    (top_y, bot_y), fascia_cy = fascia_from_brightness(bgr, scan_xl_m, scan_xr_m)
    dets = blob_detect(bgr, min_area=80, scan_xl_m=scan_xl_m, scan_xr_m=scan_xr_m)

    # Sub-fascia filter (keep only veins at/above fascia + NEAR_PX slack)
    dets_filtered = [d for d in dets
                     if np.isnan(bot_y[int(np.clip(d['centroid'][0], 0, IMG_SIZE-1))])
                     or d['centroid'][1] <= bot_y[int(np.clip(d['centroid'][0], 0, IMG_SIZE-1))] + NEAR_PX]

    vis = draw_annotations(bgr, top_y, bot_y, dets_filtered, scan_xl, scan_xr, fascia_cy)

    info = f"scan x:[{scan_xl},{scan_xr}]  fascia_cy(raw)={fascia_cy}  blobs={len(dets)} filt={len(dets_filtered)}"
    cv2.putText(vis, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    out_path = OUT_DIR / fname
    cv2.imwrite(str(out_path), vis)

    h0, w0 = bgr.shape[:2]
    scan_xl_m = max(0, round(scan_xl * IMG_SIZE / w0))
    scan_xr_m = min(IMG_SIZE, round(scan_xr * IMG_SIZE / w0))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.resize(gray.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    row_sm_all = gray_r.mean(axis=1)
    row_sm_scan = gray_r[:, scan_xl_m:scan_xr_m].mean(axis=1)
    scan_y_top = next((y for y in range(IMG_SIZE) if row_sm_scan[y] > 15), 0)
    print(f"\n{fname}:")
    print(f"  frame={w0}x{h0}  scan_x=[{scan_xl},{scan_xr}]  scan_xl_m={scan_xl_m} scan_xr_m={scan_xr_m}")
    print(f"  scan_y_top(256)={scan_y_top}  fascia_cy(raw)={fascia_cy}")
    # For Moving f146: dump ALL connected components (before NMS) to trace vein
    if '00-Moving_raw_f146' in fname:
        gray_r2 = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (IMG_SIZE,IMG_SIZE)).astype(np.float32)
        scan_ylo, scan_yhi = detect_scan_y_limits(gray_r2, scan_xl_m, scan_xr_m)
        blurred2 = cv2.GaussianBlur(gray_r2, (5,5), 0)
        xi_lo2, xi_hi2 = scan_xl_m + 2, scan_xr_m - 2
        smean = float(blurred2[scan_ylo:scan_yhi, xi_lo2:xi_hi2].mean())
        tv = max(20, int(smean*0.45))
        _, bin2 = cv2.threshold(blurred2.astype(np.uint8), tv, 255, cv2.THRESH_BINARY_INV)
        bin2[:, :scan_xl_m]=0; bin2[:, scan_xr_m:]=0; bin2[:scan_ylo,:]=0; bin2[scan_yhi:,:]=0
        n2,_,stats2,cents2 = cv2.connectedComponentsWithStats(bin2, connectivity=8)
        scan_w2 = scan_xr_m - scan_xl_m
        print(f"  [DEBUG f146] scan_y=[{scan_ylo},{scan_yhi}] scan_mean={smean:.1f} thresh={tv}")
        for i in range(1, n2):
            a=stats2[i,cv2.CC_STAT_AREA]; bw2=stats2[i,cv2.CC_STAT_WIDTH]; bh2=stats2[i,cv2.CC_STAT_HEIGHT]
            cx2,cy2=float(cents2[i][0]),float(cents2[i][1])
            wide = bw2 > scan_w2*0.60
            tiny = a < 50
            print(f"    comp cx={cx2:.0f} cy={cy2:.0f} bw={bw2} bh={bh2} area={a} wide={wide} tiny={tiny}")

    for d in dets:
        cx, cy = d['centroid']
        x,y,bw,bh = d['bbox']
        xi = int(np.clip(cx, 0, IMG_SIZE-1))
        sbot = float(bot_y[xi]) if not np.isnan(bot_y[xi]) else float('nan')
        kept = cy <= sbot + NEAR_PX if not np.isnan(sbot) else True
        in_scan = scan_xl_m <= cx <= scan_xr_m
        print(f"  blob cx={cx:.0f} cy={cy:.0f} bbox=({x},{y},{bw},{bh}) area={d['area']} "
              f"in_scan={in_scan} sbot_at_col={sbot:.0f} kept={kept}")

print(f"\nSaved to {OUT_DIR}")
