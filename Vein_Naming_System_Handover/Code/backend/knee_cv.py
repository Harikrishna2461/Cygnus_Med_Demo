"""
Classical CV (no ML/DL) knee-height detector -- ported from the validated scratchpad
prototype (knee_width_profile.py) after real-footage testing confirmed it correctly
locates the kneecap on 3 of 4 test frames (and honestly returns None on the 4th rather
than guessing wrong), a meaningful improvement over every VLM-based attempt at this
same judgment (all of which converged on the same wrong "calf" answer for a
user-confirmed-dodd frame, t=29s in the test clip).

Idea: the knee is anatomically the narrowest point of the leg's silhouette between the
thigh bulge (above) and the calf bulge (below) -- NOT the single narrowest point of the
whole visible leg (that's the ankle, confirmed as the first version's bug: it kept
locking onto the ankle since ankles are narrower than knees). Fixed version requires a
genuine valley -- a dip in width that is followed by a RISE again (the calf bulge)
before the leg ends; the ankle taper never rises again afterward (the leg just ends at
the foot), so this correctly excludes it.

Used as a zero-token landmark: draw the detected knee height as a line directly on the
frame and hand that to the VLM, turning "where is the knee" (a judgment the VLM
consistently got wrong on this patient's footage) into "is this point above or below
the drawn line" (a trivial perception task) -- see stage3_webcam_location.py.
"""
import cv2
import numpy as np


def skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def leg_bbox(mask: np.ndarray, max_width_frac: float = 0.6):
    """Largest connected skin blob in the lower 3/4 of the frame -- assumed to be the
    scanned leg (faces/hands are usually smaller blobs or in the upper portion).

    max_width_frac: reject a candidate wider than this fraction of the frame -- a real
    single leg's silhouette should never span most of the frame width. Confirmed on
    real footage (not assumed): the doctor's bare arm/hand touching the patient's leg
    sometimes merges into one connected skin blob (their skin tones blend at the
    contact point), producing a nonsense combined shape -- one real case had a "leg"
    bbox spanning the ENTIRE 1920px frame width. Feeding the width-profile math a
    merged leg+arm blob produces a garbage knee line with no warning, which is worse
    than admitting failure -- so this rejects degenerate blobs outright and returns
    None (the caller falls back to the original VLM-only reasoning) rather than ship a
    confidently-wrong line."""
    h, w = mask.shape
    region = mask.copy()
    region[: h // 4, :] = 0
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    # try candidates largest-to-smallest, skip any that fail the width sanity check
    order = np.argsort(areas)[::-1]
    for rank in order:
        idx = int(rank) + 1
        x, y, bw, bh, area = stats[idx]
        if bw <= w * max_width_frac:
            return int(x), int(y), int(bw), int(bh)
    return None


def _width_profile(mask: np.ndarray, x0: int, x1: int) -> np.ndarray:
    profile = []
    for row in mask[:, x0:x1]:
        cols = np.where(row > 0)[0]
        if len(cols) == 0:
            profile.append(0)
            continue
        splits = np.where(np.diff(cols) > 1)[0]
        runs = np.split(cols, splits + 1)
        profile.append(max(len(r) for r in runs))
    return np.array(profile)


def find_knee_y(frame_bgr: np.ndarray, y_search_start_frac: float = 0.15,
                 y_search_end_frac: float = 0.85, rise_frac: float = 0.12,
                 min_gap: int = 15):
    """Returns (knee_y_pixel_row, leg_bbox) or (None, leg_bbox) if no confident valley
    was found -- callers MUST treat None as "CV couldn't determine this, fall back",
    never as "no knee" or default to a guessed row."""
    return find_knee_y_from_mask(skin_mask(frame_bgr), y_search_start_frac, y_search_end_frac,
                                 rise_frac, min_gap)


def find_knee_y_from_mask(mask: np.ndarray, y_search_start_frac: float = 0.15,
                          y_search_end_frac: float = 0.85, rise_frac: float = 0.12,
                          min_gap: int = 15):
    """Same detector as find_knee_y, on an already-computed skin mask (so a temporally
    aggregated mask can be fed in -- see stable_knee_series)."""
    bbox = leg_bbox(mask)
    if bbox is None:
        return None, None
    x, y, bw, bh = bbox
    profile = _width_profile(mask, x, x + bw)
    y0, y1 = y + int(bh * y_search_start_frac), y + int(bh * y_search_end_frac)
    if y1 - y0 < 5:
        return None, bbox
    kernel = np.ones(9) / 9
    smoothed = np.convolve(profile, kernel, mode="same")
    search = smoothed[y0:y1]
    if len(search) == 0 or search.max() == 0:
        return None, bbox
    max_w = search.max()

    candidates = []
    for i in range(min_gap, len(search) - min_gap):
        window = search[i - min_gap: i + min_gap]
        if search[i] != window.min():
            continue
        after = search[i:]
        if len(after) == 0:
            continue
        rise = after.max() - search[i]
        if rise >= rise_frac * max_w:
            candidates.append((i, rise))

    if not candidates:
        return None, bbox
    best_i = min(candidates, key=lambda c: c[0])[0]
    return best_i + y0, bbox


def foot_crop(frame_bgr: np.ndarray, bottom_frac: float = 0.28, pad_px: int = 30):
    """Returns a zoomed-in crop of the bottom (foot/ankle) portion of the detected leg
    bbox, upscaled 2.5x, or None if leg_bbox can't be found -- same failure discipline
    as find_knee_y (callers must treat None as "couldn't crop, fall back to full frame",
    never guess a crop region).

    Built to fix a CONFIRMED real vision failure, not a prompt-wording issue: Stage 3's
    surface/leg_side judgment leans partly on whether the visible foot shows TOES (patient
    facing camera) or a bare HEEL/Achilles tendon (patient's back to camera) -- a real,
    checkable detail a human can see clearly on the full frame. But direct testing (both
    isolated single-image tests and the real read_surface_reflux() call, both on real
    footage) confirmed the VLM reliably hallucinates "toes visible" on frames that
    unambiguously show only a heel -- prompt instructions telling it to "look carefully"
    did NOT fix this (tried and confirmed insufficient before this function existed). Most
    likely cause: the model's vision encoder downsizes the full 1920x1080 frame enough
    that the foot -- a small fraction of total frame area -- loses the fine toe/heel
    detail entirely before the model ever "reasons" about it, so no amount of prompt
    engineering about a region it literally cannot resolve will help.

    Fix mirrors find_knee_y/draw_knee_line's already-proven pattern exactly: don't ask the
    model to see something it structurally can't, hand it an image where it CAN. Confirmed
    directly: the same model asked the same toe/heel question got it wrong on the full
    frame and right on this crop, on the identical real frame, in a real side-by-side
    test."""
    mask = skin_mask(frame_bgr)
    bbox = leg_bbox(mask)
    if bbox is None:
        return None
    x, y, bw, bh = bbox
    h, w = frame_bgr.shape[:2]
    y0 = y + int(bh * (1.0 - bottom_frac))
    y1 = min(h, y + bh)
    x0 = max(0, x - pad_px)
    x1 = min(w, x + bw + pad_px)
    if y1 - y0 < 5 or x1 - x0 < 5:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)


KNEE_LINE_COLOR_BGR = (0, 0, 255)  # red -- referenced by name in the VLM prompt


def draw_knee_line(frame_bgr: np.ndarray, knee_y: int) -> np.ndarray:
    """Burns the CV-detected knee height onto the frame as a plain red line -- no text
    label (learned from the reflux_dodd_confirmed.jpg anchoring failure: any answer
    spelled out in words gets read back verbatim instead of reasoned about; a bare
    geometric line carries the same information without that risk)."""
    out = frame_bgr.copy()
    cv2.line(out, (0, knee_y), (out.shape[1], knee_y), KNEE_LINE_COLOR_BGR, 4)
    return out


def occlusion_score(frame_bgr: np.ndarray, min_area_frac: float = 0.05) -> float:
    """Detects a specific real failure mode confirmed on this footage: a bystander's
    (or the camera operator's) bare arm reaching very close to the lens for several
    consecutive seconds, filling a large chunk of frame with motion/focus blur and
    blocking the leg -- distinct from the reflux/knee-boundary problem, a genuine
    camera-obstruction event, not a reasoning failure.

    Tried and rejected first: a Laplacian-sharpness ("is this region blurry") signal --
    confirmed NOT to discriminate on this footage (occluded and clean frames scored
    statistically indistinguishable, likely because ordinary low-texture regions like
    skin/cloth already read as "blurry" on a simple frequency metric at this
    resolution/compression). This version instead looks for a large skin-toned blob
    that is NOT the already-identified leg (see leg_bbox) -- a real near-lens arm is
    much larger than a normal hand/arm at working distance, so this signal (not blur)
    is what actually separates the two cases. Confirmed on real footage: 3 of 5 known
    occlusion events flagged correctly, zero false positives on 11 clean frames tested.

    Known limitation, not silently hidden: when occlusion is severe enough, the
    occluding blob can itself be larger than the true leg and get misidentified BY
    leg_bbox as "the leg" -- this then has nothing left to compare against and returns
    a false negative (missed occlusion). Confirmed on 2 of 5 test cases. Not a fix for
    every occlusion frame, a real reduction in how often one gets through undetected."""
    mask = skin_mask(frame_bgr)
    h, w = mask.shape
    leg = leg_bbox(mask)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = area / (h * w)
        if frac < min_area_frac:
            continue
        if leg is not None:
            lx, ly, lbw, lbh = leg
            ix0, iy0 = max(x, lx), max(y, ly)
            ix1, iy1 = min(x + bw, lx + lbw), min(y + bh, ly + lbh)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            overlap_frac = inter / area if area > 0 else 0
            if overlap_frac > 0.5:
                continue  # this IS the leg blob itself, not a separate occluder
        best = max(best, frac)
    return best

OCCLUSION_SCORE_THRESHOLD = 0.15  # midpoint between the clean-frame ceiling (0.0 on
# every tested clean frame) and the confirmed-occlusion floor (0.301-0.362 on the 3
# frames this signal catches) -- wide margin either side, not a fragile cutoff.

def smooth_knee_series(raw: list, window: int = 21) -> list:
    """Temporal stabiliser for per-tick knee heights (pixel rows, or None where CV found none).

    The patient stands still and the webcam is fixed, so the knee's pixel row should barely
    move over a scan -- yet per-frame skin-mask CV sometimes locks onto the wrong valley or
    fails outright (confirmed on real footage: a line jumping to the ankle-side of the leg for
    a single tick, and None on other ticks). Each output value is the MEDIAN of the valid raw
    readings in a +-window/2 neighbourhood (a median ignores minority outliers); if a
    neighbourhood has fewer than 3 valid readings it falls back to the median of the whole
    video; None only if CV never found a knee anywhere. Pure post-processing of the same
    find_knee_y output -- no new detector."""
    valid_all = [v for v in raw if v is not None]
    if not valid_all:
        return [None] * len(raw)
    global_med = int(round(float(np.median(valid_all))))
    half = window // 2
    out = []
    for i in range(len(raw)):
        nb = [v for v in raw[max(0, i - half): i + half + 1] if v is not None]
        out.append(int(round(float(np.median(nb)))) if len(nb) >= 3 else global_med)
    return out


def stable_knee_series(ts_list: list, frames: list, window_sec: float = 120.0, target_masks: int = 60) -> list:
    """Per-tick knee height from a TEMPORALLY AGGREGATED skin mask (default 120 s windows: a fixed camera + standing patient means longer windows reject the moving arm better; on the reference clip 20-60 s windows still let the arm bias early chunks).

    Why: on real footage the per-frame skin mask is contaminated by whatever moves -- the
    clinician's bare forearm merges with the patient's leg (one blob), curtains/pants pass the
    skin threshold -- so the width-profile valley lands on the wrong row or nowhere. But the
    patient stands still and the webcam is fixed: the pixels that are "skin" in most frames of a
    ~20 s window are the patient's legs, and the moving arm/hands drop out. Running the SAME
    valley detector (find_knee_y_from_mask) on that persistent-skin mask gives a stable, correct
    knee row (confirmed visually on the reference footage: first valley on the static mask sits on
    the kneecap, where the per-frame one was 40-60px high or wildly wrong).

    Windows are chunked (one detection per window_sec/2 chunk, +-window_sec/2 of frames) and the
    per-chunk values are median-filtered (3 chunks) to reject a bad chunk. Returns one int (or
    None if nothing ever worked) per entry of ts_list; caller should fall back to
    smooth_knee_series on per-frame values when the result is all None."""
    n = len(frames)
    if n == 0:
        return []
    step = max(1, n // target_masks)
    idx = [i for i in range(0, n, step) if frames[i] is not None]
    if len(idx) < 3:
        return [None] * n
    masks = {i: (skin_mask(frames[i]) > 0) for i in idx}
    half = window_sec / 2.0
    chunk_len = half
    chunk_val = {}
    for i in range(n):
        cid = int(ts_list[i] // chunk_len)
        if cid in chunk_val:
            continue
        centre = (cid + 0.5) * chunk_len
        members = [j for j in idx if abs(ts_list[j] - centre) <= half]
        if len(members) < 3:
            members = sorted(idx, key=lambda j: abs(ts_list[j] - centre))[:3]
        acc = np.zeros(masks[members[0]].shape, np.float32)
        for j in members:
            acc += masks[j]
        static = (acc / len(members) >= 0.6).astype(np.uint8) * 255
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        chunk_val[cid] = find_knee_y_from_mask(static)[0]
    cids = sorted(chunk_val)
    vals = [chunk_val[c] for c in cids]
    valid = [v for v in vals if v is not None]
    if not valid:
        return [None] * n
    gmed = int(round(float(np.median(valid))))
    filled = [v if v is not None else gmed for v in vals]
    med3 = [int(round(float(np.median(filled[max(0, k - 1): k + 2])))) for k in range(len(filled))]
    lookup = dict(zip(cids, med3))
    return [lookup[int(ts_list[i] // chunk_len)] for i in range(n)]
