"""
Temporal consistency filtering across video frames.

Per-frame BiomedParse segmentation (engine.segment_frame) has no notion of
time — every frame is judged in isolation, so a real vein visible across
many consecutive frames can vanish for one isolated frame, while a single
bad frame's spurious blob (a watermark/probe-indicator dot, a stray
bright/dark speckle) looks no different from a real detection when judged
from that one frame alone.

This module links blobs across frames into tracks by simple nearest-
centroid matching with a frame-gap tolerance (a minimal SORT-style
tracker, no motion model), then:
  - drops tracks shorter than MIN_TRACK_LEN frames — a real vein persists
    across many frames; a lone- or two-frame blob is far more likely to be
    a flicker artifact than a real vein that appeared and vanished in a
    fraction of a second.
  - fills small gaps inside a surviving track (a frame or two where the
    per-frame model missed a real, temporally-confirmed vein) by carrying
    the nearest confirmed detection's shape forward/back.

Deliberately independent of engine.py's per-frame filter constants —
those are left exactly as ported (see engine.py's docstring on why
retuning them made things worse twice already). This adds a second,
orthogonal signal — persistence over time — on top of the unchanged
per-frame output, rather than adjusting what counts as "vein-shaped" in
any single frame.
"""
import cv2
import numpy as np

MAX_FRAME_GAP = 5          # link across up to this many consecutive missing frames
MIN_TRACK_LEN = 4          # drop tracks with fewer confirmed detections than this
CENTER_DIST_FRAC = 0.06    # max centroid jump between linkable frames, as a
                            # fraction of the frame diagonal
AREA_RATIO_MAX = 3.0       # matched blobs' areas must be within this ratio of each other


class Blob:
    __slots__ = ("frame_idx", "centroid", "bbox", "contour", "area")

    def __init__(self, frame_idx, centroid, bbox, contour, area):
        self.frame_idx = frame_idx
        self.centroid = centroid
        self.bbox = bbox
        self.contour = contour
        self.area = area


def extract_blobs(frame_idx: int, mask: np.ndarray) -> list:
    """Connected-component blobs from one frame's binary {0,1} mask."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    blobs = []
    for i in range(1, n):
        mask_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cx, cy = centroids[i]
        bbox = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        blobs.append(Blob(frame_idx, (float(cx), float(cy)), bbox, cnts[0],
                           int(stats[i, cv2.CC_STAT_AREA])))
    return blobs


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def link_tracks(per_frame_blobs: list, frame_w: int, frame_h: int) -> list:
    """
    per_frame_blobs: list indexed by frame_idx of list[Blob] (possibly empty).
    Returns a list of tracks, each a list[Blob] sorted by frame_idx.
    """
    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    max_dist = CENTER_DIST_FRAC * diag

    open_tracks = []
    finished_tracks = []

    for frame_idx, blobs in enumerate(per_frame_blobs):
        used = set()
        for track in open_tracks:
            last = track[-1]
            if frame_idx - last.frame_idx > MAX_FRAME_GAP:
                continue
            best_j, best_d = None, None
            for j, b in enumerate(blobs):
                if j in used:
                    continue
                d = _dist(last.centroid, b.centroid)
                if d > max_dist:
                    continue
                area_ratio = max(b.area, last.area) / max(1, min(b.area, last.area))
                if area_ratio > AREA_RATIO_MAX:
                    continue
                if best_d is None or d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                track.append(blobs[best_j])
                used.add(best_j)

        still_open = []
        for track in open_tracks:
            if frame_idx - track[-1].frame_idx > MAX_FRAME_GAP:
                finished_tracks.append(track)
            else:
                still_open.append(track)
        open_tracks = still_open

        for j, b in enumerate(blobs):
            if j not in used:
                open_tracks.append([b])

    finished_tracks.extend(open_tracks)
    return finished_tracks


def _draw_blob(canvas: np.ndarray, contour: np.ndarray):
    cv2.drawContours(canvas, [contour], -1, 1, thickness=cv2.FILLED)


def build_confirmed_masks(per_frame_blobs: list, frame_w: int, frame_h: int) -> list:
    """
    Returns a list (indexed by frame_idx) of uint8 {0,1} masks containing
    only temporally-confirmed blobs, with small in-track gaps filled by
    carrying the nearest confirmed detection's contour forward/back.
    """
    n_frames = len(per_frame_blobs)
    tracks = link_tracks(per_frame_blobs, frame_w, frame_h)
    masks = [np.zeros((frame_h, frame_w), dtype=np.uint8) for _ in range(n_frames)]

    for track in tracks:
        if len(track) < MIN_TRACK_LEN:
            continue  # likely flicker/artifact, not a real persistent vein

        for b in track:
            _draw_blob(masks[b.frame_idx], b.contour)

        for prev_b, next_b in zip(track, track[1:]):
            gap = next_b.frame_idx - prev_b.frame_idx
            if gap <= 1:
                continue
            for g, frame_idx in enumerate(range(prev_b.frame_idx + 1, next_b.frame_idx), start=1):
                t = g / gap
                src = prev_b if t < 0.5 else next_b
                interp_x = prev_b.centroid[0] + (next_b.centroid[0] - prev_b.centroid[0]) * t
                interp_y = prev_b.centroid[1] + (next_b.centroid[1] - prev_b.centroid[1]) * t
                shift = np.array([[interp_x - src.centroid[0], interp_y - src.centroid[1]]], dtype=np.float32)
                shifted = (src.contour.astype(np.float32) + shift).astype(np.int32)
                _draw_blob(masks[frame_idx], shifted)

    return masks
