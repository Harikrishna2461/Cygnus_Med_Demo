"""
Two-pass orchestration.

Pass 1: segment the ultrasound video, classify each blob N1/N2/N3 (Stage 2), render the
intermediate video, and serialize per-tick geometry to a JSON artifact.

Pass 2: replay that artifact (no BioMedParse re-run) alongside the webcam video, read
probe location (Stage 3a) and name veins (Stage 3b), render the final video.

Between VLM calls, the last classification/naming is held and reapplied to the newly
re-segmented blobs by NEAREST CENTROID (within a distance tolerance), not by blob_id.
blob_id is only a within-tick ordinal (position-sorted, reassigned fresh every tick) — it
is NOT a stable identity across ticks, confirmed empirically on real footage: across six
consecutive 0.5s ticks, blob count flickered 3/0/1/2/1/2 and blob_id=1 referred to a
different physical vein from one tick to the next (a y=46 tributary at tick 0, a y=141
vein at tick 2). Holding a classification by blob_id equality was silently applying stale
labels to the wrong physical vein whenever segmentation noise reordered or dropped a blob
mid-hold-window — this was a real, frequent source of wrong-looking N1/N2/N3 labels
despite individual VLM classification calls themselves being accurate. Nearest-centroid
matching (with no match = treated as unclassified, not stale-labeled) fixes this; it is
still not a true tracker (no motion model, no occlusion handling) — see BLOB_MATCH_MAX_DIST_FRAC.
"""
import json
import math
from dataclasses import dataclass

import numpy as np

import os

import config
import biomedparse_engine as bpe
import stage2_fascia_classify as s2
import stage3_webcam_location as s3a
import stage3_vein_naming as s3b
import renderer
import video_io
import roi_pipeline


@dataclass
class FrameTick:
    timestamp_sec: float
    frame_index: int
    blobs: list       # list[bpe.VeinBlob]
    fascia: object    # bpe.FasciaBoundary | None


def _frame_diagonal(frame_bgr) -> float:
    h, w = frame_bgr.shape[:2]
    return math.hypot(h, w)


def _nearest_match(centroid: tuple, records: list, max_dist: float):
    """Finds the record in `records` (each a dict with a "centroid" key) closest to
    `centroid`, within max_dist. Returns None if none qualify — callers must treat "no
    match" as "unclassified", not fall back to some arbitrary record. This is the fix
    for blob_id not being a stable cross-tick identity (see module docstring)."""
    best, best_d = None, max_dist
    for rec in records:
        d = math.hypot(centroid[0] - rec["centroid"][0], centroid[1] - rec["centroid"][1])
        if d < best_d:
            best_d, best = d, rec
    return best


# --- Artifact (de)serialization -------------------------------------------------------

def _blob_to_dict(b) -> dict:
    contour = b.contour.reshape(-1, 2).tolist() if b.contour is not None and len(b.contour) else []
    return {
        "blob_id": b.blob_id, "contour": contour, "centroid": list(b.centroid),
        "bbox": list(b.bbox), "area_px": b.area_px,
        "n_class": b.n_class, "n_class_reasoning": b.n_class_reasoning,
    }


def _blob_from_dict(d: dict):
    pts = d.get("contour") or []
    contour = (np.array(pts, dtype=np.int32).reshape(-1, 1, 2) if pts
               else np.zeros((0, 1, 2), dtype=np.int32))
    return bpe.VeinBlob(
        blob_id=d["blob_id"], contour=contour, centroid=tuple(d["centroid"]),
        bbox=tuple(d["bbox"]), area_px=d["area_px"],
        n_class=d.get("n_class"), n_class_reasoning=d.get("n_class_reasoning"),
    )


def _fascia_to_dict(f):
    if f is None:
        return None
    row = lambda arr: [None if np.isnan(x) else float(x) for x in arr]
    return {"sup_row_at_col": row(f.sup_row_at_col), "deep_row_at_col": row(f.deep_row_at_col)}


def _fascia_from_dict(d):
    if d is None:
        return None
    arr = lambda lst: np.array([np.nan if x is None else x for x in lst], dtype=np.float64)
    return bpe.FasciaBoundary(sup_row_at_col=arr(d["sup_row_at_col"]), deep_row_at_col=arr(d["deep_row_at_col"]))


def write_artifact(ticks: list, path: str) -> None:
    data = [
        {"timestamp_sec": t.timestamp_sec, "frame_index": t.frame_index,
         "blobs": [_blob_to_dict(b) for b in t.blobs], "fascia": _fascia_to_dict(t.fascia)}
        for t in ticks
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def read_artifact(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return [
        FrameTick(timestamp_sec=d["timestamp_sec"], frame_index=d["frame_index"],
                  blobs=[_blob_from_dict(bd) for bd in d["blobs"]], fascia=_fascia_from_dict(d["fascia"]))
        for d in data
    ]


# --- Stage 0: crop the ultrasound video down to the scan area, excluding machine UI ---

def run_roi_crop(ultrasound_path: str, out_dir: str) -> str:
    """Real clinical ultrasound recordings are a full machine-screen capture — menus,
    parameter panels, depth rulers, logos — with the actual scan image occupying only
    part of the frame (confirmed against a real Samsung machine recording during build:
    a large fraction of the frame is Chinese-language UI chrome). Feeding that whole
    frame to BioMedParse produces false-positive vein detections inside the UI area.
    Crops to the scan area only, using the VLM/CV ROI system from ROI_Identification/
    (copied into this project). Falls back to the uncropped video if ROI detection
    itself fails — a best-effort preprocessing step should not sink the whole job."""
    try:
        result = roi_pipeline.run_pipeline(ultrasound_path, out_dir, use_registry=True, use_agent=True)
        return result["output_path"]
    except Exception as exc:
        print(f"[pipeline] ROI crop failed ({exc}); continuing with the uncropped video.")
        return ultrasound_path


# --- Pass 1: segmentation + N1/N2/N3 ---------------------------------------------------

def run_pass1(ultrasound_path: str, intermediate_video_path: str, artifact_path: str,
              progress_cb=None) -> list:
    info = video_io.probe_video(ultrasound_path)
    duration = max(info["duration_sec"], 1e-6)
    hold_frames = max(1, round(config.SEG_SAMPLE_INTERVAL_SEC * config.OUTPUT_FPS))

    writer = None
    ticks = []
    last_classified_at = -1e9
    last_records = []   # [{"centroid": (cx,cy), "n_class": str|None, "is_valid": bool}, ...]

    for ts, frame in video_io.iter_sample_frames(ultrasound_path, config.SEG_SAMPLE_INTERVAL_SEC):
        blobs, fascia = bpe.segment_frame(frame)
        max_dist = _frame_diagonal(frame) * config.BLOB_CHANGE_DEBOUNCE_FRAC

        needs_classify = bool(blobs) and (
            (ts - last_classified_at) >= config.VLM_SAMPLE_INTERVAL_SEC
            or len(blobs) != len(last_records)
            or any(_nearest_match(b.centroid, last_records, max_dist) is None for b in blobs)
        )

        if needs_classify:
            s2.classify_blobs(frame, blobs, fascia)
            last_classified_at = ts
            last_records = [{"centroid": b.centroid, "n_class": b.n_class, "is_valid": b.is_valid}
                             for b in blobs]
        else:
            for b in blobs:
                match = _nearest_match(b.centroid, last_records, max_dist)
                b.n_class = match["n_class"] if match else None
                b.is_valid = match["is_valid"] if match else True

        # Drop blobs stage2 judged non-anatomical (text/watermark/logo) before they're
        # ever rendered or handed to stage3 — see biomedparse_engine.VeinBlob.is_valid.
        blobs = [b for b in blobs if b.is_valid]

        annotated = renderer.draw_intermediate_frame(frame, blobs, fascia)
        if writer is None:
            h, w = frame.shape[:2]
            writer = video_io.OutputVideoWriter(intermediate_video_path, fps=config.OUTPUT_FPS, frame_size=(w, h))
        for _ in range(hold_frames):
            writer.write(annotated)

        ticks.append(FrameTick(timestamp_sec=ts, frame_index=int(round(ts * info["fps"])), blobs=blobs, fascia=fascia))
        if progress_cb:
            progress_cb(min(ts / duration, 1.0))

    if writer:
        writer.release()
    write_artifact(ticks, artifact_path)
    return ticks


# --- Pass 2: webcam location + vein naming ---------------------------------------------

def run_pass2(ticks: list, ultrasound_path: str, webcam_path: str, final_video_path: str,
              progress_cb=None) -> None:
    hold_frames = max(1, round(config.SEG_SAMPLE_INTERVAL_SEC * config.OUTPUT_FPS))
    n_ticks = max(len(ticks), 1)

    webcam_reader = video_io.TimestampFrameReader(webcam_path)
    us_frames = video_io.iter_sample_frames(ultrasound_path, config.SEG_SAMPLE_INTERVAL_SEC)

    writer = None
    location = s3a.normalize({})
    last_location_at = -1e9
    last_named_at = -1e9
    last_named_records = []  # [{"centroid": (cx,cy), "vein_name": str}, ...] — see
                              # pipeline module docstring: matched by nearest centroid,
                              # NOT blob_id, since blob_id isn't a stable cross-tick identity

    try:
        for i, (tick, (ts, frame)) in enumerate(zip(ticks, us_frames)):
            location_refreshed = False
            if (ts - last_location_at) >= config.WEBCAM_LOCATION_MIN_INTERVAL_SEC or i == 0:
                webcam_frame = webcam_reader.get(ts)
                if webcam_frame is not None:
                    location = s3a.read_location(webcam_frame)
                    last_location_at = ts
                    location_refreshed = True

            max_dist = _frame_diagonal(frame) * config.BLOB_CHANGE_DEBOUNCE_FRAC
            needs_naming = bool(tick.blobs) and (
                location_refreshed
                or (ts - last_named_at) >= config.VLM_SAMPLE_INTERVAL_SEC
                or len(tick.blobs) != len(last_named_records)
                or any(_nearest_match(b.centroid, last_named_records, max_dist) is None for b in tick.blobs)
            )
            if needs_naming:
                blob_dicts = [{"blob_id": b.blob_id, "n_class": b.n_class, "centroid": list(b.centroid)}
                              for b in tick.blobs]
                annotated_intermediate = renderer.draw_intermediate_frame(frame, tick.blobs, tick.fascia)
                names_result = s3b.name_veins(blob_dicts, location, annotated_ultrasound_frame_bgr=annotated_intermediate)
                by_id = {b.blob_id: b for b in tick.blobs}
                last_named_records = [
                    {"centroid": by_id[bid].centroid, "vein_name": v["vein_name"]}
                    for bid, v in names_result.items() if bid in by_id
                ]
                last_named_at = ts

            current_names = {}
            for b in tick.blobs:
                match = _nearest_match(b.centroid, last_named_records, max_dist)
                if match:
                    current_names[b.blob_id] = match["vein_name"]

            final_frame = renderer.draw_final_frame(frame, tick.blobs, tick.fascia, current_names)
            if writer is None:
                h, w = frame.shape[:2]
                writer = video_io.OutputVideoWriter(final_video_path, fps=config.OUTPUT_FPS, frame_size=(w, h))
            for _ in range(hold_frames):
                writer.write(final_frame)

            if progress_cb:
                progress_cb((i + 1) / n_ticks)
    finally:
        if writer:
            writer.release()
        webcam_reader.release()


# --- Convenience wrapper for callers (Flask job, CLI) ----------------------------------

def run_full_pipeline(ultrasound_path: str, webcam_path: str, intermediate_video_path: str,
                       final_video_path: str, artifact_path: str, roi_out_dir: str,
                       progress_cb=None) -> None:
    """progress_cb(stage: str, frac: float) if given — stage is 'cropping', 'segmenting',
    or 'naming'. The webcam video is never ROI-cropped (it's a room/leg view, not a
    machine-UI capture) — only the ultrasound video goes through Stage 0."""
    if progress_cb:
        progress_cb("cropping", 0.0)
    cropped_ultrasound_path = run_roi_crop(ultrasound_path, roi_out_dir)
    if progress_cb:
        progress_cb("cropping", 1.0)

    def p1_cb(frac):
        if progress_cb:
            progress_cb("segmenting", frac)

    def p2_cb(frac):
        if progress_cb:
            progress_cb("naming", frac)

    ticks = run_pass1(cropped_ultrasound_path, intermediate_video_path, artifact_path, progress_cb=p1_cb)
    run_pass2(ticks, cropped_ultrasound_path, webcam_path, final_video_path, progress_cb=p2_cb)
