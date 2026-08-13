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
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

import config
import groq_client
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
    """Split into 3 steps (schedule -> parallel classify -> sequential render) instead of
    one straight-through loop, specifically so the Groq calls can run concurrently. This
    is safe because the SCHEDULING decision (which ticks need a fresh classify call) only
    ever reads blob CENTROIDS -- never the actual n_class answer -- to detect a changed
    blob set (see the loop below); the n_class answer itself is only needed later, when
    replaying holds onto the render pass. That means scheduling can be computed once,
    up front, from segmentation alone (GPU, no Groq), and every scheduled classify call is
    then independent of every other one -- there is no correctness reason they were ever
    serial. See config.STAGE2_MAX_WORKERS for the concurrency level and its rate-limit
    reasoning. Trade-off: this buffers every sampled frame in memory for the duration of
    Pass 1 (previously streamed one frame at a time) -- acceptable for the short demo
    clips this project targets (a 2-minute clip at 0.5s/tick is ~240 frames), flagged here
    in case a much longer video ever makes that a real constraint.
    """
    info = video_io.probe_video(ultrasound_path)
    duration = max(info["duration_sec"], 1e-6)
    hold_frames = max(1, round(config.SEG_SAMPLE_INTERVAL_SEC * config.OUTPUT_FPS))

    # progress_cb was previously only called from Step 3 (render) -- Steps 1 and 2 (where
    # the real wall-clock time actually goes, especially Step 2's Groq calls) never
    # reported anything, so the UI sat at 0% for the whole job then jumped 0->100 in the
    # few seconds Step 3 takes. These weights split one continuous 0..1 signal across all
    # 3 steps, roughly proportional to where time is actually spent (Step 2 dominates).
    _SEG_WEIGHT, _CLASSIFY_WEIGHT, _RENDER_WEIGHT = 0.10, 0.75, 0.15

    def _report(frac_0_to_1: float) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(frac_0_to_1, 1.0)))

    pass1_t0 = time.monotonic()
    print(f"[pipeline] Pass 1 (segmentation + N1/N2/N3) starting — {duration:.1f}s of ultrasound video "
          f"at {config.SEG_SAMPLE_INTERVAL_SEC}s/tick (~{int(duration / config.SEG_SAMPLE_INTERVAL_SEC)} ticks)")

    # --- Step 1: segment every tick, decide the classify schedule (GPU + centroid math
    # only, no Groq calls yet) ---
    seg_t0 = time.monotonic()
    scheduled_ticks = []   # ticks selected for a fresh Stage 2 call
    all_ticks = []         # every tick, in order, needs_classify flag included
    last_classified_at = -1e9
    last_centroids = []    # just centroids -- scheduling never needs n_class/is_valid
    for ts, frame in video_io.iter_sample_frames(ultrasound_path, config.SEG_SAMPLE_INTERVAL_SEC):
        blobs, fascia = bpe.segment_frame(frame)
        max_dist = _frame_diagonal(frame) * config.BLOB_CHANGE_DEBOUNCE_FRAC

        elapsed_since_classify = ts - last_classified_at
        needs_classify = bool(blobs) and elapsed_since_classify >= config.VLM_MIN_INTERVAL_SEC and (
            elapsed_since_classify >= config.VLM_SAMPLE_INTERVAL_SEC
            or len(blobs) != len(last_centroids)
            or any(_nearest_match(b.centroid, last_centroids, max_dist) is None for b in blobs)
        )
        if needs_classify:
            last_classified_at = ts
            last_centroids = [{"centroid": b.centroid} for b in blobs]

        entry = {"ts": ts, "frame": frame, "blobs": blobs, "fascia": fascia, "needs_classify": needs_classify}
        all_ticks.append(entry)
        if needs_classify:
            scheduled_ticks.append(entry)
        _report(_SEG_WEIGHT * min(ts / duration, 1.0))
    print(f"[pipeline] Pass 1 segmentation+scheduling done in {time.monotonic() - seg_t0:.1f}s — "
          f"{len(all_ticks)} ticks, {len(scheduled_ticks)} scheduled for a Stage 2 Groq call")

    # --- Step 2: fire all scheduled Stage 2 calls concurrently -- each mutates its own
    # tick's blobs in place, independent of every other tick (see docstring above) ---
    classify_t0 = time.monotonic()
    n_scheduled = len(scheduled_ticks)
    n_failed = 0
    _report(_SEG_WEIGHT)  # segmentation done; classify step starts at this floor regardless
    if scheduled_ticks:
        # No submission stagger needed here (an earlier version added one) -- that was a
        # workaround for blind worker-count concurrency having no real signal for when to
        # hold back. groq_client._OtpmLimiter now makes every call reserve real budget
        # before firing, so simultaneous submission is safe: calls that don't fit the
        # budget queue inside reserve() rather than firing and eating a 429.
        with ThreadPoolExecutor(max_workers=config.STAGE2_MAX_WORKERS) as pool:
            futures = {pool.submit(s2.classify_blobs, e["frame"], e["blobs"], e["fascia"]): e
                       for e in scheduled_ticks}
            n_done = 0
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    # Confirmed real production crash this guards against: one call
                    # exhausting all rate-limit retries (or any other transient Groq/
                    # network failure) used to propagate all the way up through
                    # run_full_pipeline and kill the ENTIRE job -- $4.5 and 40 minutes of
                    # real spend producing zero usable output, because one tick out of
                    # dozens hit a bad draw. This tick's blobs simply stay unclassified
                    # (n_class=None, is_valid=True) -- already a normal, gracefully-
                    # rendered state elsewhere in this pipeline (see renderer.py), not a
                    # new failure mode. The rest of the job continues.
                    e = futures[fut]
                    e["classify_failed"] = True  # read back in Step 3 -- a failed call
                    # must NOT overwrite last_records with this tick's (blank) state; see
                    # Step 3 for why blanking it would erase an already-correct label
                    # rather than just leaving one frame unlabeled.
                    n_failed += 1
                    print(f"[pipeline] Stage 2 call FAILED for tick t={e['ts']:.1f}s, "
                          f"leaving its blob(s) unclassified -- continuing rest of job: {exc}")
                n_done += 1
                # This is the step that actually eats the wall-clock time (Groq calls) --
                # reporting per-completion here is what makes the progress bar move
                # continuously instead of sitting at one value for the whole job.
                _report(_SEG_WEIGHT + _CLASSIFY_WEIGHT * (n_done / n_scheduled))
    print(f"[pipeline] Pass 1 Stage 2 classification done in {time.monotonic() - classify_t0:.1f}s "
          f"({config.STAGE2_MAX_WORKERS} concurrent workers, {n_scheduled} calls"
          f"{f', {n_failed} FAILED (left unclassified)' if n_failed else ''})")
    _report(_SEG_WEIGHT + _CLASSIFY_WEIGHT)

    # --- Step 3: sequential replay -- apply holds (nearest-centroid, see module
    # docstring) using the now-filled-in classify results, and render ---
    writer = None
    ticks = []
    n_ticks_total = max(len(all_ticks), 1)
    last_records = []   # [{"centroid": (cx,cy), "n_class": str|None, "is_valid": bool}, ...]
    for tick_idx, entry in enumerate(all_ticks):
        ts, frame, blobs, fascia = entry["ts"], entry["frame"], entry["blobs"], entry["fascia"]
        max_dist = _frame_diagonal(frame) * config.BLOB_CHANGE_DEBOUNCE_FRAC

        # Only adopt this tick's fresh state when it was BOTH scheduled AND actually
        # succeeded -- a scheduled-but-failed call (see Step 2) must fall through to the
        # else branch exactly like a normal held tick, so it keeps showing the last
        # KNOWN-GOOD label instead of blanking every blob to n_class=None. Confirmed real
        # bug this fixes: the old version unconditionally overwrote last_records whenever
        # needs_classify was true, even on failure, which erased an already-correct
        # label a viewer had just seen a moment earlier -- not merely "one blank frame",
        # a regression of a real answer back to nothing.
        if entry["needs_classify"] and not entry.get("classify_failed"):
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
        _report(_SEG_WEIGHT + _CLASSIFY_WEIGHT + _RENDER_WEIGHT * ((tick_idx + 1) / n_ticks_total))

    if writer:
        writer.release()
    write_artifact(ticks, artifact_path)
    _report(1.0)
    pass1_elapsed = time.monotonic() - pass1_t0
    print(f"[pipeline] Pass 1 done in {pass1_elapsed:.1f}s ({len(ticks)} ticks, "
          f"{pass1_elapsed / max(len(ticks), 1):.2f}s/tick avg) — {n_scheduled} stage2_nclass Groq "
          f"calls run at {config.STAGE2_MAX_WORKERS}x concurrency (the real driver of Pass 1 wall-clock time)")
    return ticks


# --- Motion-triggered Stage 3a gating -------------------------------------------------
# Cheap CPU-only heuristic (no VLM cost) deciding when the webcam probe location is worth
# re-checking — see config.WEBCAM_LOCATION_MIN/MAX_INTERVAL_SEC / WEBCAM_MOTION_DIFF_THRESHOLD
# for why this replaced a fixed timer: a fixed interval either wastes calls during long
# static dwell periods or risks feeding stale location into vein naming when the probe
# moves mid-interval. This can't and shouldn't try to be a precise motion detector — it
# only needs to answer "does this frame look meaningfully different from the frame the
# last Stage 3a call was based on", cheaply enough to run on every tick.
_MOTION_DOWNSIZE = (64, 48)


def _small_gray(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, _MOTION_DOWNSIZE)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _webcam_changed(prev_small: np.ndarray, frame_bgr: np.ndarray, threshold: float) -> tuple[bool, np.ndarray]:
    small = _small_gray(frame_bgr)
    if prev_small is None:
        return True, small
    diff = float(np.abs(small - prev_small).mean())
    return diff >= threshold, small


# --- Pass 2: webcam location + vein naming ---------------------------------------------

def _stage3b_for_entry(entry: dict, resolved_location: dict):
    """Runs in a worker thread (see run_pass2 Step 4) — pure w.r.t. its own tick, no
    shared mutable state touched. Only blobs with a real N-class get sent to naming --
    stage3_vein_naming's whole vocabulary-gating design depends on trusting each blob's
    N-class to pick its allowed vein-name list; a blob Stage 2 never classified has no
    vocabulary to offer and would otherwise fall through to the naming module's
    defensive "union of all three lists" fallback -- exactly the unconstrained-vocabulary
    situation that let N2 blobs get named "Tributary" in the first place."""
    tick = entry["tick"]
    blob_dicts = [{"blob_id": b.blob_id, "n_class": b.n_class, "centroid": list(b.centroid)}
                  for b in tick.blobs if b.n_class in ("N1", "N2", "N3")]
    if not blob_dicts:
        return {}
    annotated_intermediate = renderer.draw_intermediate_frame(entry["frame"], tick.blobs, tick.fascia)
    return s3b.name_veins(blob_dicts, resolved_location, annotated_ultrasound_frame_bgr=annotated_intermediate)


def run_pass2(ticks: list, ultrasound_path: str, webcam_path: str, final_video_path: str,
              progress_cb=None, probe_log_dir: str = None, position_debug_video_path: str = None) -> None:
    """probe_log_dir: if given, every Stage-3a probe-location reading is appended to
    <probe_log_dir>/probe_location_log.jsonl (one JSON object per line: timestamp_sec +
    the full location dict, including probe_position_stage_a) and the exact webcam frame
    that reading was based on is saved to <probe_log_dir>/frames/t<seconds>s.jpg — lets a
    human cross-check each reading against the actual webcam video frame by frame,
    independent of what made it into the final rendered video.

    position_debug_video_path: if given, writes a continuous copy of the webcam video
    with Stage A's fast probe_position (0/1/uncertain) burned onto every frame — lets a
    human scrub through and visually verify the upstream binary split that Stage B's
    narrowed leg_level vocabulary depends on, since that split isn't otherwise visible in
    the final ultrasound-side output video at all.

    Split into 5 steps (schedule -> parallel Stage 3a -> resolve held locations -> parallel
    Stage 3b -> sequential render), mirroring run_pass1's restructure -- see that
    function's docstring for the underlying reasoning (scheduling only needs cheap
    CPU-only signals, never the actual VLM answer, so it can be computed once up front and
    every scheduled call then dispatched independently). Confirmed real complaint this
    fixes: Pass 2 was left fully sequential when Pass 1 was parallelized earlier, and was
    the actual reason a 2-minute clip took ~45 minutes end-to-end.

    One real coupling Pass 1 didn't have: Stage 3b's ANSWER depends on Stage 3a's ANSWER
    (the resolved probe location), not just on scheduling facts -- so Stage 3b calls
    cannot fire until Stage 3a's results are known. Step 3 (a fast, Groq-free sequential
    replay) resolves "what location applies to each naming tick" from Stage 3a's now-
    completed results BEFORE Stage 3b's parallel dispatch -- this is the one step that
    must stay sequential, and it's cheap (pure Python, no calls) so it doesn't cost
    meaningful wall-clock time."""
    hold_frames = max(1, round(config.SEG_SAMPLE_INTERVAL_SEC * config.OUTPUT_FPS))
    n_ticks = max(len(ticks), 1)

    _SCHED_W, _S3A_W, _S3B_W, _RENDER_W = 0.05, 0.40, 0.40, 0.15

    def _report(frac_0_to_1: float) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(frac_0_to_1, 1.0)))

    webcam_reader = video_io.TimestampFrameReader(webcam_path)
    us_frames = list(video_io.iter_sample_frames(ultrasound_path, config.SEG_SAMPLE_INTERVAL_SEC))

    pass2_t0 = time.monotonic()
    print(f"[pipeline] Pass 2 (webcam location + vein naming) starting — {n_ticks} ticks")

    # --- Step 1: schedule -- cheap CPU-only signals (motion diff, blob debounce), no Groq
    # calls, decides which ticks need Stage 3a and/or Stage 3b. Identical gating logic to
    # the old single-loop version; "location_refreshed" is now "needs_stage3a", which is
    # itself just a scheduling fact known without waiting for the actual answer. ---
    sched_t0 = time.monotonic()
    entries = []
    last_location_at = -1e9
    last_location_small_frame = None
    last_named_at = -1e9
    last_named_centroids = []  # scheduling only needs centroids, not actual vein_names --
                                # same principle as run_pass1's last_centroids
    for i, (tick, (ts, frame)) in enumerate(zip(ticks, us_frames)):
        webcam_frame = webcam_reader.get(ts)
        elapsed = ts - last_location_at
        needs_stage3a = False
        if webcam_frame is not None and (i == 0 or elapsed >= config.WEBCAM_LOCATION_MIN_INTERVAL_SEC):
            force = i == 0 or elapsed >= config.WEBCAM_LOCATION_MAX_INTERVAL_SEC
            changed, small = _webcam_changed(last_location_small_frame, webcam_frame,
                                              config.WEBCAM_MOTION_DIFF_THRESHOLD)
            if force or (changed and tick.blobs):
                needs_stage3a = True
                last_location_at = ts
                last_location_small_frame = small

        max_dist = _frame_diagonal(frame) * config.BLOB_CHANGE_DEBOUNCE_FRAC
        needs_naming = bool(tick.blobs) and (
            needs_stage3a
            or (ts - last_named_at) >= config.VLM_SAMPLE_INTERVAL_SEC
            or len(tick.blobs) != len(last_named_centroids)
            or any(_nearest_match(b.centroid, last_named_centroids, max_dist) is None for b in tick.blobs)
        )
        if needs_naming:
            last_named_at = ts
            last_named_centroids = [{"centroid": b.centroid} for b in tick.blobs]

        entries.append({"i": i, "ts": ts, "frame": frame, "webcam_frame": webcam_frame,
                         "tick": tick, "needs_stage3a": needs_stage3a, "needs_naming": needs_naming})
        _report(_SCHED_W * ((i + 1) / n_ticks))

    scheduled_3a = [e for e in entries if e["needs_stage3a"]]
    scheduled_naming = [e for e in entries if e["needs_naming"]]
    print(f"[pipeline] Pass 2 scheduling done in {time.monotonic() - sched_t0:.1f}s — "
          f"{len(scheduled_3a)} stage3a scheduled, {len(scheduled_naming)} naming scheduled")

    # --- Step 2: dispatch all scheduled Stage 3a calls concurrently -- each is a
    # standalone read_location(webcam_frame) call, independent of every other tick. ---
    s3a_t0 = time.monotonic()
    _report(_SCHED_W)
    stage3a_results: dict[int, dict] = {}
    n_s3a_failed = 0
    if scheduled_3a:
        with ThreadPoolExecutor(max_workers=config.STAGE3A_MAX_WORKERS) as pool:
            futures = {pool.submit(s3a.read_location, e["webcam_frame"]): e for e in scheduled_3a}
            n_done = 0
            for fut in as_completed(futures):
                e = futures[fut]
                try:
                    stage3a_results[e["i"]] = fut.result()
                except Exception as exc:
                    # Same non-fatal-degradation principle as Pass 1's Stage 2 dispatch --
                    # a single failed read_location() call must never crash the whole job.
                    # Step 3 below already falls back to the last successfully-held
                    # location when a tick's result is missing from this dict, so this
                    # tick's probe location just stays stale for one refresh cycle
                    # instead of the entire job dying.
                    n_s3a_failed += 1
                    print(f"[pipeline] Stage 3a call FAILED for tick t={e['ts']:.1f}s, "
                          f"holding previous location -- continuing rest of job: {exc}")
                n_done += 1
                _report(_SCHED_W + _S3A_W * (n_done / len(scheduled_3a)))
    print(f"[pipeline] Pass 2 Stage 3a done in {time.monotonic() - s3a_t0:.1f}s "
          f"({config.STAGE3A_MAX_WORKERS}x concurrency, {len(scheduled_3a)} calls"
          f"{f', {n_s3a_failed} FAILED (held previous location)' if n_s3a_failed else ''})")
    _report(_SCHED_W + _S3A_W)

    # --- Step 3: sequential, Groq-free replay -- resolve which location applies to each
    # tick (held between Stage 3a refreshes, same semantics as the old single-loop
    # version) now that Stage 3a's real answers are known. This is the one step that must
    # stay sequential (Stage 3b's answer depends on Stage 3a's answer, not just
    # scheduling), but it's pure Python/dict lookups -- no meaningful wall-clock cost. ---
    location = s3a.normalize({})
    last_probe_position = "uncertain"
    resolved_location_by_i: dict[int, dict] = {}
    for e in entries:
        if e["needs_stage3a"] and e["i"] in stage3a_results:
            # `.get`-style membership check, not a raw index -- a failed Stage 3a call
            # (see Step 2) leaves this tick's result absent rather than raising, and the
            # correct degradation here is to simply keep holding whatever location was
            # already current, exactly as if this tick had never been scheduled at all.
            location = stage3a_results[e["i"]]
            last_probe_position = location.get("probe_position_stage_a", "uncertain")
        resolved_location_by_i[e["i"]] = location
        e["resolved_probe_position"] = last_probe_position

    # --- Step 4: dispatch all scheduled Stage 3b naming calls concurrently, each using
    # its own tick's now-resolved location -- independent of every other naming tick. ---
    s3b_t0 = time.monotonic()
    naming_results: dict[int, dict] = {}
    n_s3b_failed = 0
    if scheduled_naming:
        with ThreadPoolExecutor(max_workers=config.STAGE3B_MAX_WORKERS) as pool:
            futures = {pool.submit(_stage3b_for_entry, e, resolved_location_by_i[e["i"]]): e
                       for e in scheduled_naming}
            n_done = 0
            for fut in as_completed(futures):
                e = futures[fut]
                try:
                    naming_results[e["i"]] = fut.result()
                except Exception as exc:
                    # Same non-fatal-degradation principle again -- Step 5's render
                    # already does naming_results.get(i, {}), so a missing entry here
                    # just means this tick's blobs keep showing "naming..." (see
                    # renderer.draw_final_frame) until the next naming refresh succeeds,
                    # instead of taking down the whole job.
                    n_s3b_failed += 1
                    print(f"[pipeline] Stage 3b call FAILED for tick t={e['ts']:.1f}s, "
                          f"leaving its blob(s) unnamed -- continuing rest of job: {exc}")
                n_done += 1
                _report(_SCHED_W + _S3A_W + _S3B_W * (n_done / len(scheduled_naming)))
    print(f"[pipeline] Pass 2 Stage 3b done in {time.monotonic() - s3b_t0:.1f}s "
          f"({config.STAGE3B_MAX_WORKERS}x concurrency, {len(scheduled_naming)} calls"
          f"{f', {n_s3b_failed} FAILED (left unnamed)' if n_s3b_failed else ''})")
    _report(_SCHED_W + _S3A_W + _S3B_W)

    # --- Step 5: sequential render -- apply held naming (nearest-centroid, see module
    # docstring), write the probe log / position-debug video / final video. No Groq calls
    # in this step, matching run_pass1's Step 3. ---
    render_t0 = time.monotonic()
    probe_log_file = None
    probe_frames_dir = None
    if probe_log_dir:
        os.makedirs(probe_log_dir, exist_ok=True)
        probe_frames_dir = os.path.join(probe_log_dir, "frames")
        os.makedirs(probe_frames_dir, exist_ok=True)
        probe_log_file = open(os.path.join(probe_log_dir, "probe_location_log.jsonl"), "w")

    writer = None
    position_writer = None
    last_named_records = []  # [{"centroid": (cx,cy), "vein_name": str}, ...] — matched by
                              # nearest centroid, NOT blob_id (see module docstring)
    try:
        for e in entries:
            i, ts, frame, webcam_frame, tick = e["i"], e["ts"], e["frame"], e["webcam_frame"], e["tick"]

            if e["needs_stage3a"] and probe_log_file:
                loc = resolved_location_by_i[i]
                log_entry = {"timestamp_sec": round(ts, 2), **loc}
                probe_log_file.write(json.dumps(log_entry) + "\n")
                probe_log_file.flush()
                if webcam_frame is not None:
                    cv2.imwrite(os.path.join(probe_frames_dir, f"t{ts:08.2f}s.jpg"), webcam_frame)

            if position_debug_video_path and webcam_frame is not None:
                debug_frame = renderer.draw_position_debug_frame(webcam_frame, e["resolved_probe_position"])
                if position_writer is None:
                    h, w = webcam_frame.shape[:2]
                    position_writer = video_io.OutputVideoWriter(position_debug_video_path,
                                                                  fps=config.OUTPUT_FPS, frame_size=(w, h))
                for _ in range(hold_frames):
                    position_writer.write(debug_frame)

            # Only adopt fresh naming when the call actually SUCCEEDED (i in
            # naming_results) -- confirmed real bug this fixes: the old version rebuilt
            # last_named_records from naming_results.get(i, {}) unconditionally whenever
            # needs_naming was true, and a FAILED call means that dict lookup returns {},
            # which wiped last_named_records to empty and erased an already-correct name
            # that had just been showing a moment earlier. Same principle as run_pass1's
            # classify_failed fix -- a failed call must fall through and keep showing the
            # last known-good state, not blank it.
            if e["needs_naming"] and i in naming_results:
                names_result = naming_results[i]
                by_id = {b.blob_id: b for b in tick.blobs}
                last_named_records = [
                    {"centroid": by_id[bid].centroid, "vein_name": v["vein_name"]}
                    for bid, v in names_result.items() if bid in by_id
                ]

            max_dist = _frame_diagonal(frame) * config.BLOB_CHANGE_DEBOUNCE_FRAC
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

            _report(_SCHED_W + _S3A_W + _S3B_W + _RENDER_W * ((i + 1) / n_ticks))
    finally:
        if writer:
            writer.release()
        if position_writer:
            position_writer.release()
        webcam_reader.release()
        if probe_log_file:
            probe_log_file.close()
        _report(1.0)
        print(f"[pipeline] Pass 2 render done in {time.monotonic() - render_t0:.1f}s")
        pass2_elapsed = time.monotonic() - pass2_t0
        print(f"[pipeline] Pass 2 done in {pass2_elapsed:.1f}s — {len(scheduled_3a)} stage3a calls, "
              f"{len(scheduled_naming)} stage3b naming calls ({len(scheduled_3a) + len(scheduled_naming)} "
              f"Groq calls total, at {config.STAGE3A_MAX_WORKERS}x/{config.STAGE3B_MAX_WORKERS}x concurrency)")


# --- Convenience wrapper for callers (Flask job, CLI) ----------------------------------

def run_full_pipeline(ultrasound_path: str, webcam_path: str, intermediate_video_path: str,
                       final_video_path: str, artifact_path: str, roi_out_dir: str,
                       progress_cb=None, probe_log_dir: str = None,
                       position_debug_video_path: str = None) -> None:
    """progress_cb(stage: str, frac: float) if given — stage is 'cropping', 'segmenting',
    or 'naming'. The webcam video is never ROI-cropped (it's a room/leg view, not a
    machine-UI capture) — only the ultrasound video goes through Stage 0.
    probe_log_dir: see run_pass2's docstring — logs every Stage-3a reading + the exact
    webcam frame it saw, for manual cross-checking against the source video.
    position_debug_video_path: see run_pass2's docstring — the Stage A 0/1/uncertain
    debug video."""
    job_t0 = time.monotonic()
    groq_client.usage_tracker.reset()  # per-job total, not cumulative across jobs in this
                                        # server process -- see groq_client._UsageTracker
    if progress_cb:
        progress_cb("cropping", 0.0)
    crop_t0 = time.monotonic()
    cropped_ultrasound_path = run_roi_crop(ultrasound_path, roi_out_dir)
    print(f"[pipeline] ROI crop done in {time.monotonic() - crop_t0:.1f}s")
    if progress_cb:
        progress_cb("cropping", 1.0)

    def p1_cb(frac):
        if progress_cb:
            progress_cb("segmenting", frac)

    def p2_cb(frac):
        if progress_cb:
            progress_cb("naming", frac)

    try:
        ticks = run_pass1(cropped_ultrasound_path, intermediate_video_path, artifact_path, progress_cb=p1_cb)
        run_pass2(ticks, cropped_ultrasound_path, webcam_path, final_video_path, progress_cb=p2_cb,
                  probe_log_dir=probe_log_dir, position_debug_video_path=position_debug_video_path)
        print(f"[pipeline] Full job done in {time.monotonic() - job_t0:.1f}s total "
              f"(cropping + Pass 1 segmentation + Pass 2 location/naming)")
    finally:
        # Printed even on a genuine crash (per-tick Groq failures no longer reach here at
        # all -- see run_pass1/run_pass2's try/except around every concurrent dispatch --
        # so anything that DOES still raise past this point is a real, different failure,
        # and the user should still see exactly how much was spent before it happened
        # rather than losing that visibility along with the job).
        usage = groq_client.usage_tracker.summary()
        print(f"[pipeline] Groq usage this job: {usage['calls']} calls, "
              f"{usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion "
              f"= {usage['total_tokens']:,} tokens total, ~${usage['est_cost_usd']:.2f} "
              f"estimated (pricing approximate -- see _UsageTracker.summary)")
