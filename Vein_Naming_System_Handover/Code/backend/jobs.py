"""In-memory job registry for the upload -> process -> download flow. No DB — job state
is lost on server restart, which is an acceptable trade-off for a single-shot batch tool."""
import os
import re
import subprocess
import threading
import traceback
import uuid

import config
import pipeline


class Job:
    def __init__(self, job_id: str, ultrasound_path: str, webcam_path: str, out_dir: str):
        self.job_id = job_id
        self.ultrasound_path = ultrasound_path
        self.webcam_path = webcam_path
        self.status = "queued"   # queued|running|done|error
        self.stage = ""
        self.progress_pct = 0.0
        self.error = None
        self.intermediate_path = os.path.join(out_dir, "intermediate.mp4")
        self.final_path = os.path.join(out_dir, "final.mp4")
        self.artifact_path = os.path.join(out_dir, "artifact.json")
        self.roi_out_dir = os.path.join(out_dir, "roi_cropped")
        self.probe_log_dir = os.path.join(out_dir, "probe_location_log")
        self.position_debug_path = os.path.join(out_dir, "position_debug.mp4")
        # Browser-playable H.264 copy of the webcam upload (see _prepare_webcam_playback). Only
        # exists when the upload is not already H.264.
        self.webcam_playback_path = os.path.join(out_dir, "webcam_playback.mp4")

    def to_status_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress_pct": round(self.progress_pct * 100, 1),
            "error": self.error,
            # JSONL of every Stage-3a probe-location reading (timestamp + the exact
            # webcam frame it was based on, under frames/) — for manual cross-checking
            # against the source webcam video, not used by the app itself.
            "probe_log_path": os.path.join(self.probe_log_dir, "probe_location_log.jsonl"),
            # Whether the position-debug video (Stage A's 0/1/uncertain burned onto the
            # webcam video) is ready to fetch from /api/jobs/<id>/result/position_debug
            # — the frontend uses this to decide whether to show that optional player.
            "position_debug_ready": self.status == "done" and os.path.exists(self.position_debug_path),
        }


def _prepare_webcam_playback(job: "Job") -> None:
    """Browsers cannot play MPEG-4 Part 2 / HEVC / MJPEG-in-mp4 etc., so the webcam pane of the
    results page stays black when the raw upload uses one of those (confirmed: an `mp4v` webcam
    upload). If the upload is not H.264, transcode a playback copy with imageio-ffmpeg's bundled
    ffmpeg (video only, this copy is never used for analysis). Best effort -- a failure just leaves
    the original file in place."""
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        info = subprocess.run([ff, "-hide_banner", "-i", job.webcam_path],
                              capture_output=True, text=True).stderr
        m = re.search(r"Video: (\w+)", info)
        if m and m.group(1).lower() == "h264":
            return
        tmp = job.webcam_playback_path + ".part.mp4"
        r = subprocess.run([ff, "-y", "-i", job.webcam_path, "-an", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
                           capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(tmp, job.webcam_playback_path)
            print(f"[jobs] webcam re-encoded to H.264 for browser playback (source codec: {m.group(1) if m else '?'})")
        else:
            print(f"[jobs] webcam playback transcode failed: {r.stderr[-300:]}")
    except Exception as exc:  # noqa: BLE001
        print(f"[jobs] webcam playback transcode skipped: {exc}")


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(ultrasound_path: str, webcam_path: str) -> Job:
    job_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(config.OUTPUTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    job = Job(job_id, ultrasound_path, webcam_path, out_dir)
    with _lock:
        _jobs[job_id] = job
    threading.Thread(target=_prepare_webcam_playback, args=(job,), daemon=True).start()
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def _run_job(job: Job) -> None:
    job.status = "running"
    job.stage = "starting"

    def progress_cb(stage, frac):
        job.stage = stage
        job.progress_pct = frac

    try:
        pipeline.run_full_pipeline(
            job.ultrasound_path, job.webcam_path,
            job.intermediate_path, job.final_path, job.artifact_path, job.roi_out_dir,
            progress_cb=progress_cb, probe_log_dir=job.probe_log_dir,
            position_debug_video_path=job.position_debug_path,
        )
        job.status = "done"
        job.stage = "done"
        job.progress_pct = 1.0
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        traceback.print_exc()
