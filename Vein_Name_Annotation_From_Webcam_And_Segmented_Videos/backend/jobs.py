"""In-memory job registry for the upload -> process -> download flow. No DB — job state
is lost on server restart, which is an acceptable trade-off for a single-shot batch tool."""
import os
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

    def to_status_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress_pct": round(self.progress_pct * 100, 1),
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(ultrasound_path: str, webcam_path: str) -> Job:
    job_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(config.OUTPUTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    job = Job(job_id, ultrasound_path, webcam_path, out_dir)
    with _lock:
        _jobs[job_id] = job
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
            progress_cb=progress_cb,
        )
        job.status = "done"
        job.stage = "done"
        job.progress_pct = 1.0
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        traceback.print_exc()
