"""
Flask web app — upload a raw ultrasound machine UI video, get back an
ROI-cropped video that is viewable and downloadable in the browser.

Routes:
  GET  /                     → serve frontend
  POST /upload               → accept video, start background job → {job_id}
  GET  /status/<job_id>      → {status, progress, stage, roi, method, confidence}
  GET  /stream/<job_id>      → serve output video for inline browser playback
  GET  /download/<job_id>    → serve output video as attachment download
"""

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from pipeline import run_pipeline

# ── App setup ────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
AUDIT_LOG   = os.path.join(BASE_DIR, "audit_log.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB upload limit

ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

# ── In-memory job store ───────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set(job_id: str, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def _get(job_id: str) -> dict | None:
    with _lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


# ── Background worker ─────────────────────────────────────────────────────────

def _worker(job_id: str, input_path: str):
    try:
        def on_progress(frac: float):
            _set(job_id, progress=min(99, int(frac * 100)))

        def on_status(msg: str):
            _set(job_id, stage=msg)

        result = run_pipeline(
            input_path,
            OUTPUT_DIR,
            use_registry=True,
            use_agent=True,
            on_progress=on_progress,
            audit_log_path=AUDIT_LOG,
            status_callback=on_status,
        )
        _set(
            job_id,
            status="done",
            progress=100,
            stage="Complete",
            output_filename=Path(result["output_path"]).name,
            roi=list(result["roi"]),
            method=result["method"],
            confidence=round(result["confidence"], 3),
            view_type=result.get("view_type", "UNKNOWN"),
        )
    except Exception as e:
        _set(job_id, status="error", stage="Failed", error=str(e))
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video field in request"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    job_id    = str(uuid.uuid4())
    save_name = f"{job_id}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    f.save(save_path)

    with _lock:
        _jobs[job_id] = {
            "status":   "processing",
            "progress": 0,
            "stage":    "Uploading complete — starting pipeline…",
        }

    thread = threading.Thread(target=_worker, args=(job_id, save_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = _get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/stream/<job_id>")
def stream(job_id: str):
    """Serve output video for inline browser playback (supports Range requests)."""
    job = _get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Job not ready"}), 404
    path = os.path.join(OUTPUT_DIR, job["output_filename"])
    if not os.path.exists(path):
        return jsonify({"error": "Output file missing"}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/download/<job_id>")
def download(job_id: str):
    """Serve output video as a browser download."""
    job = _get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Job not ready"}), 404
    path = os.path.join(OUTPUT_DIR, job["output_filename"])
    if not os.path.exists(path):
        return jsonify({"error": "Output file missing"}), 404
    return send_file(path, as_attachment=True, download_name=job["output_filename"])


if __name__ == "__main__":
    print("ROI Crop app running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
