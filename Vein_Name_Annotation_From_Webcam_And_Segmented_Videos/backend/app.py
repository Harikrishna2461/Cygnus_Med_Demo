"""Flask app: upload an ultrasound + webcam video pair, process in the background,
serve both annotated result videos. Deliberately simple — no DB, no auth, no queue."""
import os

from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

import config
import jobs

# Frontend lives in a sibling frontend/ folder, not inside backend/ — served directly as
# static files (no Jinja templating needed, index.html has no server-side variables).
FRONTEND_DIR = os.path.join(config.PROJECT_DIR, "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _save_upload(file_storage, upload_dir: str, field_name: str) -> str:
    filename = secure_filename(file_storage.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported file type for {field_name}: {filename!r}")
    dest = os.path.join(upload_dir, f"{field_name}{ext}")
    file_storage.save(dest)
    return dest


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/jobs", methods=["POST"])
def create_job():
    if "ultrasound_video" not in request.files or "webcam_video" not in request.files:
        return jsonify({"error": "both ultrasound_video and webcam_video files are required"}), 400

    upload_dir = os.path.join(config.UPLOADS_DIR, os.urandom(6).hex())
    os.makedirs(upload_dir, exist_ok=True)
    try:
        us_path = _save_upload(request.files["ultrasound_video"], upload_dir, "ultrasound")
        wc_path = _save_upload(request.files["webcam_video"], upload_dir, "webcam")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job = jobs.create_job(us_path, wc_path)
    return jsonify({"job_id": job.job_id})


@app.route("/api/jobs/<job_id>/status")
def job_status(job_id):
    job = jobs.get_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    return jsonify(job.to_status_dict())


@app.route("/api/jobs/<job_id>/result/<which>")
def job_result(job_id, which):
    job = jobs.get_job(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    path = {"intermediate": job.intermediate_path, "final": job.final_path}.get(which)
    if path is None:
        return jsonify({"error": "which must be 'intermediate' or 'final'"}), 400
    if job.status != "done" or not os.path.exists(path):
        return jsonify({"error": "result not ready"}), 409
    return send_file(path, mimetype="video/mp4")


if __name__ == "__main__":
    # Task_1_App=7860, Task_2_App=7861 — following the same port series.
    app.run(host="0.0.0.0", port=7862, debug=False, threaded=True)
