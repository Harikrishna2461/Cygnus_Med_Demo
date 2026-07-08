import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BACKEND_DIR, "..", "frontend"))
DIST_DIR     = os.path.join(FRONTEND_DIR, "dist")
UPLOAD_DIR   = os.path.join(BACKEND_DIR, "uploads")
OUTPUT_DIR   = os.path.join(BACKEND_DIR, "outputs")
PORT         = 5001

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

for d in (UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)


# ── Frontend build ──────────────────────────────────────────────────────────
def _npm(*args, **kwargs):
    cmd = ["npm.cmd" if sys.platform == "win32" else "npm", *args]
    return subprocess.run(cmd, **kwargs)


def ensure_frontend():
    built = os.path.isdir(DIST_DIR) and any(
        f for f in os.listdir(DIST_DIR) if f != ".gitkeep"
    )
    if built:
        return
    nm = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.isdir(nm):
        print("[setup] Installing npm packages...")
        _npm("install", cwd=FRONTEND_DIR, check=True)
    print("[setup] Building React frontend...")
    _npm("run", "build", cwd=FRONTEND_DIR, check=True)
    print("[setup] Frontend ready.")


# ── Port helpers ────────────────────────────────────────────────────────────
def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(port: int):
    if not port_in_use(port):
        return
    print(f"[setup] Freeing port {port} from existing process...")
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                f'netstat -ano | findstr ":{port} "',
                shell=True, text=True, stderr=subprocess.DEVNULL,
            )
            pids = {ln.split()[-1] for ln in out.strip().splitlines()
                    if ln.split() and ln.split()[-1].isdigit()}
            for pid in pids:
                if int(pid) > 0:
                    subprocess.run(f"taskkill /F /PID {pid}",
                                   shell=True, capture_output=True)
        else:
            subprocess.run(f"fuser -k {port}/tcp",
                           shell=True, capture_output=True)
    except Exception:
        pass
    time.sleep(0.8)


def wait_for_server(port: int, timeout: float = 12.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(port):
            return True
        time.sleep(0.1)
    return False


# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# job_id -> job dict
jobs: dict[str, dict] = {}

# job_id -> latest annotated frame (JPEG bytes) for live preview
latest_frames: dict[str, bytes] = {}


# ── API ─────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "port": PORT})


@app.route("/api/process", methods=["POST"])
def process():
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    file        = request.files["video"]
    sample_rate = max(1, int(request.form.get("sample_rate", 10)))
    job_id      = str(uuid.uuid4())

    input_path  = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_annotated.mp4")
    file.save(input_path)

    jobs[job_id] = {
        "status":           "queued",
        "progress":         0,
        "message":          "Queued - waiting to start...",
        "output_path":      output_path,
        "total_frames":     0,
        "frames_processed": 0,
        "veins_found":      0,
    }

    def run():
        from video_processor import process_video

        def on_progress(pct: int, msg: str, extra: dict | None = None):
            jobs[job_id]["progress"] = pct
            jobs[job_id]["message"]  = msg
            if extra:
                # video_processor may update output_path (e.g. .avi fallback)
                real_path = extra.pop("output_path", None)
                if real_path:
                    jobs[job_id]["output_path"] = real_path
                jobs[job_id].update(extra)

        def on_frame(jpeg: bytes):
            latest_frames[job_id] = jpeg

        try:
            jobs[job_id]["status"] = "processing"
            process_video(input_path, output_path, sample_rate, on_progress, on_frame)
            jobs[job_id]["status"] = "done"
        except Exception as exc:
            import traceback
            jobs[job_id]["status"]  = "error"
            jobs[job_id]["message"] = str(exc)
            print(traceback.format_exc())

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def get_status(job_id: str):
    def generate():
        while True:
            if job_id not in jobs:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                return
            payload = {k: v for k, v in jobs[job_id].items() if k != "output_path"}
            yield f"data: {json.dumps(payload)}\n\n"
            if jobs[job_id]["status"] in ("done", "error"):
                return
            time.sleep(0.4)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/stream/<job_id>")
def stream_frames(job_id: str):
    """MJPEG live-preview stream — one frame per loop iteration."""
    def generate():
        try:
            while True:
                frame = latest_frames.get(job_id)
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )

                status = jobs.get(job_id, {}).get("status", "unknown")
                if status in ("done", "error"):
                    # Flush one final frame then close
                    frame = latest_frames.get(job_id)
                    if frame:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                        )
                    break

                time.sleep(0.05)    # ~20 fps cap
        except GeneratorExit:
            pass

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/preview/<job_id>")
def preview(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    path = jobs[job_id]["output_path"]
    mime = "video/mp4" if path.endswith(".mp4") else "video/x-msvideo"
    return send_file(path, mimetype=mime, conditional=True)


@app.route("/api/download/<job_id>")
def download(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    path = jobs[job_id]["output_path"]
    mime = "video/mp4" if path.endswith(".mp4") else "video/x-msvideo"
    name = "vein_classified.mp4" if path.endswith(".mp4") else "vein_classified.avi"
    return send_file(path, mimetype=mime, as_attachment=True, download_name=name)


# ── Serve the React SPA (catch-all — registered LAST) ───────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path: str):
    full = os.path.join(DIST_DIR, path)
    if path and os.path.isfile(full):
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, "index.html")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    free_port(PORT)
    ensure_frontend()

    def open_browser():
        if wait_for_server(PORT):
            webbrowser.open(f"http://localhost:{PORT}")
        else:
            print(f"[warn] Server did not start on port {PORT}.")

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n  Vein VLM Classifier  ->  http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
