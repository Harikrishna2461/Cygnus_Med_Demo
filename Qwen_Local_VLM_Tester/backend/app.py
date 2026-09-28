"""Latency/quality tester for the local Qwen VLM engine against the kinds of tasks the
real vein-naming pipeline uses it for. Upload an image, pick a task preset (or write a
custom prompt), run it, see the raw output, parsed JSON, and wall-clock latency.

Flask serves the sibling frontend/ folder as static files -- same convention as
Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend/app.py (no Jinja).
"""
import base64
import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory

import config
import local_vlm_engine
import task_presets

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/tasks")
def api_tasks():
    return jsonify(task_presets.TASKS)


@app.route("/api/status")
def api_status():
    return jsonify(local_vlm_engine.status())


@app.route("/api/history")
def api_history():
    """Reads back every run ever logged to LOG_PATH (append-only, survives restarts and
    page refreshes) so the frontend can repopulate its history table on load instead of
    losing it whenever the tab is closed."""
    if not os.path.exists(config.LOG_PATH):
        return jsonify([])
    records = []
    with open(config.LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    limit = request.args.get("limit", type=int) or 200
    return jsonify(records[-limit:])


def _log_run(record: dict) -> None:
    with open(config.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.route("/api/run", methods=["POST"])
def api_run():
    status = local_vlm_engine.status()
    if status["state"] != "ready":
        return jsonify({"error": f"model not ready (state={status['state']}, "
                                  f"error={status.get('error')})"}), 503

    task_id = request.form.get("task_id", "custom")
    system_prompt = request.form.get("system_prompt", "")
    user_text = request.form.get("user_text", "")
    max_new_tokens = request.form.get("max_new_tokens", type=int)
    temperature = request.form.get("temperature", type=float)
    enable_thinking = request.form.get("enable_thinking", "true").lower() != "false"

    images = request.files.getlist("images")
    if not images:
        return jsonify({"error": "at least one image is required"}), 400

    image_b64_list = []
    saved_paths = []
    for i, img_file in enumerate(images):
        raw = img_file.read()
        image_b64_list.append(base64.b64encode(raw).decode())
        fname = f"{int(time.time() * 1000)}_{i}_{img_file.filename or 'image.png'}"
        fpath = os.path.join(config.UPLOAD_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(raw)
        saved_paths.append(fname)

    wall_t0 = time.monotonic()
    result = local_vlm_engine.run_inference(
        system_prompt=system_prompt,
        user_text=user_text,
        image_b64_list=image_b64_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    wall_elapsed = round(time.monotonic() - wall_t0, 3)
    result["wall_elapsed_seconds"] = wall_elapsed

    record = {
        "timestamp": time.time(),
        "task_id": task_id,
        "system_prompt": system_prompt,
        "user_text": user_text,
        "images": saved_paths,
        "result": result,
    }
    _log_run(record)

    return jsonify(result)


if __name__ == "__main__":
    # The model itself is served by a separately-launched llama-server process (see
    # README) -- local_vlm_engine just polls its /health endpoint, nothing to kick off
    # here.
    app.run(host="0.0.0.0", port=config.PORT, debug=False, threaded=True)
