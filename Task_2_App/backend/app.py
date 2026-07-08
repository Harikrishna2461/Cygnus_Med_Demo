"""Task-2 backend — Probe Localisation & Active Guidance (+ Streaming mode)."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from config import CORS_ORIGINS, PORT, STREAM_VIDEO_PATH

socketio = SocketIO()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, origins=CORS_ORIGINS)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    socketio.init_app(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )

    # ── REST blueprints ────────────────────────────────────────────────────────
    from routes.localization import localization_bp
    from routes.guidance import guidance_bp
    from routes.status import status_bp

    app.register_blueprint(localization_bp)
    app.register_blueprint(guidance_bp)
    app.register_blueprint(status_bp)

    # ── WebSocket events ───────────────────────────────────────────────────────
    from routes.stream import register_stream_events
    register_stream_events(socketio)

    # ── scenario helpers ───────────────────────────────────────────────────────
    @app.route("/api/scenario/<scenario_id>")
    def get_scenario(scenario_id):
        from flask import jsonify
        from mock_data_generator import generate_scenario
        try:
            return jsonify(generate_scenario(scenario_id.upper()))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/api/scenarios")
    def list_scenarios():
        from flask import jsonify
        from mock_data_generator import _SCENARIO_BUILDERS
        return jsonify(list(_SCENARIO_BUILDERS.keys()))

    # ── video streaming (Range-request aware) ──────────────────────────────────
    @app.route("/video/stream-main")
    def stream_video():
        from flask import Response, request as freq, jsonify as fjson
        if not os.path.isfile(STREAM_VIDEO_PATH):
            return fjson({"error": "Stream video not found", "path": STREAM_VIDEO_PATH}), 404

        file_size = os.path.getsize(STREAM_VIDEO_PATH)
        range_header = freq.headers.get("Range")

        if range_header:
            try:
                parts = range_header.replace("bytes=", "").split("-")
                start = int(parts[0]) if parts[0] else 0
                end   = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            except (ValueError, IndexError):
                start, end = 0, file_size - 1
            end    = min(end, file_size - 1)
            length = end - start + 1

            def _chunk():
                with open(STREAM_VIDEO_PATH, "rb") as f:
                    f.seek(start)
                    rem = length
                    while rem > 0:
                        data = f.read(min(65536, rem))
                        if not data:
                            break
                        rem -= len(data)
                        yield data

            resp = Response(_chunk(), status=206, mimetype="video/mp4",
                            direct_passthrough=True)
            resp.headers["Content-Range"]  = f"bytes {start}-{end}/{file_size}"
            resp.headers["Accept-Ranges"]  = "bytes"
            resp.headers["Content-Length"] = str(length)
        else:
            def _full():
                with open(STREAM_VIDEO_PATH, "rb") as f:
                    while True:
                        data = f.read(65536)
                        if not data:
                            break
                        yield data

            resp = Response(_full(), status=200, mimetype="video/mp4",
                            direct_passthrough=True)
            resp.headers["Accept-Ranges"]  = "bytes"
            resp.headers["Content-Length"] = str(file_size)

        return resp

    # ── single frame image endpoint ────────────────────────────────────────────
    @app.route("/video/frame")
    def video_frame_endpoint():
        import base64
        from flask import Response, request as freq
        from streaming_guidance_engine import extract_frame_at
        try:
            pos_y = float(freq.args.get("pos_y", 0.0))
        except (TypeError, ValueError):
            return "Bad pos_y", 400
        # annotated=0 → raw video (Giacomini zone); annotated=1 → label overlay (default)
        annotated = freq.args.get("annotated", "1") != "0"
        b64 = extract_frame_at(pos_y, annotated=annotated)
        if not b64:
            return "Frame unavailable", 404
        resp = Response(base64.b64decode(b64), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # ── static assets ──────────────────────────────────────────────────────────
    assets_dir  = os.path.join(os.path.dirname(__file__), "..", "assets")
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

    @app.route("/assets/<path:filename>")
    def serve_assets(filename):
        return send_from_directory(assets_dir, filename)

    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/stream")
    def stream_page():
        return send_from_directory(frontend_dir, "stream.html")

    @app.route("/test")
    def test_page():
        return send_from_directory(frontend_dir, "test.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(frontend_dir, filename)

    return app


if __name__ == "__main__":
    application = create_app()
    log = logging.getLogger(__name__)
    log.info("Task-2 server starting on http://127.0.0.1:%d", PORT)
    log.info("  Single-frame mode : http://127.0.0.1:%d/", PORT)
    log.info("  Stream mode       : http://127.0.0.1:%d/stream", PORT)
    try:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{PORT}/stream")
    except Exception:
        pass
    socketio.run(
        application,
        host="0.0.0.0",
        port=PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
    )