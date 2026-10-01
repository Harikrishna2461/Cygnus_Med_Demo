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

    # ── static asset paths (used by vein-frame route and static handlers below) ─
    assets_dir   = os.path.join(os.path.dirname(__file__), "..", "assets")
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

    # ── vein label display names (used by vein-frame route header) ──────────────
    _VEIN_LABELS: dict[str, str] = {
        "GSV_Prox":       "GSV proximal trunk",
        "GSV_Distal":     "GSV distal trunk",
        "GSV":            "GSV (great saphenous)",
        "SSV":            "SSV (small saphenous)",
        "CFV":            "CFV (common femoral)",
        "FV":             "FV (femoral vein)",
        "DFV":            "DFV (deep femoral)",
        "PV":             "PV (popliteal vein)",
        "Deep_Vein_Calf": "Deep calf vein",
        "FV_CFV":         "Deep vein (FV / CFV)",
        "AASV":           "AASV (anterior accessory)",
        "PASV":           "PASV (posterior accessory)",
        "Tributary":      "Superficial tributary",
        "Hunt_Perf":      "Hunterian perforator",
        "Dodd_Perf":      "Dodd perforator",
        "Boyd_Perf":      "Boyd perforator",
        "Cockett_Perf":   "Cockett perforator",
        "Ankle_Perf":     "Ankle perforator",
        "sfj":            "SFJ (saphenofemoral junction)",
        "gsv_thigh":      "GSV thigh",
        "gsv_calf":       "GSV calf",
        "spj":            "SPJ (saphenopopliteal junction)",
        "ssv":            "SSV (small saphenous)",
    }

    @app.route("/api/vein-frame")
    def vein_frame_ref():
        from flask import request as freq, send_file, jsonify as fjson
        from streaming_guidance_engine import _build_region_sources
        region = freq.args.get("region", "UNKNOWN")
        try:
            pos_y = float(freq.args.get("pos_y", 0.0))
        except (TypeError, ValueError):
            pos_y = 0.0

        # Use the identical source pool + selection logic as the VLM frame picker
        # so the placeholder always shows the same frame VLM will analyze.
        sources = _build_region_sources(region, assets_dir)
        if not sources:
            return fjson({"error": "no frames available for region"}), 404

        bucket = int(round(pos_y * 20))
        n_src  = len(sources)
        _, d, jpgs, _ = sources[bucket % n_src]
        frame_idx  = (bucket // n_src) % len(jpgs)
        chosen_path = os.path.join(d, jpgs[frame_idx])
        vein_label  = os.path.basename(d)

        resp = send_file(chosen_path, mimetype="image/jpeg")
        resp.headers["X-Vein-Type"]  = vein_label
        resp.headers["X-Vein-Label"] = _VEIN_LABELS.get(vein_label, vein_label)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Access-Control-Expose-Headers"] = "X-Vein-Type, X-Vein-Label"
        return resp

    # ── static assets ──────────────────────────────────────────────────────────
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