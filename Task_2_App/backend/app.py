"""Task-2 backend — Probe Localisation & Active Guidance."""

import logging
import os
import sys

# Allow importing sibling modules without package install
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, send_from_directory
from flask_cors import CORS

from config import CORS_ORIGINS, PORT


def create_app() -> Flask:
    app = Flask(__name__)

    CORS(app, origins=CORS_ORIGINS)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ── register blueprints ────────────────────────────────────────────────────
    from routes.localization import localization_bp
    from routes.guidance import guidance_bp
    from routes.status import status_bp

    app.register_blueprint(localization_bp)
    app.register_blueprint(guidance_bp)
    app.register_blueprint(status_bp)

    # ── scenario API ───────────────────────────────────────────────────────────
    @app.route("/api/scenario/<scenario_id>")
    def get_scenario(scenario_id):
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from mock_data_generator import generate_scenario
        from flask import jsonify
        try:
            return jsonify(generate_scenario(scenario_id.upper()))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/api/scenarios")
    def list_scenarios():
        from flask import jsonify
        from mock_data_generator import _SCENARIO_BUILDERS
        return jsonify(list(_SCENARIO_BUILDERS.keys()))

    # ── serve assets (concatenated videos, etc.) ───────────────────────────────
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")

    @app.route("/assets/<path:filename>")
    def serve_assets(filename):
        return send_from_directory(assets_dir, filename)

    # ── serve frontend ─────────────────────────────────────────────────────────
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/test")
    def test_page():
        return send_from_directory(frontend_dir, "test.html")

    @app.route("/stream")
    def stream_page():
        return send_from_directory(frontend_dir, "stream.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(frontend_dir, filename)

    return app


if __name__ == "__main__":
    application = create_app()
    logging.getLogger(__name__).info(
        "Task-2 Probe Localisation server starting on http://127.0.0.1:%d", PORT
    )
    try:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception:
        pass
    application.run(host="0.0.0.0", port=PORT, debug=False)
