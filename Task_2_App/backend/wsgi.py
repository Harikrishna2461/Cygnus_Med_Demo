"""Gunicorn entry point — exposes the Flask+SocketIO app for production."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app, socketio  # noqa: E402

application = create_app()
