"""Gunicorn entry point — imports app and runs startup before workers fork."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app, _startup  # noqa: E402

_startup()
