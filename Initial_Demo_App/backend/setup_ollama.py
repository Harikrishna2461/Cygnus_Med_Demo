"""Check, start, and install the Ollama models used by this repository."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Iterable, List, Optional

import requests

from config import OLLAMA_BASE_URL, OLLAMA_SETUP_MODELS


WINDOWS_OLLAMA_CANDIDATES = [
    r"C:\Program Files\Ollama\ollama.exe",
    r"C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe",
]


def find_ollama_executable() -> Optional[str]:
    """Return a usable ollama executable path if one is available."""
    direct = os.environ.get("OLLAMA_EXE")
    if direct and os.path.exists(direct):
        return direct

    path = shutil.which("ollama")
    if path:
        return path

    expanded_candidates = [os.path.expandvars(candidate) for candidate in WINDOWS_OLLAMA_CANDIDATES]
    for candidate in expanded_candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def _service_ready(timeout_seconds: float = 1.0) -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_seconds)
        return response.status_code == 200
    except Exception:
        return False


def _normalize_model_name(model: str) -> str:
    return model.strip().lower()


def _model_matches(installed_name: str, requested_name: str) -> bool:
    installed = _normalize_model_name(installed_name)
    requested = _normalize_model_name(requested_name)
    return (
        installed == requested
        or installed.startswith(f"{requested}:")
        or installed.startswith(f"{requested}@")
    )


def list_installed_models() -> List[str]:
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    response.raise_for_status()
    return [model.get("name", "") for model in response.json().get("models", []) if model.get("name")]


def start_ollama_service() -> bool:
    """Start Ollama in the background if it is installed locally."""
    if _service_ready():
        return True

    executable = find_ollama_executable()
    if executable is None:
        return False

    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
    except FileNotFoundError:
        return False
    return True


def wait_for_service(timeout_seconds: int = 45) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _service_ready():
            return True
        time.sleep(1)
    return False


def pull_model(model_name: str) -> bool:
    executable = find_ollama_executable()
    if executable is None:
        return False

    completed = subprocess.run([executable, "pull", model_name], check=False)
    return completed.returncode == 0


def ensure_models(models: Iterable[str], auto_start: bool = True, pull_missing: bool = True) -> bool:
    """Ensure Ollama is reachable and the requested models are installed."""
    models_to_check = [model for model in dict.fromkeys(model.strip() for model in models if model and model.strip())]

    if not _service_ready():
        if not auto_start:
            return False

        if not start_ollama_service():
            return False

        if not wait_for_service():
            return False

    try:
        installed_models = list_installed_models()
    except Exception:
        return False

    missing_models = [
        model for model in models_to_check
        if not any(_model_matches(installed, model) for installed in installed_models)
    ]

    if not missing_models:
        return True

    if not pull_missing:
        return False

    for model in missing_models:
        if not pull_model(model):
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Ollama and install repository models.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check Ollama and report missing models without pulling anything.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=OLLAMA_SETUP_MODELS,
        help="Explicit model list to verify or install. Defaults to the repo model set.",
    )
    args = parser.parse_args()

    models = args.models or OLLAMA_SETUP_MODELS
    print(f"Checking Ollama at {OLLAMA_BASE_URL}")
    print(f"Models: {', '.join(models)}")

    ready = ensure_models(models, auto_start=not args.check_only, pull_missing=not args.check_only)
    if ready:
        print("Ollama setup is ready")
        return 0

    if not _service_ready():
        executable = find_ollama_executable()
        if executable is None:
            print("Ollama executable not found. Install Ollama first, then rerun this command.")
        else:
            print("Ollama is installed but the service is not reachable yet. Start it with: ollama serve")
    else:
        print("Ollama is reachable, but one or more models are missing or failed to pull.")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())