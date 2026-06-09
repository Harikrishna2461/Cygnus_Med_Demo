import requests as _requests
from flask import Blueprint, jsonify

from rag_engine import collection_exists, get_collection_size

bp = Blueprint("status", __name__)

_parent_module_loaded: bool = False


def set_parent_module_flag(loaded: bool) -> None:
    global _parent_module_loaded
    _parent_module_loaded = loaded


@bp.route("/api/status")
def api_status():
    ollama_ok = False
    ollama_model_ok = False
    try:
        r = _requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            models = [m.get("name", "") for m in r.json().get("models", [])]
            ollama_model_ok = any("llama3.2" in m for m in models)
    except Exception:
        pass

    return jsonify({
        "status": "running",
        "parent_module_loaded": _parent_module_loaded,
        "qdrant": {
            "collection_ready": collection_exists(),
            "document_count": get_collection_size(),
        },
        "ollama": {
            "running": ollama_ok,
            "model_ready": ollama_model_ok,
        },
    })
