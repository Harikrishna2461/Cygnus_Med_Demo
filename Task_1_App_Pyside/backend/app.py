"""
CHIVA Clinical Assistant — Flask application entry point.
"""

import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
_SELF_DIR = Path(__file__).resolve().parent
_PARENT_BACKEND = _SELF_DIR.parent.parent / "backend"
sys.path.insert(0, str(_PARENT_BACKEND))
sys.path.insert(0, str(_SELF_DIR))

# ── Config (must come after path bootstrap) ───────────────────────────────────
from config import (
    HOST, PORT, DEBUG,
    LOG_FILE, LOG_LEVEL,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask
from flask_cors import CORS

app = Flask(__name__, static_folder=None)
CORS(app, origins=[f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"])

# ── Services (Groq + Qdrant clients) ─────────────────────────────────────────
import services
services.init_services()

# ── Classification module (optional parent backend) ───────────────────────────
try:
    from shunt_classification_and_ligation_llm import classify_and_plan_ligation_with_llm
    _PARENT_MODULE = True
except ImportError as _err:
    logger.warning(
        f"Parent module not found ({_err}). "
        "Copy shunt_classification_and_ligation_llm.py from the parent backend/ into "
        "this backend/ folder and restart."
    )
    _PARENT_MODULE = False

    def classify_and_plan_ligation_with_llm(*args, **kwargs):
        raise RuntimeError(
            "shunt_classification_and_ligation_llm.py not found. "
            "Copy it from backend/ into this folder and restart."
        )

# ── Wire flags into route modules ────────────────────────────────────────────
from routes.clinical import set_classification_fn
from routes.status import set_parent_module_flag

set_classification_fn(_PARENT_MODULE, classify_and_plan_ligation_with_llm)
set_parent_module_flag(_PARENT_MODULE)

# ── Register blueprints ───────────────────────────────────────────────────────
from routes import (
    views_bp, status_bp, sessions_bp,
    clinical_bp, general_bp, feedback_bp,
)

app.register_blueprint(views_bp)
app.register_blueprint(status_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(clinical_bp)
app.register_blueprint(general_bp)
app.register_blueprint(feedback_bp)


# ── Startup ───────────────────────────────────────────────────────────────────

def _startup() -> None:
    from chat_db import init_db
    from rag_engine import collection_exists, get_collection_size, load_bm25_from_qdrant
    from general_chat_engine import (
        collection_exists as gen_exists,
        get_collection_size as gen_size,
        load_bm25_from_collection as load_gen_bm25,
    )

    init_db()
    logger.info("Database initialised.")

    if not collection_exists():
        logger.error(
            f"Qdrant collection '{QDRANT_COLLECTION}' not found in {QDRANT_PATH}. "
            "Ensure the qdrant_storage directory is intact."
        )
    else:
        load_bm25_from_qdrant()
        logger.info(f"Clinical collection ready ({get_collection_size()} points).")

    if not gen_exists():
        logger.warning(
            "General collection 'final_structured_rag' not found. "
            "General chat mode will not be available."
        )
    else:
        load_gen_bm25()
        logger.info(f"General collection ready ({gen_size()} points).")

    logger.info(f"Medical Assistant ready at http://{HOST}:{PORT}")


def _open_browser() -> None:
    time.sleep(2)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    _startup()
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)