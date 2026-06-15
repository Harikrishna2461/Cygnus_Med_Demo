import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PORT = int(os.getenv("PORT", 7861))

# All LLM/VLM calls use Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
GROQ_TEXT_MODEL  = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Probe localisation — sliding window for stability
SLIDING_WINDOW_SIZE = 10
STABILITY_THRESHOLD = 0.6   # fraction of window that must agree

# Anatomical boundary thresholds (segment_dist)
# Front of thigh: dist <= threshold → SFJ zone; above → GSV Thigh
SFJ_THIGH_MAX_DIST = 0.12
# Back of thigh: dist >= threshold → popliteal / SPJ zone
SPJ_THIGH_MIN_DIST = 0.78
# Back of calf: dist <= threshold → popliteal / SPJ zone; below → SSV
SPJ_CALF_MAX_DIST = 0.12

DB_PATH = os.path.join(os.path.dirname(__file__), "task2_state.db")

# CORS origins allowed
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
