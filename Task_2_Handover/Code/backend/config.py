import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PORT = int(os.getenv("PORT", 7861))

# All LLM/VLM calls use Groq
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_TEXT_MODEL   = os.getenv("GROQ_TEXT_MODEL",   "openai/gpt-oss-120b")   # ShuntAnalyst, GuidanceSpecialist — strict rule reasoning
GROQ_MID_MODEL    = os.getenv("GROQ_MID_MODEL",    "openai/gpt-oss-120b")   # ClinicalInterpreter, CircuitAnalyst, NavigationPlanner
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# Probe localisation — sliding window for stability
SLIDING_WINDOW_SIZE = 10
STABILITY_THRESHOLD = 0.6   # fraction of window that must agree

# Anatomical boundary thresholds (segment_dist, recomputed from bounding-box coordinates)
# seg_dist_thigh = posY / KNEE_N (KNEE_N=0.5497)
# seg_dist_calf  = (posY - KNEE_N) / (1 - KNEE_N)
# Front of thigh: dist <= threshold → SFJ zone (posY ≤ 0.07 → dist ≤ 0.07/0.5497 ≈ 0.127)
SFJ_THIGH_MAX_DIST = 0.13
# Back of thigh: dist >= threshold → popliteal / SPJ zone (posY ≥ 0.44 → dist ≥ 0.44/0.5497 ≈ 0.800)
SPJ_THIGH_MIN_DIST = 0.80
# Back of calf: dist <= threshold → popliteal / SPJ zone
# 0.10 ≈ upper 5–6 cm of posterior calf, covering the full popliteal fossa approach
SPJ_CALF_MAX_DIST = 0.10

# CORS origins allowed
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# Streaming mode — video path
STREAM_VIDEO_PATH = os.getenv(
    "STREAM_VIDEO_PATH",
    os.path.join(os.path.dirname(__file__), "..", "sample_data", "202207191643_00-Moving.mp4"),
)

# Placeholder — reserved for future windowing; full history is used while this is disabled
STREAM_HISTORY_WINDOW = 8

# Minimum posYRatio change before triggering a new VLM analysis
STREAM_VLM_THRESHOLD = 0.05

# Minimum posYRatio change before triggering a new LLM guidance call
# Raised from 0.03 → 0.06 to prevent flooding the crew on every mousemove tick
STREAM_LLM_THRESHOLD = 0.06
