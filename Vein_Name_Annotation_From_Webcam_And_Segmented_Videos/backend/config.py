"""Central config: paths, model ids, sampling cadence. No classification logic lives here."""
import os

# --- Paths (all captured as absolute before biomedparse_engine ever chdirs) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend
PROJECT_DIR = os.path.dirname(BASE_DIR)                          # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos
CYGNUS_ROOT = os.path.dirname(PROJECT_DIR)                       # .../Cygnus_Med_Demo
TASK4_DIR = os.path.join(CYGNUS_ROOT, "Task_4_VLM_Fascia_Vein_Detection")

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- BioMedParse (referenced in place at Task_4, never copied) ---
BIOMEDPARSE_DIR = os.path.join(TASK4_DIR, "BiomedParse")
STUBS_DIR = os.path.join(TASK4_DIR, "stubs")
BIOMEDPARSE_CONFIG = os.path.join(BIOMEDPARSE_DIR, "configs", "biomed_fascia_finetuning.yaml")
FASCIA_CKPT_DIR = os.path.join(BIOMEDPARSE_DIR, "output", "fascia_finetuning_v2_production")
VEIN_CKPT_DIR = os.path.join(BIOMEDPARSE_DIR, "output", "fascia_vein_finetuning")
LOCAL_FALLBACK_WEIGHTS = os.path.join(TASK4_DIR, "pretrained", "biomedparse_v1.pt")

FASCIA_PROMPT = "fascia layer in PeripheralVascular Ultrasound"
VEIN_PROMPT = (
    "small oval anechoic dark void vein lumen in cross-section "
    "peripheral vascular ultrasound below fascia"
)
INFER_SIZE = 512

# Vein blob filtering (ported verbatim from Task_4/app.py::prob_to_vein_mask)
VEIN_PROB_THRESHOLD = 0.25
VEIN_MIN_AREA_FRAC = 0.0002   # of total image pixels
VEIN_MAX_AREA_FRAC = 0.025
VEIN_MAX_ASPECT_RATIO = 4.0
VEIN_MIN_CIRCULARITY = 0.15
VEIN_MAX_ANECHOIC_MEAN = 65.0

# Fascia two-line extraction (ported from Task_4/app.py::prob_to_fascia_two_lines)
FASCIA_PROB_THRESHOLD = 0.15

# --- Groq VLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
GROQ_VLM_MODEL = "qwen/qwen3.6-27b"
GROQ_MAX_TOKENS = 3072
GROQ_TEMPERATURE = 0.0
GROQ_TIMEOUT_SEC = 60

# --- Sampling / debounce cadence (tune here, not inline in pipeline code) ---
SEG_SAMPLE_INTERVAL_SEC = 0.5
VLM_SAMPLE_INTERVAL_SEC = 4.0
WEBCAM_LOCATION_MIN_INTERVAL_SEC = 8.0
BLOB_CHANGE_DEBOUNCE_FRAC = 0.05
OUTPUT_FPS = 10
WEBCAM_TIME_OFFSET_SEC = 0.0   # add to ultrasound timestamp before indexing into webcam video

# --- Video output encoding ---
# OpenCV's VideoWriter has no working H.264 encoder on this machine (OpenH264 DLL
# missing) and falls back to mp4v, which Chrome/Edge frequently refuse to play via
# <video>. Output is written with imageio-ffmpeg's bundled static ffmpeg binary instead
# (see video_io.OutputVideoWriter) — no system/admin install required.
OUTPUT_CODEC = "libx264"
