"""Central config: paths, model ids, sampling cadence. No classification logic lives here."""
import os

# --- Paths (all captured as absolute before biomedparse_engine ever chdirs) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend
PROJECT_DIR = os.path.dirname(BASE_DIR)                          # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos
CYGNUS_ROOT = os.path.dirname(PROJECT_DIR)                       # .../Cygnus_Med_Demo
# Folder was renamed from "Task_4_VLM_Fascia_Vein_Detection" to "VLM_Fascia_Detection"
# mid-project (confirmed: old name no longer exists on disk, new one has identical
# internal structure — stubs/, BiomedParse/output/ with the same checkpoint dirs).
TASK4_DIR = os.path.join(CYGNUS_ROOT, "VLM_Fascia_Detection")

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- BioMedParse (referenced in place at Task_4, never copied) ---
BIOMEDPARSE_DIR = os.path.join(TASK4_DIR, "BiomedParse")
STUBS_DIR = os.path.join(TASK4_DIR, "stubs")
BIOMEDPARSE_CONFIG = os.path.join(BIOMEDPARSE_DIR, "configs", "biomed_fascia_finetuning.yaml")
FASCIA_CKPT_DIR = os.path.join(BIOMEDPARSE_DIR, "output", "fascia_finetuning_v2_production")
VEIN_CKPT_DIR = r"D:\vein_phase3"
LOCAL_FALLBACK_WEIGHTS = os.path.join(TASK4_DIR, "pretrained", "biomedparse_v1.pt")

FASCIA_PROMPT = "fascia layer in PeripheralVascular Ultrasound"
VEIN_PROMPT = (
    "small oval anechoic dark void vein lumen in cross-section "
    "peripheral vascular ultrasound below fascia"
)
INFER_SIZE = 512

# Vein blob filtering (ported from Task_4/app.py::prob_to_vein_mask)
VEIN_PROB_THRESHOLD = 0.25
VEIN_MIN_AREA_FRAC = 0.0002   # fraction of VEIN_AREA_REFERENCE_PX, not of the current frame
VEIN_MAX_AREA_FRAC = 0.025
VEIN_MAX_ASPECT_RATIO = 4.0
VEIN_MIN_CIRCULARITY = 0.15
VEIN_MAX_ANECHOIC_MEAN = 65.0
# Fixed reference pixel count (~802x805, Task_4's own validated test-frame size) that
# VEIN_MIN/MAX_AREA_FRAC are fractions of. Keeps the size filter scale-invariant across
# different ROI-crop dimensions instead of rescaling with whatever the current frame
# happens to be — see prob_to_vein_mask for the real-data case this fixes.
VEIN_AREA_REFERENCE_PX = 802 * 805

# Fascia two-line extraction (ported from Task_4/app.py::prob_to_fascia_two_lines)
FASCIA_PROB_THRESHOLD = 0.15

# --- Groq VLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
GROQ_VLM_MODEL = "qwen/qwen3.6-27b"
GROQ_MAX_TOKENS = 3072
GROQ_TEMPERATURE = 0.0
# "none" disables the model's separate verbose <think> preamble (same setting the copied
# ROI_Identification agents already use on this model) — it still makes the call, just
# without spending most of the latency on visible chain-of-thought tokens first. This was
# the dominant cost in end-to-end runtime (~3min for a 20s test clip beforehand).
GROQ_REASONING_EFFORT = "none"
GROQ_TIMEOUT_SEC = 60

# --- Sampling / debounce cadence (tune here, not inline in pipeline code) ---
SEG_SAMPLE_INTERVAL_SEC = 0.5
VLM_SAMPLE_INTERVAL_SEC = 4.0
# Was 8.0s — set that high back when each Groq call took several seconds (reasoning
# mode) and needed cost-controlling. Now that GROQ_REASONING_EFFORT="none" makes calls
# ~1s each, that's no longer a good tradeoff: an 8s window means a mid-window probe move
# doesn't get picked up for up to 8s, so vein names can lag well behind where the probe
# actually is. Tightened to track probe movement responsively instead of the naming
# cadence (4.0s) being the more frequent of the two, which was backwards.
WEBCAM_LOCATION_MIN_INTERVAL_SEC = 2.0
BLOB_CHANGE_DEBOUNCE_FRAC = 0.05
OUTPUT_FPS = 10
WEBCAM_TIME_OFFSET_SEC = 0.0   # add to ultrasound timestamp before indexing into webcam video

# --- Video output encoding ---
# OpenCV's VideoWriter has no working H.264 encoder on this machine (OpenH264 DLL
# missing) and falls back to mp4v, which Chrome/Edge frequently refuse to play via
# <video>. Output is written with imageio-ffmpeg's bundled static ffmpeg binary instead
# (see video_io.OutputVideoWriter) — no system/admin install required.
OUTPUT_CODEC = "libx264"
