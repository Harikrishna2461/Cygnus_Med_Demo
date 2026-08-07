"""Central config: paths, model ids, sampling cadence. No classification logic lives here."""
import os

# --- Paths (all captured as absolute before biomedparse_engine ever chdirs) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend
PROJECT_DIR = os.path.dirname(BASE_DIR)                          # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos
CYGNUS_ROOT = os.path.dirname(PROJECT_DIR)                       # .../Cygnus_Med_Demo
# This folder has been renamed back and forth at least twice mid-project
# (Task_4_VLM_Fascia_Vein_Detection <-> VLM_Fascia_Detection) — confirmed each rename
# broke the hardcoded path and crashed model loading (ModuleNotFoundError: detectron2,
# since the stubs/ shim path stopped resolving). Rather than hand-fixing this constant
# every time it happens again, resolve it dynamically: try each known name and use
# whichever actually exists on disk right now.
_TASK4_DIR_CANDIDATES = ["Task_4_VLM_Fascia_Vein_Detection", "VLM_Fascia_Detection"]
for _name in _TASK4_DIR_CANDIDATES:
    _candidate = os.path.join(CYGNUS_ROOT, _name)
    if os.path.isdir(_candidate):
        TASK4_DIR = _candidate
        break
else:
    # None exist — fall back to the most recently known name so the resulting error
    # message at least shows a real, debuggable path instead of failing silently here.
    TASK4_DIR = os.path.join(CYGNUS_ROOT, _TASK4_DIR_CANDIDATES[0])

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

# Fascia line smoothing: degree of the global least-squares polynomial fit through the
# raw per-column readings (see biomedparse_engine._fit_fascia_curve). 3 = cubic: enough
# flexibility for a genuine gentle asymmetric curve, low enough to not chase pixel noise.
FASCIA_POLY_DEGREE = 3

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
# Stage 3a (webcam probe location) always runs reasoning_effort="default" (full
# chain-of-thought) — confirmed necessary TWICE: "none" mode was tried once with a
# previous-reading prior in the prompt (caused the model to blindly anchor on the prior
# and repeat one leg_level for an entire video), and tried again with that prior removed
# entirely (STILL collapsed leg_level to the reference image's own scenario on every
# frame, AND still flipped leg_side unpredictably — the exact original bug this setting
# was chosen to fix in the first place). This is a genuine capability gap for this
# model/task, not a fixable prompt issue — call QUALITY cannot be cheapened here.
#
# Call FREQUENCY is the real, correct cost lever, but a fixed timer is the wrong version
# of that lever: a long fixed interval (e.g. 6s) risks feeding Stage 3b (vein naming) a
# stale location if the probe moves mid-interval, while a short fixed interval (e.g. 2s)
# wastes calls re-confirming a position that hasn't changed for the many seconds a
# clinician typically dwells at one scan location. See run_pass2 in pipeline.py: it now
# fires this call MOTION-TRIGGERED — a cheap CPU-only frame-diff check (no VLM cost) runs
# every tick, and only calls Stage 3a early when the webcam frame has actually changed
# meaningfully. These two constants bound that behavior:
WEBCAM_LOCATION_MIN_INTERVAL_SEC = 1.5   # never re-call faster than this even if motion
                                          # detected, to avoid jitter/noise spamming calls
WEBCAM_LOCATION_MAX_INTERVAL_SEC = 5.0   # safety net: force a call after this long even
                                          # with no detected motion, in case of slow drift
                                          # too gradual for the frame-diff check to catch.
                                          # Bounds worst-case staleness fed into vein
                                          # naming (Stage 3b) at 5s even if the motion
                                          # heuristic misses a real change entirely.
                                          # Simulated against a real 2-minute clip: 23
                                          # Stage 3a calls total (vs. 60 at the old flat
                                          # 2.0s interval, ~similar to a flat 6.0s interval
                                          # by count) but concentrated where the webcam
                                          # frame actually changes rather than spread
                                          # evenly — reacts within MIN_INTERVAL of real
                                          # movement instead of waiting out a fixed timer.
WEBCAM_MOTION_DIFF_THRESHOLD = 20.0      # mean abs grayscale pixel diff (0-255 scale) on
                                          # a 64x48 downsized frame vs. the frame from the
                                          # last actual Stage 3a call. Calibrated against 6
                                          # real frame-pairs from actual footage, not
                                          # guessed: same-clinical-position pairs (just
                                          # hand/cable movement, e.g. during the reflux
                                          # compression test) scored up to 19.6; genuine
                                          # probe-location changes scored 25-34. 20.0 sits
                                          # cleanly between those two clusters on this
                                          # sample, but it's a small sample (6 pairs) from
                                          # one video — re-tune if real usage shows
                                          # too-frequent or too-rare triggering.
                                          # WEBCAM_LOCATION_MAX_INTERVAL_SEC is the safety
                                          # net for whatever this heuristic misses.
BLOB_CHANGE_DEBOUNCE_FRAC = 0.05
OUTPUT_FPS = 10
WEBCAM_TIME_OFFSET_SEC = 0.0   # add to ultrasound timestamp before indexing into webcam video

# --- Video output encoding ---
# OpenCV's VideoWriter has no working H.264 encoder on this machine (OpenH264 DLL
# missing) and falls back to mp4v, which Chrome/Edge frequently refuse to play via
# <video>. Output is written with imageio-ffmpeg's bundled static ffmpeg binary instead
# (see video_io.OutputVideoWriter) — no system/admin install required.
OUTPUT_CODEC = "libx264"
