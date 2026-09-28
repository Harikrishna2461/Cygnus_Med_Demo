"""Central config: paths, model ids, sampling cadence. No classification logic lives here."""
import glob
import os

# --- Paths (all captured as absolute before biomedparse_engine ever chdirs) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend
PROJECT_DIR = os.path.dirname(BASE_DIR)                          # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos
CYGNUS_ROOT = os.path.dirname(PROJECT_DIR)                       # .../Cygnus_Med_Demo
# BioMedParse: the inference SOURCE (modeling/, utilities/, configs/, stubs/) is vendored in
# ../BiomedParse_Assets (~1MB). The two ~1.7GB finetuned WEIGHT files are not bundled -- drop
# them into BiomedParse_Assets/checkpoints/fascia/ and .../vein/, or point the CMED_* env vars
# below elsewhere (Developer Guide, "Model assets"). If neither is done, the original
# dev-machine locations are tried as a last resort.
def _find_task4_dir() -> str:
    """Root that holds BiomedParse/ (modeling, utilities, configs) and stubs/. Order: env
    override; the bundled BiomedParse_Assets/ next to backend/ (handover default); else walk
    up the ancestors for the original dev layout's sibling Task_4_VLM_Fascia_Vein_Detection."""
    env = os.getenv("CMED_TASK4_DIR")
    if env:
        return env
    bundled = os.path.join(PROJECT_DIR, "BiomedParse_Assets")
    if os.path.isdir(os.path.join(bundled, "BiomedParse")):
        return bundled
    cur = PROJECT_DIR
    for _ in range(5):
        cand = os.path.join(cur, "Task_4_VLM_Fascia_Vein_Detection")
        if os.path.isdir(cand):
            return cand
        cur = os.path.dirname(cur)
    return bundled


TASK4_DIR = _find_task4_dir()

UPLOADS_DIR = os.getenv("CMED_UPLOADS_DIR") or os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.getenv("CMED_OUTPUTS_DIR") or os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- BioMedParse (source vendored in BiomedParse_Assets/; weights placed or referenced) ---
BIOMEDPARSE_DIR = os.getenv("CMED_BIOMEDPARSE_DIR") or os.path.join(TASK4_DIR, "BiomedParse")
STUBS_DIR = os.getenv("CMED_STUBS_DIR") or os.path.join(TASK4_DIR, "stubs")
BIOMEDPARSE_CONFIG = os.path.join(BIOMEDPARSE_DIR, "configs", "biomed_fascia_finetuning.yaml")


def _resolve_ckpt_dir(env_name: str, bundled_subdir: str, dev_fallback: str) -> str:
    """Checkpoint dir: env override; else BiomedParse_Assets/checkpoints/<sub> if a
    model_state_dict.pt has been placed there (handover layout); else the original dev
    machine location. biomedparse_engine globs the newest model_state_dict.pt under it."""
    env = os.getenv(env_name)
    if env:
        return env
    bundled = os.path.join(PROJECT_DIR, "BiomedParse_Assets", "checkpoints", bundled_subdir)
    if glob.glob(os.path.join(bundled, "**", "model_state_dict.pt"), recursive=True):
        return bundled
    return dev_fallback


def _dev_task4_dir() -> str:
    """The original dev layout's Task_4 folder if it exists up the tree (last-resort weights)."""
    cur = PROJECT_DIR
    for _ in range(5):
        cand = os.path.join(cur, "Task_4_VLM_Fascia_Vein_Detection")
        if os.path.isdir(cand):
            return cand
        cur = os.path.dirname(cur)
    return os.path.join(PROJECT_DIR, "Task_4_VLM_Fascia_Vein_Detection")  # nonexistent; debuggable path


FASCIA_CKPT_DIR = _resolve_ckpt_dir(
    "CMED_FASCIA_CKPT_DIR", "fascia",
    os.path.join(_dev_task4_dir(), "BiomedParse", "output", "fascia_finetuning_v2_production"))
VEIN_CKPT_DIR = _resolve_ckpt_dir("CMED_VEIN_CKPT_DIR", "vein", r"D:\vein_phase3")
LOCAL_FALLBACK_WEIGHTS = os.getenv("CMED_FALLBACK_WEIGHTS") or os.path.join(TASK4_DIR, "pretrained", "biomedparse_v1.pt")

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
VEIN_MIN_AREA_FRAC = 0.002    # was 0.0002 in the original: 0.0002*(802*805)=129px let tiny speckle blobs through; on real output 22% of blobs were <1500px, mostly noise. 0.002 ~ 1290px. Lower it if genuinely small tributaries get dropped   # fraction of VEIN_AREA_REFERENCE_PX, not of the current frame
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

# --- Local Qwen VLM/LLM (llama.cpp `llama-server`, 4-bit GGUF) ---
# Replaces the hosted Groq API. ONE local model serves every reasoning/vision call: Stage 2
# (N1/N2/N3), Stage 3a (probe location), Stage 3b (vein naming), the ROI-crop LangGraph agent
# and the ROI VLM helper. Launched separately (start_llama_server.bat or docker-compose.yml);
# the app only needs its URL.
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "qwen-local")   # informational; server hosts one model
VLM_MAX_TOKENS = 3072            # default ceiling when a caller does not pass max_tokens
VLM_TEMPERATURE = 0.0
# "none" = Qwen chat-template enable_thinking=False (answer directly); "default" = full
# chain-of-thought (enable_thinking=True). Same two modes the pipeline always used.
VLM_REASONING_EFFORT = "none"
# Local evaluation (Qwen_Local_VLM_Evaluation_Report.docx, 50 calls over 5 tasks): thinking OFF was
# 25/25 correct at ~1.5-2.5s per call; thinking ON was 18/25 at ~15s avg, with every leg-level call
# truncating at its token cap. So thinking is forced OFF for EVERY call, overriding the per-call
# reasoning_effort="default" that the stage modules still pass (those were tuned for the hosted API).
# Set False only to A/B against thinking mode.
VLM_FORCE_NO_THINKING = True
VLM_TIMEOUT_SEC = 600            # local reasoning calls can legitimately run minutes
VLM_STARTUP_WAIT_SEC = 30        # how long a new job waits for llama-server before failing
VLM_TRANSIENT_RETRIES = 2        # connection resets / 503 "loading" / timeouts
# Hard clamp on any single call's max_tokens. Must stay below (server --ctx-size /
# --parallel) minus the prompt (up to 3 images + text, ~10k tokens) or the server rejects
# the request. With the shipped launch settings (--ctx-size 65536 --parallel 2 => 32768 per
# slot) 16384 leaves ample room.
VLM_MAX_TOKENS_CAP = 16384
# Simultaneous in-flight calls. MUST equal llama-server's --parallel value: each concurrent
# call occupies one decode slot (and its share of the KV cache). Extra callers queue inside
# vlm_client instead of hitting the server.
VLM_MAX_CONCURRENT = int(os.getenv("VLM_MAX_CONCURRENT", "2"))

# --- Sampling / debounce cadence (tune here, not inline in pipeline code) ---
SEG_SAMPLE_INTERVAL_SEC = 0.5
VLM_SAMPLE_INTERVAL_SEC = 4.0
# Stage 2 (N1/N2/N3) early-refresh floor -- HISTORICAL, disabled (0.0).
#
# run_pass1's needs_classify fires early (before VLM_SAMPLE_INTERVAL_SEC is up) whenever the
# blob count changes or a blob can't be centroid-matched to the last classified set. This
# floor was added to throttle BioMedParse's segmentation flicker back when the only backstop
# was the hosted API's rate limit. It had a real downside (delaying the very reclassify call
# that fixes a blank N-label after a genuine blob change), so it stays at 0.0. Locally the
# throttle that matters is VLM_MAX_CONCURRENT (decode slots), not call frequency. Kept as a
# knob in case a real run shows reclassify spam is saturating the GPU.
VLM_MIN_INTERVAL_SEC = 0.0

# Stage 2 classify calls are dispatched concurrently (see pipeline.run_pass1) -- each call
# only needs its own tick's frame/blobs/fascia, no cross-tick state. These constants only bound
# how many calls are submitted at once (CPU-side image encoding / prompt building); the REAL
# concurrency limit is VLM_MAX_CONCURRENT, enforced inside vlm_client (calls beyond it queue
# there). A single local GPU decodes one stream at a time per slot, so raising workers well
# past VLM_MAX_CONCURRENT buys nothing -- it just keeps the queue full.
STAGE2_MAX_WORKERS = 4

# Pass 2 (webcam location + vein naming) concurrency -- same reasoning as STAGE2_MAX_WORKERS.
# One read_location() call can itself fire 2-3 sequential sub-calls (Stage A, then Stage B or
# the reflux Agent LN+Agent S pair), so STAGE3A is kept lower than STAGE3B.
STAGE3A_MAX_WORKERS = 2
STAGE3B_MAX_WORKERS = 3
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
