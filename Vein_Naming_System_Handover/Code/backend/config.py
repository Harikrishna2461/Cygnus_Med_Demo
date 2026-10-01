"""Central config: paths, model ids, sampling cadence. No classification logic lives here."""
import os

# --- Paths (all captured as absolute before biomedparse_engine ever chdirs) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend
PROJECT_DIR = os.path.dirname(BASE_DIR)                          # .../Vein_Name_Annotation_From_Webcam_And_Segmented_Videos
CYGNUS_ROOT = os.path.dirname(PROJECT_DIR)                       # .../Cygnus_Med_Demo

# Default HF_HOME to the bundled hf_cache/ next to Code/ -- set here (not just in run.bat) so
# it applies to every entry point (direct `python3 app.py`, a smoketest script; Docker's own
# HF_HOME already overrides this via docker-compose). Must run before biomedparse_engine's
# `import transformers` chain, which reads HF_HOME at import time. The BiomedBERT text encoder
# is bundled at hf_cache/hub/models--microsoft--BiomedNLP-.../, so the first job needs no
# internet. Never overrides an HF_HOME the caller already set.
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_DIR, "hf_cache"))
# BioMedParse: the inference SOURCE (modeling/, utilities/, configs/, stubs/) is vendored in
# ../BiomedParse_Assets (~1MB) -- this package is standalone and is never shipped next to the
# original Task_4_VLM_Fascia_Vein_Detection project, so there is deliberately no dev-machine
# fallback path here. The two ~1.7GB finetuned WEIGHT files are not bundled by default -- drop
# them into BiomedParse_Assets/checkpoints/fascia/ and .../vein/ (or point CMED_FASCIA_CKPT_DIR /
# CMED_VEIN_CKPT_DIR elsewhere). Missing weights fail loudly at load time (biomedparse_engine
# raises, naming the exact path expected) rather than silently downloading a different,
# non-finetuned model from Hugging Face -- see that module's _newest_ckpt.
TASK4_DIR = os.getenv("CMED_TASK4_DIR") or os.path.join(PROJECT_DIR, "BiomedParse_Assets")

UPLOADS_DIR = os.getenv("CMED_UPLOADS_DIR") or os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.getenv("CMED_OUTPUTS_DIR") or os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- BioMedParse (source vendored in BiomedParse_Assets/; weights placed or referenced) ---
BIOMEDPARSE_DIR = os.getenv("CMED_BIOMEDPARSE_DIR") or os.path.join(TASK4_DIR, "BiomedParse")
STUBS_DIR = os.getenv("CMED_STUBS_DIR") or os.path.join(TASK4_DIR, "stubs")
BIOMEDPARSE_CONFIG = os.path.join(BIOMEDPARSE_DIR, "configs", "biomed_fascia_finetuning.yaml")


def _resolve_ckpt_dir(env_name: str, bundled_subdir: str) -> str:
    """Checkpoint dir: env override, else always the bundled BiomedParse_Assets/checkpoints/<sub>
    (whether or not a model_state_dict.pt actually lives there yet). This package is standalone,
    so there is no dev-machine path to fall back to -- biomedparse_engine._newest_ckpt is what
    actually checks whether the file exists, and raises a clear, actionable error if it doesn't,
    naming this exact path. Never silently substitutes a different (non-finetuned) model."""
    return os.getenv(env_name) or os.path.join(PROJECT_DIR, "BiomedParse_Assets", "checkpoints", bundled_subdir)


FASCIA_CKPT_DIR = _resolve_ckpt_dir("CMED_FASCIA_CKPT_DIR", "fascia")
VEIN_CKPT_DIR = _resolve_ckpt_dir("CMED_VEIN_CKPT_DIR", "vein")

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
# Size-tiered admission for small blobs (config.VEIN_MIN_AREA_FRAC above is a hard floor --
# left untouched, do NOT lower it, that is what removed the speckle noise). A real small
# tributary can still be smaller than that floor, so anything between
# VEIN_SMALL_MIN_AREA_FRAC and VEIN_MIN_AREA_FRAC gets a SECOND CHANCE under stricter shape
# checks (rounder, darker, less elongated) instead of being dropped outright -- noise blobs
# in that size band are typically irregular/brighter (speckle clumps, not a real anechoic
# lumen), so this recovers genuine small veins without reopening the door the area-floor
# raise closed. Blobs at/above VEIN_MIN_AREA_FRAC are completely unaffected by this (same
# checks as before).
VEIN_SMALL_MIN_AREA_FRAC = 0.0006   # ~482px -- floor for the second-chance band
VEIN_SMALL_MAX_ANECHOIC_MEAN = 45.0  # stricter/darker than VEIN_MAX_ANECHOIC_MEAN (65)
VEIN_SMALL_MIN_CIRCULARITY = 0.55    # stricter/rounder than VEIN_MIN_CIRCULARITY (0.15)
VEIN_SMALL_MAX_ASPECT_RATIO = 2.2    # stricter than VEIN_MAX_ASPECT_RATIO (4.0)
# Fixed reference pixel count (~802x805, Task_4's own validated test-frame size) that
# VEIN_MIN/MAX_AREA_FRAC are fractions of. Keeps the size filter scale-invariant across
# different ROI-crop dimensions instead of rescaling with whatever the current frame
# happens to be — see prob_to_vein_mask for the real-data case this fixes.
VEIN_AREA_REFERENCE_PX = 802 * 805

# Fascia two-line extraction (ported from Task_4/app.py::prob_to_fascia_two_lines)
FASCIA_PROB_THRESHOLD = 0.15

# Pass 1 scheduling: forces a fresh Stage 2 call for a HELD (not freshly reclassified) blob
# whose position relative to the fascia lines has drifted this many px (in either d_sup or
# d_deep -- see stage2_fascia_classify._geometry_hint for the convention), or crossed a line
# outright, since it was last classified -- see pipeline._geometry_drifted. Deliberately well
# outside the few-px zone Stage 2's own prompt treats as still-ambiguous-but-N2, so this only
# fires on a real, class-relevant move, not sampling jitter. Verified on a real 2-minute clip:
# 67 of 164 Stage 2 calls were drift-triggered, and a full post-hoc scan of every held label
# against its own tick's geometry found zero remaining contradictions (see Developer Guide 10.4).
GEOMETRY_DRIFT_PX = 20.0

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