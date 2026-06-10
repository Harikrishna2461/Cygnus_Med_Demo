import os, subprocess
from pathlib import Path

os.environ['TF_CUDNN_USE_AUTOTUNE']        = '0'
os.environ['TF_XLA_FLAGS']                = '--tf_xla_auto_jit=0'
os.environ['TF_DISABLE_MKL']              = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

# ── Build frontend if dist is missing ────────────────────────────────────────
HERE         = Path(__file__).parent
FRONTEND_DIR = HERE / 'frontend'
DIST_DIR     = FRONTEND_DIR / 'dist'

if not DIST_DIR.exists():
    print("==> Building frontend (first run)...")
    npm = 'npm.cmd' if os.name == 'nt' else 'npm'
    subprocess.run([npm, 'install'], cwd=FRONTEND_DIR, check=True)
    subprocess.run([npm, 'run', 'build'], cwd=FRONTEND_DIR, check=True)
    print("==> Frontend built.")

# ── Imports ───────────────────────────────────────────────────────────────────
import json, uuid, base64, re, tempfile, io
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import pandas as pd

tf.config.optimizer.set_jit(False)

# ── Flask — serve React build as static files ─────────────────────────────────
app = Flask(__name__, static_folder=str(DIST_DIR), static_url_path='')
CORS(app)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    target = DIST_DIR / path
    if path and target.exists():
        return send_from_directory(str(DIST_DIR), path)
    return send_from_directory(str(DIST_DIR), 'index.html')

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
FASCIA_SIZE = 384
MEAN_NP     = np.array([0.485, 0.456, 0.406], np.float32)
STD_NP      = np.array([0.229, 0.224, 0.225], np.float32)
SEQ_LEN     = 5
DATA_ROOT   = HERE.parent / 'vein_detection_task_3_training'

LABEL_COLOR = {
    'N1':      (0,   0,   220),   # red
    'N2':      (0,   200,  50),   # green
    'N3':      (0,   140, 255),   # orange
    'unknown': (150, 150, 150),   # gray
}

VLM_PROMPT = (
    "Ultrasound image. Numbered yellow boxes = detected veins.\n"
    "If CYAN lines are visible they mark the fascia boundary.\n"
    "Classify each numbered vein using fascia lines if present, "
    "otherwise use tissue depth (shallow=N3, mid-depth at bright fascial layer=N2, deep=N1):\n"
    "- N3: vein is ABOVE the fascia (superficial)\n"
    "- N2: vein is ON or VERY CLOSE to fascia (GSV)\n"
    "- N1: vein is BELOW the fascia (deep)\n"
    "Reply ONLY with JSON: {\"1\": \"N2\", \"2\": \"N3\"}"
)

ECHOVLM_PROMPT = (
    "This is a B-mode ultrasound image (256x256 pixels, 0,0 = top-left).\n"
    "Identify:\n"
    "  1. The fascia band — the bright hyperechoic horizontal layer separating "
    "subcutaneous fat (above) from muscle (below).\n"
    "  2. Any veins — dark anechoic or hypoechoic oval/circular structures.\n"
    "Classify each vein:\n"
    "  N3 = above fascia (superficial vein)\n"
    "  N2 = at or very near the fascia (Great Saphenous Vein)\n"
    "  N1 = below fascia (deep vein)\n"
    "Reply ONLY with valid JSON, no extra text:\n"
    "{\"fascia_y\":<integer 0-255 for the fascia centre row, or null if not visible>,"
    "\"veins\":[{\"x\":<int>,\"y\":<int>,\"w\":<int>,\"h\":<int>,\"label\":\"N2\"}]}"
)

FOUNDATION_FEW_SHOT_PROMPT = (
    "You are analysing B-mode ultrasound images.\n"
    "The annotated reference images you already saw show:\n"
    "  • YELLOW lines  = fascia band (bright hyperechoic layer between fat and muscle)\n"
    "  • Coloured numbered boxes = veins (dark hypoechoic oval/circular structures)\n"
    "  • Labels: N3=above fascia (superficial), N2=at/near fascia (GSV), N1=below fascia (deep)\n\n"
    "Now analyse this new unannotated frame using the same anatomy.\n"
    "Find: (1) the fascia band row, (2) any dark oval vein structures.\n"
    "Reply ONLY with valid JSON — no extra text:\n"
    "{\"fascia_y\":<integer 0-255 for fascia centre row in this 256x256 image, or null>,"
    "\"veins\":[{\"x\":<int>,\"y\":<int>,\"w\":<int>,\"h\":<int>,\"label\":\"N2\"}]}\n"
    "All pixel coords are in 256x256 space (0,0 = top-left)."
)

# ── Hardcoded API keys ────────────────────────────────────────────────────────
DEFAULT_GROQ_KEY = ""
HF_TOKEN         = "REPLACE_WITH_ENV_HF_TOKEN"

def _make_groq(user_key: str = ''):
    from groq import Groq
    return Groq(api_key=user_key or DEFAULT_GROQ_KEY)

_store: dict = {}

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading models...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

fascia_model = tf.keras.models.load_model(
    str(DATA_ROOT / 'output/fascia/checkpoints/unet_resnet50_tf_best.keras'), compile=False)
vein_model = tf.keras.models.load_model(
    str(DATA_ROOT / 'output/vein/checkpoints/unet_resnet50_vein_best.keras'), compile=False)
lstm_model = tf.keras.models.load_model(
    str(DATA_ROOT / 'output/tracking/lstm_motion_predictor.keras'), compile=False)
print("Models ready.")

# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(bgr, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return ((cv2.resize(rgb, (size, size)) - MEAN_NP) / STD_NP)[None]

def predict_fascia(bgr):
    mask = np.argmax(fascia_model(preprocess(bgr, FASCIA_SIZE), training=False).numpy()[0], -1).astype(np.uint8)
    return cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

def predict_vein(bgr):
    probs     = vein_model(preprocess(bgr, IMG_SIZE), training=False).numpy()[0]
    vein_prob = probs[:, :, 1]
    mask      = (vein_prob > 0.50).astype(np.uint8)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def get_scan_x_limits(bgr):
    """
    Find (x_left, x_right) pixel columns of the actual ultrasound scan image.

    Strategy: morphological OPEN with a large rectangular kernel (~6% of frame
    height, min 25px).  This erases anything smaller than the kernel (text chars,
    UI markers, depth-ruler ticks) while preserving the large bright region of
    the actual scan image.  Computed once per video (caller caches the result).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, bright = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    k      = max(25, int(h * 0.06))          # ~25-50 px depending on resolution
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    # Columns where >5% of pixels survived the open = actual scan content
    col_bright = (opened > 0).sum(axis=0)
    xs = np.where(col_bright > h * 0.05)[0]
    if len(xs) < 10:
        return 0, w
    x_left, x_right = int(xs[0]), int(xs[-1] + 1)
    # If detected scan width is less than 40% of frame width the detection is
    # unreliable (e.g. transducer coupling artifact in corner triggers it).
    if (x_right - x_left) < w * 0.4:
        return 0, w
    return x_left, x_right

NEAR_PX = 20

def geometric_classify(cx, cy, top_y, bot_y):
    if top_y is None:
        return 'unknown'
    xi = max(0, min(IMG_SIZE - 1, int(cx)))
    if np.isnan(top_y[xi]):
        return 'unknown'
    ty, by = float(top_y[xi]), float(bot_y[xi])
    if cy < ty - NEAR_PX:
        return 'N3'
    elif cy > by + NEAR_PX:
        return 'N1'
    else:
        return 'N2'

def get_fascia_boundary(mask):
    # 1. Dominant-row: the row with the most class-1 pixels is the most reliable
    #    estimate of fascia depth.  Sparse outlier columns at wrong depths
    #    contribute ≤1 pixel/row there vs ~150 pixels/row at the true fascia.
    row_class1 = (mask == 1).sum(axis=1)
    if int(row_class1.max()) < 2:
        return None, None
    fascia_cy = int(np.argmax(row_class1))
    win  = 25
    y_lo = max(0, fascia_cy - win)
    y_hi = min(IMG_SIZE, fascia_cy + win + 1)

    # 2. Per-column: compute the CENTRE of the class-1 band (median of its y-values)
    #    instead of min/max.  min/max blow up when the model predicts a wide,
    #    noisy class-1 region in a column — the centre stays stable.
    center_y = np.full(IMG_SIZE, np.nan, np.float32)
    half_w   = np.full(IMG_SIZE, np.nan, np.float32)
    for x in range(IMG_SIZE):
        ys = np.where(mask[y_lo:y_hi, x] == 1)[0]
        if len(ys) == 0:
            continue
        ys = ys + y_lo
        center_y[x] = float(np.median(ys))
        half_w[x]   = float((ys[-1] - ys[0]) / 2) if len(ys) > 1 else 1.0

    valid = ~np.isnan(center_y)
    if valid.sum() < 20:   # require fascia spanning at least 20 columns to draw
        return None, None

    # 3. Use a FIXED half-width (median across all valid columns) so both lines
    #    move in parallel — no more asymmetric slopes.
    half = max(2.0, float(np.nanmedian(half_w)))

    # 4. IQR outlier rejection before interpolation — clamps column spikes
    cy_vals = center_y[valid]
    q25, q75 = float(np.percentile(cy_vals, 25)), float(np.percentile(cy_vals, 75))
    iqr = q75 - q25 if q75 > q25 else 6.0
    center_y[valid] = np.clip(cy_vals, q25 - 2.0 * iqr, q75 + 2.0 * iqr)

    xs_v  = np.where(valid)[0].astype(np.float32)
    x_min, x_max = int(xs_v.min()), int(xs_v.max())
    rng   = np.arange(x_min, x_max + 1, dtype=np.float32)
    ctr   = uniform_filter1d(np.interp(rng, xs_v, center_y[valid]), 60)

    top_out = np.full(IMG_SIZE, np.nan, np.float32)
    bot_out = np.full(IMG_SIZE, np.nan, np.float32)
    top_out[x_min:x_max+1] = ctr - half
    bot_out[x_min:x_max+1] = ctr + half
    return top_out, bot_out

def nms(dets, iou_thresh=0.3):
    if len(dets) <= 1:
        return dets
    dets_sorted = sorted(dets, key=lambda d: d['area'], reverse=True)
    kept = []
    for d in dets_sorted:
        if all(bbox_iou(d['bbox'], k['bbox']) < iou_thresh for k in kept):
            kept.append(d)
    return kept

def extract_detections(mask, min_area=50):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        dets.append({
            'bbox':     (stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP],
                         stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]),
            'centroid': (float(centroids[i][0]), float(centroids[i][1])),
            'area':     int(a),
        })
    return nms(dets)

# ── Pipeline B: Foundation-Model Fallback ─────────────────────────────────────
class PipelineB:
    """
    Activated when Pipeline A (task-specific DL) cannot detect the fascia or
    finds no veins.  Uses only foundation models so no task-specific training
    is required.

    Fascia  → Depth Anything V2 (monocular depth gradient) with a
              brightness-profile fallback.
    Veins   → Grounding DINO zero-shot detection with an OpenCV dark-blob
              fallback.

    All models are loaded lazily on first use so server start-up is unaffected.
    """

    def __init__(self):
        self._depth_pipe       = None   # HF depth-estimation pipeline (unused for US)
        self._gdino            = None   # (processor, model, device) tuple
        self._sam              = None   # (model, processor, device) tuple
        self._echovlm          = None   # (model, processor) tuple — primary VLM
        self._few_shot_b64     = []
        self._vlm_context_msgs = []
        self._tried            = False

    def ensure_loaded(self):
        if self._tried:
            return
        self._tried = True
        self._load_depth()
        self._load_gdino()
        self._load_sam()
        self._load_echovlm()
        self._load_few_shot()

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_depth(self):
        try:
            from transformers import pipeline as hf_pipe
            import torch
            dev = 0 if torch.cuda.is_available() else -1
            print("PipelineB: loading Depth Anything V2 Small …")
            self._depth_pipe = hf_pipe(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=dev,
                token=HF_TOKEN,
            )
            print("PipelineB: Depth Anything V2 ready.")
        except Exception as e:
            print(f"PipelineB: depth model unavailable ({e}); brightness fallback active.")

    def _load_gdino(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print("PipelineB: loading Grounding DINO Tiny …")
            proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", token=HF_TOKEN)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny", token=HF_TOKEN
            ).to(dev).eval()
            self._gdino = (proc, model, dev)
            print("PipelineB: Grounding DINO ready.")
        except Exception as e:
            print(f"PipelineB: GDINO unavailable ({e}); blob fallback active.")

    def _load_sam(self):
        try:
            from transformers import SamModel, SamProcessor
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print("PipelineB: loading SAM ViT-Base ...")
            model = SamModel.from_pretrained("facebook/sam-vit-base", token=HF_TOKEN).to(dev).eval()
            proc  = SamProcessor.from_pretrained("facebook/sam-vit-base", token=HF_TOKEN)
            self._sam = (model, proc, dev)
            print("PipelineB: SAM ready.")
        except Exception as e:
            print(f"PipelineB: SAM unavailable ({e}); per-column brightness fallback active.")

    def _load_echovlm(self):
        """
        Load EchoVLM (Qwen2.5VL-MoE fine-tuned on ultrasound) as the primary VLM.
        The repo has no setup.py so we clone it and add to sys.path directly.
        Falls back to Groq if unavailable (no GPU, git unavailable, etc.).
        Also requires: pip install qwen-vl-utils
        """
        try:
            import sys, subprocess, torch
            from transformers import AutoProcessor

            # Clone repo once into vendor/EchoVLM alongside this file
            echovlm_dir = HERE / 'vendor' / 'EchoVLM'
            if not echovlm_dir.exists():
                print("PipelineB: cloning EchoVLM repo (first run)...")
                subprocess.run(
                    ['git', 'clone', '--depth=1',
                     'https://github.com/Asunatan/EchoVLM.git',
                     str(echovlm_dir)],
                    check=True,
                )
                print("PipelineB: EchoVLM repo cloned.")

            if str(echovlm_dir) not in sys.path:
                sys.path.insert(0, str(echovlm_dir))

            from EchoVLM import Qwen2VLMOEForConditionalGeneration

            dev   = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            attn  = "flash_attention_2" if torch.cuda.is_available() else "eager"

            print("PipelineB: loading EchoVLM weights (chaoyinshe/EchoVLM)...")
            model = Qwen2VLMOEForConditionalGeneration.from_pretrained(
                "chaoyinshe/EchoVLM",
                torch_dtype=dtype,
                attn_implementation=attn,
                device_map="auto",
            )
            model.config.output_router_logits = False
            proc = AutoProcessor.from_pretrained("chaoyinshe/EchoVLM")
            self._echovlm = (model, proc)
            print(f"PipelineB: EchoVLM ready on {dev}.")
        except Exception as e:
            print(f"PipelineB: EchoVLM unavailable ({e}); Groq VLM fallback active.")
            self._echovlm = None

    def _load_few_shot(self):
        """
        Extract annotated reference frames and pre-build the VLM context messages.
        Everything is encoded HERE at startup — vlm_detect_classify only sends
        the current query frame at inference time.
        """
        self._few_shot_b64     = []
        self._vlm_context_msgs = []
        data_dir = DATA_ROOT / 'Data'

        for folder_prefix in ['3', '2']:
            anno_dir = next((p for p in sorted(data_dir.glob(f'{folder_prefix}*')) if p.is_dir()), None)
            if anno_dir is None:
                continue
            for fname in ['sample_data.mp4', '202207191643_00-Moving.mp4', '202207111318_38-Perf.mp4']:
                p = anno_dir / fname
                if not p.exists():
                    continue
                cap   = cv2.VideoCapture(str(p))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total < 5:
                    cap.release(); continue
                # 20 % and 50 % tend to show the large GSV vein and a range of anatomy
                for pct in [0.20, 0.50]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * pct)))
                    ret, frm = cap.read()
                    if ret:
                        small = cv2.resize(frm, (IMG_SIZE, IMG_SIZE))
                        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        self._few_shot_b64.append(base64.b64encode(buf.tobytes()).decode())
                cap.release()
                if len(self._few_shot_b64) >= 2:
                    break
            if len(self._few_shot_b64) >= 2:
                print(f"PipelineB: {len(self._few_shot_b64)} reference frames loaded from {anno_dir.name}")
                break

        if not self._few_shot_b64:
            print(f"PipelineB: no reference frames found under {data_dir}")
            return

        # ── Pre-build the VLM conversation context ONCE (startup cost, not per-frame) ──
        # Structured as a completed first turn so the model already "knows" the
        # annotated examples when each new query frame arrives.
        ref_content = [
            {"type": "text", "text": (
                "These are annotated B-mode ultrasound reference images.\n"
                "YELLOW lines    = fascia band (bright hyperechoic layer dividing fat from muscle).\n"
                "Coloured boxes  = veins — dark anechoic/hypoechoic ovals. "
                "Veins can be LARGE (filling 20-50%% of the image) or small.\n"
                "Labels: N3 = above fascia (superficial), "
                "N2 = at or very near fascia (GSV), N1 = below fascia (deep)."
            )},
        ]
        for b64 in self._few_shot_b64[:2]:
            ref_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        self._vlm_context_msgs = [
            {"role": "user",      "content": ref_content},
            {"role": "assistant", "content": (
                "Understood. I can see both annotated reference frames: yellow fascia lines "
                "and numbered vein boxes (including large dark oval vessels) with N1/N2/N3 labels. "
                "Ready to analyse new frames."
            )},
        ]
        print(f"PipelineB: VLM context pre-built with {len(self._few_shot_b64)} reference images "
              f"({sum(len(b) for b in self._few_shot_b64) // 1024} KB encoded at startup).")

    def vlm_detect_classify_echo(self, bgr):
        """
        Combined fascia + vein detection + N1/N2/N3 classification via EchoVLM
        (chaoyinshe/EchoVLM — Qwen2-VL-7B MoE fine-tuned on ultrasound).
        No few-shot examples needed — the model understands US anatomy natively.
        Returns (fascia_y_int_or_None, dets_list_with_vlm_label).
        """
        if self._echovlm is None:
            return None, []
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            import tempfile, os

            model, proc = self._echovlm

            # Save frame to a temp JPEG — qwen_vl_utils expects a file path or URL
            vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(tmp.name, vis)
            tmp.close()

            messages = [{"role": "user", "content": [
                {"type": "image", "image": tmp.name},
                {"type": "text",  "text":  ECHOVLM_PROMPT},
            ]}]

            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = proc(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=300)

            # Trim prompt tokens exactly as shown in the HF README
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            raw = proc.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            os.unlink(tmp.name)

            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return None, []
            data = json.loads(m.group())

            fascia_y = data.get('fascia_y')
            if fascia_y is not None:
                fascia_y = int(np.clip(int(fascia_y), 0, IMG_SIZE - 1))

            dets = []
            for v in data.get('veins', []):
                try:
                    x = int(np.clip(int(v['x']), 0, IMG_SIZE - 1))
                    y = int(np.clip(int(v['y']), 0, IMG_SIZE - 1))
                    w = max(4, int(np.clip(int(v['w']), 1, IMG_SIZE - x)))
                    h = max(4, int(np.clip(int(v['h']), 1, IMG_SIZE - y)))
                    label = v.get('label', 'N2')
                    if label not in ('N1', 'N2', 'N3'):
                        label = 'N2'
                    dets.append({
                        'bbox':      (x, y, w, h),
                        'centroid':  (float(x + w / 2), float(y + h / 2)),
                        'area':      w * h,
                        'vlm_label': label,
                    })
                except (KeyError, ValueError, TypeError):
                    continue
            return fascia_y, dets
        except Exception as e:
            print(f"EchoVLM detect error: {e}")
            return None, []

    def vlm_detect_classify(self, bgr, groq_client, vlm_model_nm):
        """
        Combined fascia detection + vein detection + classification via few-shot VLM.
        Passes annotated reference frames as examples before the query frame.
        Returns (fascia_y_int_or_None, dets_list_with_vlm_label).
        """
        vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
        cur_b64 = base64.b64encode(buf.tobytes()).decode()

        # Append the new query frame to the pre-built context (reference images are
        # already encoded in self._vlm_context_msgs — not re-sent each call).
        query_msg = {
            "role": "user",
            "content": [
                {"type": "text",      "text": "Analyse this new frame:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_b64}"}},
                {"type": "text",      "text": FOUNDATION_FEW_SHOT_PROMPT},
            ],
        }
        messages = (self._vlm_context_msgs or []) + [query_msg]

        try:
            resp = groq_client.chat.completions.create(
                model=vlm_model_nm,
                messages=messages,
                max_tokens=500, temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return None, []
            data = json.loads(m.group())

            fascia_y = data.get('fascia_y')
            if fascia_y is not None:
                fascia_y = int(np.clip(int(fascia_y), 0, IMG_SIZE - 1))

            dets = []
            for v in data.get('veins', []):
                try:
                    x = int(np.clip(int(v['x']), 0, IMG_SIZE - 1))
                    y = int(np.clip(int(v['y']), 0, IMG_SIZE - 1))
                    w = max(4, int(np.clip(int(v['w']), 1, IMG_SIZE - x)))
                    h = max(4, int(np.clip(int(v['h']), 1, IMG_SIZE - y)))
                    label = v.get('label', 'N2')
                    if label not in ('N1', 'N2', 'N3'):
                        label = 'N2'
                    dets.append({
                        'bbox':      (x, y, w, h),
                        'centroid':  (float(x + w / 2), float(y + h / 2)),
                        'area':      w * h,
                        'vlm_label': label,
                    })
                except (KeyError, ValueError, TypeError):
                    continue
            return fascia_y, dets
        except Exception as e:
            print(f"Foundation VLM detect error: {e}")
            return None, []

    # ── fascia ────────────────────────────────────────────────────────────────

    def _detect_scan_y_limits(self, gray_r, scan_xl_m, scan_xr_m):
        x_lo = max(0, scan_xl_m)
        x_hi = min(IMG_SIZE, scan_xr_m)
        row_mean = gray_r[:, x_lo:x_hi].mean(axis=1).astype(np.float64)
        grad = np.gradient(row_mean)
        top_candidates = np.where(grad[:IMG_SIZE // 2] > 3.0)[0]
        y_top = int(top_candidates[0]) + 3 if len(top_candidates) else int(IMG_SIZE * 0.10)
        thresh_bot = max(12.0, float(row_mean.max()) * 0.08)
        y_bot = next((y for y in range(IMG_SIZE-1, IMG_SIZE//2, -1) if row_mean[y] > thresh_bot), IMG_SIZE-1) + 1
        return y_top, y_bot

    def detect_fascia(self, bgr, scan_xl_m=0, scan_xr_m=IMG_SIZE):
        """GDINO -> SAM fascia detection. Returns (None, None) if either model unavailable."""
        approx_y = self._gdino_fascia(bgr)
        if approx_y is not None:
            return self._fascia_from_sam(bgr, approx_y, scan_xl_m, scan_xr_m)
        return None, None

    def _gdino_fascia(self, bgr):
        """Use GDINO to detect the fascia band. Returns approx fascia row in IMG_SIZE coords, or None."""
        if self._gdino is None:
            return None
        try:
            import torch
            from PIL import Image as PILImage
            proc, model, dev = self._gdino
            h0, w0 = bgr.shape[:2]
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            inputs = proc(
                images=pil,
                text="fascia . bright horizontal band . connective tissue layer",
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                outputs = model(**inputs)

            try:
                results = proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, target_sizes=[(h0, w0)]
                )[0]
                keep  = results['scores'].cpu() >= 0.25
                boxes = results['boxes'][keep].cpu().numpy()
            except TypeError:
                results = proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=0.25, text_threshold=0.20, target_sizes=[(h0, w0)]
                )[0]
                boxes = results['boxes'].cpu().numpy()

            if len(boxes) == 0:
                return None

            # Pick the widest, flattest box in the expected fascia zone (12–75 % depth)
            best_y, best_score = None, -1.0
            for x1, y1, x2, y2 in boxes:
                cy = (y1 + y2) / 2
                cy_norm = cy / h0
                if cy_norm < 0.12 or cy_norm > 0.75:
                    continue
                bw, bh = x2 - x1, max(y2 - y1, 1)
                score = (bw / w0) * (bw / bh)   # favour wide + flat
                if score > best_score:
                    best_score = score
                    best_y = int(cy * IMG_SIZE / h0)
            return best_y
        except Exception as e:
            print(f"PipelineB GDINO fascia error: {e}")
            return None

    def _fascia_from_sam(self, bgr, approx_y, scan_xl_m=0, scan_xr_m=IMG_SIZE):
        """
        Run SAM with a wide box prompt around approx_y to segment the fascia band.
        Extracts a per-column curved (top_y, bot_y) from the resulting mask.
        Falls through to brightness refinement if SAM is unavailable or mask is too sparse.
        """
        if self._sam is None:
            return None, None
        try:
            import torch
            from PIL import Image as PILImage
            model, proc, dev = self._sam
            h0, w0 = bgr.shape[:2]
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            # Wide box spanning full width, ±50 px around approx_y (in original coords)
            search = 50
            y1_box = max(0,  int((approx_y - search) * h0 / IMG_SIZE))
            y2_box = min(h0, int((approx_y + search) * h0 / IMG_SIZE))
            inputs = proc(
                pil,
                input_boxes=[[[0, y1_box, w0, y2_box]]],
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                outputs = model(**inputs)

            masks = proc.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            # masks[0] shape: (num_input_boxes, num_candidates, H, W) = (1, 3, H, W)
            iou_scores = outputs.iou_scores[0, 0].cpu().numpy()  # shape (3,)
            best_mask  = masks[0][0, int(np.argmax(iou_scores))].numpy().astype(np.uint8)
            mask_r     = cv2.resize(best_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

            x_lo = max(0, scan_xl_m)
            x_hi = min(IMG_SIZE, scan_xr_m)
            top_out = np.full(IMG_SIZE, np.nan, np.float32)
            bot_out = np.full(IMG_SIZE, np.nan, np.float32)
            for x in range(x_lo, x_hi):
                ys = np.where(mask_r[:, x] > 0)[0]
                if len(ys) == 0:
                    continue
                cy_col = float(np.median(ys))
                top_out[x] = float(max(0,            cy_col - 8))
                bot_out[x] = float(min(IMG_SIZE - 1, cy_col + 8))

            valid = ~np.isnan(top_out)
            if valid.sum() < 20:
                return None, None

            # IQR outlier rejection before smoothing
            tv = top_out[valid]
            q25, q75 = float(np.percentile(tv, 25)), float(np.percentile(tv, 75))
            iqr = q75 - q25 if q75 > q25 else 6.0
            top_out[valid] = np.clip(tv, q25 - 2.0 * iqr, q75 + 2.0 * iqr)
            bot_out[valid] = np.clip(bot_out[valid], q25 - 2.0 * iqr + 16, q75 + 2.0 * iqr + 16)

            xs  = np.arange(IMG_SIZE, dtype=np.float32)
            xv  = xs[valid]
            scan_xs = np.arange(x_lo, x_hi, dtype=np.float32)
            top_out[x_lo:x_hi] = uniform_filter1d(np.interp(scan_xs, xv, top_out[valid]), 60).astype(np.float32)
            bot_out[x_lo:x_hi] = uniform_filter1d(np.interp(scan_xs, xv, bot_out[valid]), 60).astype(np.float32)
            return top_out, bot_out
        except Exception as e:
            print(f"PipelineB SAM fascia error: {e}")
            return None, None

    def _refine_fascia_per_column(self, bgr, approx_y, search_half=25, scan_xl_m=0, scan_xr_m=IMG_SIZE):
        """
        Refine a rough fascia row estimate into a curved per-column representation.

        For each scan column finds the brightest pixel within [approx_y ± search_half]
        and uses that as the fascia peak.  IQR outlier rejection removes scan-edge
        artifacts before sigma-25 smoothing.

        Returns (top_y, bot_y) arrays of shape (IMG_SIZE,), NaN outside scan range.
        """
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.resize(gray.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        gray_s = cv2.GaussianBlur(gray_r, (7, 7), 0)

        y_lo = int(np.clip(approx_y - search_half, 0, IMG_SIZE - 1))
        y_hi = int(np.clip(approx_y + search_half + 1, 1, IMG_SIZE))
        half = 8
        x_lo = max(0, scan_xl_m)
        x_hi = min(IMG_SIZE, scan_xr_m)

        top_out = np.full(IMG_SIZE, np.nan, np.float32)
        bot_out = np.full(IMG_SIZE, np.nan, np.float32)

        for x in range(x_lo, x_hi):   # only scan columns
            col_strip = gray_s[y_lo:y_hi, x]
            if float(col_strip.max()) < 15:   # dark gel/shadow column — skip
                continue
            peak = y_lo + int(np.argmax(col_strip))
            top_out[x] = float(max(0,            peak - half))
            bot_out[x] = float(min(IMG_SIZE - 1, peak + half))

        valid = ~np.isnan(top_out)
        if valid.sum() < 20:
            # Not enough per-column signal — fall back to flat lines within scan range
            cy = float(np.clip(approx_y, half, IMG_SIZE - 1 - half))
            top_out[x_lo:x_hi] = cy - half
            bot_out[x_lo:x_hi] = cy + half
            return top_out, bot_out

        # IQR outlier rejection: clamp scan-edge spikes before smoothing
        tops = top_out[valid]
        q25, q75 = float(np.percentile(tops, 25)), float(np.percentile(tops, 75))
        iqr = q75 - q25 if q75 > q25 else 8.0
        lo_fence, hi_fence = q25 - 2.0 * iqr, q75 + 2.0 * iqr
        tops = np.clip(tops, lo_fence, hi_fence)
        top_out[valid] = tops
        bots = bot_out[valid]
        bot_out[valid] = np.clip(bots, lo_fence + 16, hi_fence + 16)

        xs  = np.arange(IMG_SIZE, dtype=np.float32)
        xv  = xs[valid]
        scan_xs = np.arange(x_lo, x_hi, dtype=np.float32)
        top_out[x_lo:x_hi] = uniform_filter1d(np.interp(scan_xs, xv, top_out[valid]), 60).astype(np.float32)
        bot_out[x_lo:x_hi] = uniform_filter1d(np.interp(scan_xs, xv, bot_out[valid]), 60).astype(np.float32)
        return top_out, bot_out

    def _fascia_from_brightness(self, bgr, scan_xl_m=0, scan_xr_m=IMG_SIZE):
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.resize(gray.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        x_lo, x_hi = max(0, scan_xl_m), min(IMG_SIZE, scan_xr_m)
        # Average only scan columns — dark side panels bias the gradient otherwise
        row_sm = uniform_filter1d(gray_r[:, x_lo:x_hi].mean(axis=1).astype(np.float64), 15)
        # Start search below the scan coupling line (bright echo at scan top)
        scan_y_top, _ = self._detect_scan_y_limits(gray_r, scan_xl_m, scan_xr_m)
        y0 = scan_y_top + 15
        y0 = max(y0, int(IMG_SIZE * 0.18))   # at least 18% from top
        y1 = int(IMG_SIZE * 0.75)
        if y0 >= y1:
            y0 = int(IMG_SIZE * 0.15)
        grad = np.gradient(row_sm)
        fascia_cy = y0 + int(np.argmax(grad[y0:y1]))
        return self._refine_fascia_per_column(bgr, fascia_cy, scan_xl_m=scan_xl_m, scan_xr_m=scan_xr_m)

    # ── veins ─────────────────────────────────────────────────────────────────

    def detect_veins(self, bgr, min_area=80):
        """GDINO -> SAM vein detection. Returns [] if GDINO unavailable or finds nothing."""
        if self._gdino is not None:
            dets = self._gdino_detect(bgr, min_area)
            if dets:
                return self._sam_refine_dets(bgr, dets)
        return []

    def _sam_refine_dets(self, bgr, gdino_dets):
        """
        Refine GDINO bounding boxes with SAM segmentation.
        Each GDINO box is used as a SAM prompt; the resulting mask gives a tighter,
        more accurate bounding box and centroid.
        Falls back to the original GDINO det if SAM is unavailable or mask is empty.
        """
        if self._sam is None or not gdino_dets:
            return gdino_dets
        try:
            import torch
            from PIL import Image as PILImage
            model, proc, dev = self._sam
            h0, w0 = bgr.shape[:2]
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            refined = []
            for det in gdino_dets:
                x, y, bw, bh = det['bbox']
                # Convert IMG_SIZE coords → original frame coords for SAM
                x1 = int(x  * w0 / IMG_SIZE);  y1 = int(y  * h0 / IMG_SIZE)
                x2 = int((x + bw) * w0 / IMG_SIZE); y2 = int((y + bh) * h0 / IMG_SIZE)

                inputs = proc(
                    pil,
                    input_boxes=[[[x1, y1, x2, y2]]],
                    return_tensors="pt",
                ).to(dev)
                with torch.no_grad():
                    outputs = model(**inputs)

                masks = proc.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )
                # masks[0] shape: (num_input_boxes, num_candidates, H, W) = (1, 3, H, W)
                iou_scores = outputs.iou_scores[0, 0].cpu().numpy()  # shape (3,)
                best_mask  = masks[0][0, int(np.argmax(iou_scores))].numpy().astype(np.uint8)
                mask_r     = cv2.resize(best_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

                ys_m, xs_m = np.where(mask_r > 0)
                if len(xs_m) < 4:
                    refined.append(det)
                    continue

                nx, ny = int(xs_m.min()), int(ys_m.min())
                nw = int(xs_m.max()) - nx
                nh = int(ys_m.max()) - ny
                if nw < 2 or nh < 2:
                    refined.append(det)
                    continue

                refined.append({
                    'bbox':     (nx, ny, nw, nh),
                    'centroid': (float(nx + nw / 2), float(ny + nh / 2)),
                    'area':     int(mask_r.sum()),
                })
            return nms(refined)
        except Exception as e:
            print(f"PipelineB SAM refine error: {e}")
            return gdino_dets

    def _gdino_detect(self, bgr, min_area):
        try:
            import torch
            from PIL import Image as PILImage
            proc, model, dev = self._gdino
            h0, w0 = bgr.shape[:2]
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            inputs = proc(
                images=pil,
                text="vein . dark oval blood vessel . anechoic structure",
                return_tensors="pt",
            ).to(dev)

            with torch.no_grad():
                outputs = model(**inputs)

            # transformers >=4.44 dropped box_threshold from post_process;
            # try new API first, fall back to old signature.
            try:
                results = proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, target_sizes=[(h0, w0)]
                )[0]
                keep = results['scores'].cpu() >= 0.20
                results = {'boxes': results['boxes'][keep]}
            except TypeError:
                results = proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=0.20, text_threshold=0.15, target_sizes=[(h0, w0)]
                )[0]

            edge_skip = int(IMG_SIZE * 0.08)   # ignore scan-border region at top
            dets = []
            for box in results['boxes'].cpu().numpy():
                x1, y1, x2, y2 = box
                x  = int(x1 * IMG_SIZE / w0)
                y  = int(y1 * IMG_SIZE / h0)
                bw = max(1, int((x2 - x1) * IMG_SIZE / w0))
                bh = max(1, int((y2 - y1) * IMG_SIZE / h0))
                area = bw * bh
                if area < min_area:
                    continue
                # Reject full-scan false positives: boxes covering >40% of image
                if area > IMG_SIZE * IMG_SIZE * 0.40:
                    continue
                # Reject scan-border artifacts: too close to top edge
                if y < edge_skip:
                    continue
                # Veins are roughly oval — reject very wide flat bars (bw > 4×bh)
                if bh > 0 and bw > bh * 4:
                    continue
                dets.append({
                    'bbox':     (x, y, bw, bh),
                    'centroid': (float(x + bw / 2), float(y + bh / 2)),
                    'area':     int(area),
                })
            return nms(dets)
        except Exception as e:
            print(f"PipelineB GDINO detect error: {e}")
            return []

    def _blob_detect(self, bgr, min_area, scan_xl_m=0, scan_xr_m=IMG_SIZE):
        """
        Dark-blob fallback for vein detection.

        Thresholds relative to the scan-area mean (not full-frame mean) to avoid
        the dark side-panel background dominating the threshold.  Scan masking
        prevents the machine UI panels from becoming false-positive detections.
        """
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        # 5×5 blur preserves small dark vein interiors; 9×9 was too aggressive
        blurred = cv2.GaussianBlur(gray_r, (5, 5), 0)

        x_lo, x_hi = max(0, scan_xl_m), min(IMG_SIZE, scan_xr_m)
        xi_lo, xi_hi = x_lo + 2, x_hi - 2   # 2-px inset to avoid border artifacts
        scan_y_top, scan_y_bot = self._detect_scan_y_limits(gray_r, scan_xl_m, scan_xr_m)

        # Use scan-area mean so dark side panels don't inflate the threshold
        scan_mean  = float(blurred[scan_y_top:scan_y_bot, xi_lo:xi_hi].mean())
        thresh_val = max(20, int(scan_mean * 0.45))
        _, binary  = cv2.threshold(
            blurred.astype(np.uint8), thresh_val, 255, cv2.THRESH_BINARY_INV
        )

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
        # Blank everything outside the inset scan area (kills background + UI panels)
        binary[:, :xi_lo]       = 0
        binary[:, xi_hi:]       = 0
        binary[:scan_y_top, :]  = 0
        binary[scan_y_bot:, :]  = 0

        scan_w = max(1, xi_hi - xi_lo)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        dets = []
        for i in range(1, n):
            a  = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if a < min_area:
                continue
            # Reject blobs spanning >55% of scan width (deep tissue bands, not veins)
            if bw > scan_w * 0.55:
                continue
            if min(bw, bh) == 0 or max(bw, bh) / min(bw, bh) > 3.5:
                continue
            comp = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            perim = cv2.arcLength(contours[0], True)
            if perim > 0 and (4 * np.pi * a / perim ** 2) < 0.20:
                continue   # too elongated or irregular to be a vein
            dets.append({
                'bbox':     (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], bw, bh),
                'centroid': (float(centroids[i][0]), float(centroids[i][1])),
                'area':     int(a),
            })
        return nms(dets)


_pipe_b = PipelineB()   # singleton — shared across all SSE requests


def vlm_classify(bgr, track_recs, top_y, bot_y, groq_client, vlm_model_name):
    """Classify a list of track_recs. Returns {1-based-index: label}."""
    if not track_recs or groq_client is None:
        return {}
    h0, w0 = bgr.shape[:2]
    sx, sy  = w0 / IMG_SIZE, h0 / IMG_SIZE
    vis = bgr.copy()
    if top_y is not None:
        xs_v = [x for x in range(IMG_SIZE) if not np.isnan(top_y[x])]
        if len(xs_v) > 1:
            pts_top = np.array([[int(x*sx), int(top_y[x]*sy)] for x in xs_v], np.int32)
            pts_bot = np.array([[int(x*sx), int(bot_y[x]*sy)] for x in xs_v], np.int32)
            cv2.polylines(vis, [pts_top], False, (255, 255, 0), 2)
            cv2.polylines(vis, [pts_bot], False, (255, 255, 0), 2)
    for idx, r in enumerate(track_recs, 1):
        x, y, w, h = int(r['x']*sx), int(r['y']*sy), int(r['w']*sx), int(r['h']*sy)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(vis, str(idx), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    vis_small = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode('.jpg', vis_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64    = base64.b64encode(buf.tobytes()).decode()
    try:
        resp = groq_client.chat.completions.create(
            model=vlm_model_name,
            messages=[{'role':'user','content':[
                {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}},
                {'type':'text','text':VLM_PROMPT},
            ]}],
            max_tokens=150, temperature=0.0)
        raw = resp.choices[0].message.content.strip()
        m   = re.search(r'\{[^}]+\}', raw)
        if m:
            return {int(k): v for k, v in json.loads(m.group()).items() if v in ('N1','N2','N3')}
    except Exception as e:
        print(f'VLM error: {e}')
    return {}

# ── Tracker ───────────────────────────────────────────────────────────────────
def bbox_iou(b1, b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    ix = max(0, min(x1+w1,x2+w2) - max(x1,x2))
    iy = max(0, min(y1+h1,y2+h2) - max(y1,y2))
    return (ix*iy) / (w1*h1 + w2*h2 - ix*iy + 1e-6)

def _centroid(det):
    x, y, w, h = det['bbox']
    return np.array([x + w / 2, y + h / 2], np.float32)

def match_dets(prev, curr, thresh=0.2, max_dist=45):
    if not prev or not curr:
        return [], list(range(len(curr)))
    iou_cost  = np.array([[1.-bbox_iou(p['bbox'],c['bbox']) for c in curr] for p in prev], np.float32)
    dist_mat  = np.array([[float(np.linalg.norm(_centroid(p)-_centroid(c))) for c in curr] for p in prev], np.float32)
    # Where boxes don't overlap at all, replace cost with normalised centroid distance
    no_overlap = iou_cost >= 1.0
    combined   = iou_cost.copy()
    combined[no_overlap] = np.clip(dist_mat[no_overlap] / max_dist, 0, 1)
    rows, cols = linear_sum_assignment(combined)
    matches, matched = [], set()
    for r, c in zip(rows, cols):
        if iou_cost[r,c] < (1.-thresh) or dist_mat[r,c] < max_dist:
            matches.append((r, c)); matched.add(c)
    return matches, [j for j in range(len(curr)) if j not in matched]

class Tracker:
    def __init__(self, max_lost=15):
        self.tracks   = {}
        self.next_id  = 0
        self.max_lost = max_lost

    def _predict(self):
        tids, seqs = [], []
        for tid, t in self.tracks.items():
            if len(t['history']) >= SEQ_LEN:
                tids.append(tid); seqs.append(t['history'][-SEQ_LEN:])
        pred = {}
        if seqs:
            ps = lstm_model(np.array(seqs, np.float32), training=False).numpy()
            for tid, p in zip(tids, ps):
                cx = float(np.clip(p[0],0,1))*IMG_SIZE; cy = float(np.clip(p[1],0,1))*IMG_SIZE
                w  = float(np.clip(p[2],0.01,1))*IMG_SIZE; h  = float(np.clip(p[3],0.01,1))*IMG_SIZE
                pred[tid] = (int(cx-w/2), int(cy-h/2), int(w), int(h))
        return pred

    def update(self, dets, fi):
        pred   = self._predict()
        active = sorted(self.tracks.keys())
        prev   = []
        for tid in active:
            d = dict(self.tracks[tid]['last_det'])
            if tid in pred: d['bbox'] = pred[tid]
            prev.append(d)
        matches, unmatched = match_dets(prev, dets)
        matched_prev = {p for p,_ in matches}
        for p_idx, c_idx in matches:
            tid = active[p_idx]; det = dets[c_idx]
            cx, cy = det['centroid']; x,y,w,h = det['bbox']
            self.tracks[tid].update({'last_det':det,'lost':0,'age':self.tracks[tid]['age']+1})
            self.tracks[tid]['history'].append([cx/IMG_SIZE,cy/IMG_SIZE,w/IMG_SIZE,h/IMG_SIZE])
        for i, tid in enumerate(active):
            if i not in matched_prev: self.tracks[tid]['lost'] += 1
        self.tracks = {tid:t for tid,t in self.tracks.items() if t['lost']<=self.max_lost}
        for c_idx in unmatched:
            det=dets[c_idx]; cx,cy=det['centroid']; x,y,w,h=det['bbox']
            self.tracks[self.next_id]={'last_det':det,'lost':0,'age':1,
                'history':[[cx/IMG_SIZE,cy/IMG_SIZE,w/IMG_SIZE,h/IMG_SIZE]]}
            self.next_id += 1
        records = []
        for tid, t in self.tracks.items():
            if t['lost'] == 0:
                det=t['last_det']; x,y,w,h=det['bbox']; cx,cy=det['centroid']
                records.append({'track_id':tid,'cx':round(cx,1),'cy':round(cy,1),
                    'x':x,'y':y,'w':w,'h':h,'area':det['area'],'age':t['age']})
        return records

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_frame(bgr, records, top_y, bot_y):
    vis = bgr.copy()
    h0, w0 = bgr.shape[:2]
    sx, sy  = w0 / IMG_SIZE, h0 / IMG_SIZE
    if top_y is not None:
        # Only draw where fascia was actually detected (non-NaN) — stays inside scan area
        xs_valid = [x for x in range(IMG_SIZE) if not np.isnan(top_y[x])]
        if len(xs_valid) > 1:
            pts_top = np.array([[int(x*sx), int(top_y[x]*sy)] for x in xs_valid], np.int32)
            pts_bot = np.array([[int(x*sx), int(bot_y[x]*sy)] for x in xs_valid], np.int32)
            cv2.polylines(vis, [pts_top], False, (0, 255, 255), 2)
            cv2.polylines(vis, [pts_bot], False, (0, 255, 255), 2)
    for r in records:
        col = LABEL_COLOR.get(r.get('label', 'unknown'), (150, 150, 150))
        x   = int(r['x'] * sx)
        y   = int(r['y'] * sy)
        w   = int(r['w'] * sx)
        h   = int(r['h'] * sy)
        cv2.rectangle(vis, (x, y), (x+w, y+h), col, 2)
        tag = f"T{r['track_id']} {r.get('label', '?')}"
        cv2.putText(vis, tag, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return vis

def frame_to_b64(bgr):
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()

# ── API Routes ────────────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files.get('video')
    if not f:
        return jsonify({'error': 'No file'}), 400
    uid = str(uuid.uuid4())
    ext = Path(f.filename).suffix or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name)
    _store[uid] = {'tmp_path': tmp.name, 'records': []}
    return jsonify({'upload_id': uid})


# ── Detection strategies ──────────────────────────────────────────────────────

def _dl_fascia_fn(frame, scan_xl_m, scan_xr_m):
    fascia_mask = predict_fascia(frame)
    fascia_mask[:, :scan_xl_m] = 0
    fascia_mask[:, scan_xr_m:] = 0
    return get_fascia_boundary(fascia_mask)

def _dl_vein_fn(frame, s_bot, min_area):
    vein_mask = predict_vein(frame)
    if s_bot is not None:
        for x in range(IMG_SIZE):
            if not np.isnan(s_bot[x]):
                cutoff = min(IMG_SIZE, int(s_bot[x]) + NEAR_PX + 1)
                vein_mask[cutoff:, x] = 0
    return extract_detections(vein_mask, min_area)

def _foundation_fascia_fn(frame, scan_xl_m, scan_xr_m):
    _pipe_b.ensure_loaded()
    return _pipe_b.detect_fascia(frame, scan_xl_m, scan_xr_m)

def _foundation_vein_fn(frame, s_bot, min_area):
    _pipe_b.ensure_loaded()
    return _pipe_b.detect_veins(frame, min_area)


def _make_generate(upload_id, target_fps, min_area, use_vlm, groq_key, vlm_model_nm,
                   fascia_fn, vein_fn):
    """
    fascia_fn(frame, scan_xl_m, scan_xr_m) -> (top_y, bot_y) or (None, None)
    vein_fn(frame, s_bot, min_area)        -> list[det]
    """
    groq_client = _make_groq(groq_key) if use_vlm else None

    tmp_path = _store[upload_id]['tmp_path']

    def generate():
        cap     = cv2.VideoCapture(tmp_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip    = max(1, round(src_fps / target_fps))
        n_proc  = max(1, total // skip)

        tracker        = Tracker()
        all_recs       = []
        counts         = {'N1': 0, 'N2': 0, 'N3': 0, 'unknown': 0}
        track_labels   = {}
        known_tids     = set()
        last_centroids = {}
        ghost_labels   = []
        scan_xl_m      = None
        scan_xr_m      = None
        fascia_buf     = []
        FASCIA_SMOOTH  = 7
        raw_idx        = 0
        proc_idx       = 0

        yield f"data: {json.dumps({'type':'meta','total':n_proc,'src_fps':src_fps,'skip':skip})}\n\n"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % skip == 0:
                h0, w0 = frame.shape[:2]

                if scan_xl_m is None:
                    xl_px, xr_px = get_scan_x_limits(frame)
                    scan_xl_m = max(0, round(xl_px * IMG_SIZE / w0))
                    scan_xr_m = min(IMG_SIZE, round(xr_px * IMG_SIZE / w0))

                top_y, bot_y = fascia_fn(frame, scan_xl_m, scan_xr_m)

                if top_y is not None:
                    fascia_buf.append((top_y, bot_y))
                    if len(fascia_buf) > FASCIA_SMOOTH:
                        fascia_buf.pop(0)
                if fascia_buf:
                    s_top = np.nanmedian(np.stack([f[0] for f in fascia_buf]), axis=0).astype(np.float32)
                    s_bot = np.nanmedian(np.stack([f[1] for f in fascia_buf]), axis=0).astype(np.float32)
                else:
                    s_top = s_bot = None

                dets = vein_fn(frame, s_bot, min_area)

                track_recs   = tracker.update(dets, proc_idx)
                current_tids = {r['track_id'] for r in track_recs}

                for tid in known_tids - current_tids:
                    lbl = track_labels.get(tid, 'unknown')
                    if lbl != 'unknown' and tid in last_centroids:
                        ghost_labels.append((*last_centroids[tid], lbl))
                if len(ghost_labels) > 40:
                    ghost_labels = ghost_labels[-40:]

                for r in track_recs:
                    if s_top is not None and r['age'] >= 2:
                        geo = geometric_classify(r['cx'], r['cy'], s_top, s_bot)
                        if geo != 'unknown':
                            track_labels[r['track_id']] = geo
                    elif r['track_id'] not in track_labels:
                        for gcx, gcy, glbl in reversed(ghost_labels):
                            if abs(r['cx'] - gcx) < 60 and abs(r['cy'] - gcy) < 60:
                                track_labels[r['track_id']] = glbl; break

                known_tids     = current_tids
                last_centroids = {r['track_id']: (r['cx'], r['cy']) for r in track_recs}

                unclassified = [r for r in track_recs
                                if track_labels.get(r['track_id'], 'unknown') == 'unknown']
                if use_vlm and groq_client and unclassified:
                    raw_vlm = vlm_classify(frame, unclassified, s_top, s_bot, groq_client, vlm_model_nm)
                    for idx, label in raw_vlm.items():
                        if 1 <= idx <= len(unclassified):
                            track_labels[unclassified[idx - 1]['track_id']] = label

                for r in track_recs:
                    r['label']     = track_labels.get(r['track_id'], 'unknown')
                    r['frame_idx'] = proc_idx
                    counts[r['label']] = counts.get(r['label'], 0) + 1
                    all_recs.append(r)

                annotated = draw_frame(frame, track_recs, s_top, s_bot)
                b64 = frame_to_b64(annotated)

                payload = {
                    'type':      'frame',
                    'frame_idx': proc_idx,
                    'total':     n_proc,
                    'image':     b64,
                    'counts':    counts.copy(),
                    'dets':      len(track_recs),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                proc_idx += 1

            raw_idx += 1

        cap.release()
        _store[upload_id]['records'] = all_recs
        yield f"data: {json.dumps({'type':'done','counts':counts,'total_processed':proc_idx})}\n\n"

    return generate


def _process_view(upload_id, fascia_fn, vein_fn):
    if upload_id not in _store:
        return jsonify({'error': 'Unknown upload_id'}), 404
    target_fps   = float(request.args.get('fps', 5))
    min_area     = int(request.args.get('min_area', 150))
    use_vlm      = request.args.get('use_vlm', 'true').lower() == 'true'
    groq_key     = request.args.get('groq_key', '')
    vlm_model_nm = request.args.get('vlm_model', 'meta-llama/llama-4-scout-17b-16e-instruct')
    generate = _make_generate(upload_id, target_fps, min_area, use_vlm, groq_key, vlm_model_nm,
                              fascia_fn, vein_fn)
    return Response(generate(),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/process/dl/<upload_id>', methods=['GET'])
def process_dl(upload_id):
    return _process_view(upload_id, _dl_fascia_fn, _dl_vein_fn)


VLM_DETECT_INTERVAL = 1   # run VLM on every processed frame — best recall


def _make_generate_foundation(upload_id, target_fps, min_area, use_vlm, groq_key, vlm_model_nm):
    """
    Foundation Pipeline generate loop.

    Every processed frame the VLM (EchoVLM or Groq) receives the current frame
    alongside few-shot reference images and returns:
      • fascia_y  — row of fascia centre in 256-px space
      • veins     — bounding boxes + N1/N2/N3 labels
    SAM refines the VLM fascia_y into a smooth per-column curved band.
    When VLM finds nothing, GDINO+SAM runs as fallback.
    """
    try:
        groq_client = _make_groq(groq_key) if (groq_key or DEFAULT_GROQ_KEY) else None
    except Exception as e:
        print(f"Groq unavailable for foundation pipeline: {e}")
        groq_client = None

    _pipe_b.ensure_loaded()
    tmp_path = _store[upload_id]['tmp_path']

    def generate():
        cap     = cv2.VideoCapture(tmp_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip    = max(1, round(src_fps / target_fps))
        n_proc  = max(1, total // skip)

        tracker        = Tracker()
        all_recs       = []
        counts         = {'N1': 0, 'N2': 0, 'N3': 0, 'unknown': 0}
        track_labels   = {}
        known_tids     = set()
        last_centroids = {}
        ghost_labels   = []
        FASCIA_SMOOTH  = 7
        raw_idx        = 0
        proc_idx       = 0
        scan_xl_m      = None
        scan_xr_m      = None

        vlm_vein_dets  = []  # most-recent vein detections from VLM (have 'vlm_label')
        vlm_fascia_buf = []  # smoothing buffer: VLM fascia_y refined through SAM
        gdino_fascia_buf = []  # smoothing buffer: GDINO+SAM fascia (fallback when VLM misses)

        yield f"data: {json.dumps({'type':'meta','total':n_proc,'src_fps':src_fps,'skip':skip})}\n\n"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % skip == 0:
                w0 = frame.shape[1]

                if scan_xl_m is None:
                    xl_px, xr_px = get_scan_x_limits(frame)
                    scan_xl_m = max(0, round(xl_px * IMG_SIZE / w0))
                    scan_xr_m = min(IMG_SIZE, round(xr_px * IMG_SIZE / w0))

                # ── Combined VLM detection ────────────────────────────────────────
                # EchoVLM (ultrasound-specific, local) takes priority over Groq.
                # Falls back to Groq when EchoVLM is unavailable (no GPU / not installed).
                vlm_available = _pipe_b._echovlm is not None or groq_client is not None
                if vlm_available and (proc_idx % VLM_DETECT_INTERVAL == 0):
                    if _pipe_b._echovlm is not None:
                        print(f"[EchoVLM] frame {proc_idx}: running ultrasound-specific VLM")
                        fy, vd = _pipe_b.vlm_detect_classify_echo(frame)
                    else:
                        print(f"[Groq VLM] frame {proc_idx}: few-shot detect "
                              f"({len(_pipe_b._few_shot_b64)} ref images)")
                        fy, vd = _pipe_b.vlm_detect_classify(frame, groq_client, vlm_model_nm)
                    print(f"[Foundation VLM] fascia_y={fy}, veins={len(vd)}")
                    if fy is not None:
                        t, b = _pipe_b._fascia_from_sam(frame, fy,
                                                        scan_xl_m=scan_xl_m,
                                                        scan_xr_m=scan_xr_m)
                        if t is None:
                            # SAM unavailable or mask too sparse — draw flat line at VLM's y
                            half = 8
                            x_lo_f = max(0, scan_xl_m)
                            x_hi_f = min(IMG_SIZE, scan_xr_m)
                            fy_c = int(np.clip(fy, half, IMG_SIZE - 1 - half))
                            t = np.full(IMG_SIZE, np.nan, np.float32)
                            b = np.full(IMG_SIZE, np.nan, np.float32)
                            t[x_lo_f:x_hi_f] = float(fy_c - half)
                            b[x_lo_f:x_hi_f] = float(fy_c + half)
                            print(f"[Foundation] SAM unavailable — flat fascia line at y={fy_c}")
                        vlm_fascia_buf.append((t, b))
                        if len(vlm_fascia_buf) > FASCIA_SMOOTH:
                            vlm_fascia_buf.pop(0)
                    if vd:
                        vlm_vein_dets = vd

                # ── Fascia: VLM buffer takes priority; brightness is independent fallback ──
                # Keeping VLM and brightness results in separate buffers prevents them
                # from polluting each other's median when they disagree on fascia position.
                if vlm_fascia_buf:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        s_top = np.nanmedian(np.stack([f[0] for f in vlm_fascia_buf]), axis=0).astype(np.float32)
                        s_bot = np.nanmedian(np.stack([f[1] for f in vlm_fascia_buf]), axis=0).astype(np.float32)
                    # Columns that are all-NaN (outside scan area) stay NaN — draw_frame skips them
                else:
                    top_y, bot_y = _pipe_b.detect_fascia(frame, scan_xl_m, scan_xr_m)
                    if top_y is not None:
                        gdino_fascia_buf.append((top_y, bot_y))
                        if len(gdino_fascia_buf) > FASCIA_SMOOTH:
                            gdino_fascia_buf.pop(0)
                    if gdino_fascia_buf:
                        s_top = np.nanmedian(np.stack([f[0] for f in gdino_fascia_buf]), axis=0).astype(np.float32)
                        s_bot = np.nanmedian(np.stack([f[1] for f in gdino_fascia_buf]), axis=0).astype(np.float32)
                    else:
                        s_top = s_bot = None

                # ── Veins: on VLM frames use VLM boxes; on all other frames run
                # fresh GDINO/SAM so the tracker never gets stale positions ──
                if vlm_available and (proc_idx % VLM_DETECT_INTERVAL == 0) and vlm_vein_dets:
                    dets = vlm_vein_dets
                else:
                    dets = _pipe_b.detect_veins(frame, min_area)

                # Remove any detection whose centroid is below the fascia band —
                # same anatomical constraint as the DL pipeline.
                if s_bot is not None:
                    dets = [d for d in dets
                            if np.isnan(s_bot[int(np.clip(d['centroid'][0], 0, IMG_SIZE - 1))])
                            or d['centroid'][1] <= (
                                s_bot[int(np.clip(d['centroid'][0], 0, IMG_SIZE - 1))] + NEAR_PX)]

                track_recs   = tracker.update(dets, proc_idx)
                current_tids = {r['track_id'] for r in track_recs}

                for tid in known_tids - current_tids:
                    lbl = track_labels.get(tid, 'unknown')
                    if lbl != 'unknown' and tid in last_centroids:
                        ghost_labels.append((*last_centroids[tid], lbl))
                if len(ghost_labels) > 40:
                    ghost_labels = ghost_labels[-40:]

                # Classify: geometric > VLM-embedded label > ghost label
                for r in track_recs:
                    if s_top is not None and r['age'] >= 2:
                        geo = geometric_classify(r['cx'], r['cy'], s_top, s_bot)
                        if geo != 'unknown':
                            track_labels[r['track_id']] = geo
                    elif r['track_id'] not in track_labels:
                        # Use the VLM label that arrived with the detection
                        matched = next(
                            (d for d in dets
                             if 'vlm_label' in d
                             and abs(d['centroid'][0] - r['cx']) < 20
                             and abs(d['centroid'][1] - r['cy']) < 20),
                            None
                        )
                        if matched:
                            track_labels[r['track_id']] = matched['vlm_label']
                        else:
                            for gcx, gcy, glbl in reversed(ghost_labels):
                                if abs(r['cx'] - gcx) < 60 and abs(r['cy'] - gcy) < 60:
                                    track_labels[r['track_id']] = glbl; break

                known_tids     = current_tids
                last_centroids = {r['track_id']: (r['cx'], r['cy']) for r in track_recs}

                # Standalone VLM classification for any still-unknown tracks
                # Only falls back to Groq here — EchoVLM classification happens
                # inside vlm_detect_classify_echo via the vlm_label field.
                unclassified = [r for r in track_recs
                                if track_labels.get(r['track_id'], 'unknown') == 'unknown']
                if use_vlm and groq_client and unclassified:
                    raw_vlm = vlm_classify(frame, unclassified, s_top, s_bot, groq_client, vlm_model_nm)
                    for idx, label in raw_vlm.items():
                        if 1 <= idx <= len(unclassified):
                            track_labels[unclassified[idx - 1]['track_id']] = label

                for r in track_recs:
                    r['label']     = track_labels.get(r['track_id'], 'unknown')
                    r['frame_idx'] = proc_idx
                    counts[r['label']] = counts.get(r['label'], 0) + 1
                    all_recs.append(r)

                annotated = draw_frame(frame, track_recs, s_top, s_bot)
                b64 = frame_to_b64(annotated)

                payload = {
                    'type':      'frame',
                    'frame_idx': proc_idx,
                    'total':     n_proc,
                    'image':     b64,
                    'counts':    counts.copy(),
                    'dets':      len(track_recs),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                proc_idx += 1

            raw_idx += 1

        cap.release()
        _store[upload_id]['records'] = all_recs
        yield f"data: {json.dumps({'type':'done','counts':counts,'total_processed':proc_idx})}\n\n"

    return generate


@app.route('/api/process/foundation/<upload_id>', methods=['GET'])
def process_foundation(upload_id):
    if upload_id not in _store:
        return jsonify({'error': 'Unknown upload_id'}), 404
    target_fps   = float(request.args.get('fps', 5))
    min_area     = int(request.args.get('min_area', 150))
    use_vlm      = request.args.get('use_vlm', 'true').lower() == 'true'
    groq_key     = request.args.get('groq_key', '')
    vlm_model_nm = request.args.get('vlm_model', 'meta-llama/llama-4-scout-17b-16e-instruct')
    generate = _make_generate_foundation(upload_id, target_fps, min_area, use_vlm, groq_key, vlm_model_nm)
    return Response(generate(),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/process/<upload_id>', methods=['GET'])
def process(upload_id):
    return _process_view(upload_id, _dl_fascia_fn, _dl_vein_fn)


@app.route('/api/report/<upload_id>', methods=['GET'])
def download_report(upload_id):
    if upload_id not in _store:
        return jsonify({'error': 'Not found'}), 404
    records = _store[upload_id].get('records', [])
    if not records:
        return jsonify({'error': 'No records yet'}), 400
    df   = pd.DataFrame(records)
    cols = ['frame_idx','track_id','label','cx','cy','x','y','w','h','area','age']
    df   = df[[c for c in cols if c in df.columns]]
    buf  = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True,
                     download_name=f'vein_report_{upload_id[:8]}.csv')

if __name__ == '__main__':
    print(f"==> App running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)