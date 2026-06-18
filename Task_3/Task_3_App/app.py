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
import threading, queue, warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
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
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
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
    "  1. The fascia band — the first prominent bright (hyperechoic) near-horizontal line "
    "scanning from the top of the image downward. It separates subcutaneous fat (above, "
    "speckled/grainy) from muscle (below, darker striated texture). "
    "IMPORTANT: fascia_y can be as small as 5 when the probe is over a knee or ankle. "
    "Do NOT require a fat layer above — a bright line with striated muscle below is the fascia "
    "even if it sits near row 0. Return null ONLY if no boundary line is visible anywhere.\n"
    "  2. Any veins — dark anechoic or hypoechoic oval/circular structures.\n"
    "Classify each vein:\n"
    "  N3 = above fascia (superficial vein)\n"
    "  N2 = at or very near the fascia (Great Saphenous Vein)\n"
    "  N1 = below fascia (deep vein)\n"
    "Reply ONLY with valid JSON, no extra text:\n"
    "{\"fascia_y\":<integer 0-255 for the fascia centre row, or null if truly not visible>,"
    "\"veins\":[{\"x\":<int>,\"y\":<int>,\"w\":<int>,\"h\":<int>,\"label\":\"N2\"}]}"
)

FOUNDATION_FEW_SHOT_PROMPT = (
    "Analyse this new B-mode ultrasound frame (256x256 px, origin top-left).\n\n"
    "STEP 1 — FASCIA: Locate the first prominent BRIGHT (hyperechoic) near-horizontal line scanning "
    "from the top downward. It separates speckled fat above from darker striated muscle below. "
    "fascia_y can be very small (even < 20) for knee/ankle scans — do NOT require a fat layer above. "
    "Return null ONLY if no bright boundary line exists anywhere.\n\n"
    "STEP 2 — VEINS: Veins in B-mode ultrasound appear as NEARLY BLACK (anechoic), ROUND or OVAL "
    "cross-sections with a thin bright echogenic wall/border. They are the most visually prominent "
    "DARK CIRCULAR spots in the image, clearly darker than the surrounding grey muscle tissue. "
    "Key identification rules:\n"
    "  • Interior is almost black — much darker than any surrounding tissue\n"
    "  • Shape is round or oval (not irregular or speckled)\n"
    "  • Thin bright rim/border visible at the edge\n"
    "  • Minimum size ~15px wide — ignore tiny speckle noise\n"
    "  • Found below or at the fascia line (rarely above)\n\n"
    "Label each vein by where its centre falls relative to fascia_y:\n"
    "  N1 = centre BELOW fascia_y (deep vein, inside muscle)\n"
    "  N2 = centre WITHIN ±15 px of fascia_y (GSV, at the fascia)\n"
    "  N3 = centre ABOVE fascia_y (superficial vein, in fat)\n\n"
    "IMPORTANT: Draw bounding boxes tightly around the ENTIRE dark oval, including the bright wall. "
    "Only report clearly visible round/oval black structures — do NOT report artifacts or noise.\n\n"
    "Reply ONLY with valid JSON, no markdown, no extra text:\n"
    "{\"fascia_y\":<int 0-255 or null>,\"veins\":[{\"x\":<int>,\"y\":<int>,\"w\":<int>,\"h\":<int>,\"label\":\"N1|N2|N3\"}]}"
)

VEIN_ONLY_PROMPT = (
    "This is a B-mode ultrasound image (256×256 pixels, origin top-left).\n\n"
    "Your ONLY task: find and localize every BLOOD VESSEL / VEIN visible in this image.\n\n"
    "Veins appear in two ways depending on scan orientation:\n"
    "  TRANSVERSE (cross-section): round or circular dark spots — like a black circle\n"
    "  LONGITUDINAL (along the vessel): elongated dark oval or tube shape — wider than tall\n\n"
    "In BOTH cases the key feature is:\n"
    "  • DARK (nearly black, anechoic) interior — clearly much darker than surrounding grey tissue\n"
    "  • Thin BRIGHT (hyperechoic) echogenic wall/border around the dark interior\n"
    "  • Smooth well-defined edges\n"
    "  • Size: 10–150 pixels in this 256px image\n\n"
    "Draw a tight bounding box around each dark vessel (include the bright wall).\n"
    "Ignore: speckle noise, bright fascia lines, muscle striations.\n\n"
    "Reply ONLY with valid JSON — no markdown:\n"
    "{\"veins\":[{\"x\":<left edge>,\"y\":<top edge>,\"w\":<width>,\"h\":<height>}]}\n"
    "If no vessels visible: {\"veins\":[]}"
)

# Retry prompt when VLM returns fascia_y=null — asks it to look specifically at the top rows
FOUNDATION_TOP_RETRY_PROMPT = (
    "Look again at this ultrasound image. The fascia may be very close to the top of the image "
    "(rows 5–50). Scan specifically the top 50 rows for a bright hyperechoic horizontal line. "
    "If you find one, return its row as fascia_y. If the entire image is striated muscle texture "
    "with no clear upper boundary, set fascia_y to the topmost row where striated texture begins.\n"
    "Reply ONLY with valid JSON: {\"fascia_y\":<int 0-255 or null>}"
)

# ── Hardcoded API keys ────────────────────────────────────────────────────────
DEFAULT_GROQ_KEY = ""
HF_TOKEN         = ""

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

def nms(dets, iou_thresh=0.3, contain_thresh=0.7):
    if len(dets) <= 1:
        return dets
    dets_sorted = sorted(dets, key=lambda d: d['area'], reverse=True)
    kept = []
    for d in dets_sorted:
        x1, y1, w1, h1 = d['bbox']
        suppress = False
        for k in kept:
            xk, yk, wk, hk = k['bbox']
            ix = max(0, min(x1+w1, xk+wk) - max(x1, xk))
            iy = max(0, min(y1+h1, yk+hk) - max(y1, yk))
            inter = ix * iy
            iou = inter / (w1*h1 + wk*hk - inter + 1e-6)
            # Also suppress if this box is mostly contained within a kept (larger) box
            containment = inter / (w1*h1 + 1e-6)
            if iou >= iou_thresh or containment >= contain_thresh:
                suppress = True
                break
        if not suppress:
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
        self._echovlm_queue    = queue.Queue(maxsize=1)   # latest frame to infer
        self._echovlm_cache    = (None, [])               # last (fascia_y, dets) result
        self._echovlm_lock     = threading.Lock()
        self._few_shot_b64     = []
        self._vlm_context_msgs = []
        self._vein_ctx_b64     = []
        self._vein_ctx_msgs    = []
        self._tried            = False
        self._load_vein_few_shot()

    def _load_vein_few_shot(self):
        """Build vein-only annotated reference images for the vein test endpoint."""
        self._vein_ctx_b64  = []
        self._vein_ctx_msgs = []

        output_dir = DATA_ROOT / 'output'
        frames_dir = output_dir / 'fascia' / 'frames'
        vein_csv   = output_dir / 'classification' / 'vein_classified_vlm.csv'

        if not vein_csv.exists() or not frames_dir.exists():
            print("PipelineB: vein few-shot data not found, skipping.")
            return

        try:
            import csv as _csv

            # Group vein annotations by (video, frame_idx)
            vein_rows = {}
            with open(vein_csv, newline='') as f:
                for row in _csv.DictReader(f):
                    key = (row['video'], int(row['frame_idx']))
                    vein_rows.setdefault(key, []).append(row)

            # Bucket frames by max single-vein area → small / medium / large
            def _max_area(rows):
                return max(float(r.get('w', 1)) * float(r.get('h', 1)) for r in rows)

            buckets = {'small': [], 'medium': [], 'large': []}
            for key, rows in vein_rows.items():
                a = _max_area(rows)
                if a < 500:
                    buckets['small'].append((key, rows, a))
                elif a < 2500:
                    buckets['medium'].append((key, rows, a))
                else:
                    buckets['large'].append((key, rows, a))

            # From each bucket pick up to 4-5 frames, favouring different videos
            def _diverse(bucket, n):
                bucket.sort(key=lambda t: t[2], reverse=True)
                seen, out = set(), []
                for key, rows, _ in bucket:
                    if key[0] not in seen or len(out) < n // 2:
                        if (frames_dir / key[0] / f'{key[1]:05d}.png').exists():
                            out.append((key, rows))
                            seen.add(key[0])
                    if len(out) >= n:
                        break
                return out

            # llama-4-scout supports max 5 images total → 4 reference + 1 query
            # Pick 2 large, 1 medium, 1 small for maximum diversity within the limit
            selected = (_diverse(buckets['large'],  2) +
                        _diverse(buckets['medium'], 1) +
                        _diverse(buckets['small'],  1))

            # Fallback: if buckets are sparse just take the top-area frames overall
            if len(selected) < 2:
                all_sorted = sorted(vein_rows.items(),
                                    key=lambda kv: _max_area(kv[1]), reverse=True)
                selected = [(k, r) for k, r in all_sorted
                            if (frames_dir / k[0] / f'{k[1]:05d}.png').exists()][:4]

            for (video, fidx), rows in selected:
                frame_path = frames_dir / video / f'{fidx:05d}.png'
                frm = cv2.imread(str(frame_path))
                if frm is None:
                    continue
                frm = cv2.resize(frm, (IMG_SIZE, IMG_SIZE))
                for r in rows:
                    x = int(float(r['x'])); y = int(float(r['y']))
                    w = max(4, int(float(r['w']))); h = max(4, int(float(r['h'])))
                    cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    sz = f"{w}x{h}"
                    cv2.putText(frm, sz, (x, max(0, y - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
                _, buf = cv2.imencode('.jpg', frm, [cv2.IMWRITE_JPEG_QUALITY, 88])
                self._vein_ctx_b64.append(base64.b64encode(buf.tobytes()).decode())
                if len(self._vein_ctx_b64) >= 4:   # hard cap: 4 ref + 1 query = 5 total
                    break

        except Exception as e:
            print(f"PipelineB: vein few-shot build failed ({e})")

        if not self._vein_ctx_b64:
            print("PipelineB: no vein reference frames built.")
            return

        ref_content = [{"type": "text", "text": (
            "These are REAL B-mode ultrasound frames from the same scanner with GROUND-TRUTH annotations.\n"
            "CYAN boxes = confirmed blood vessel cross-sections. The size label (e.g. '14x10') shows "
            "the actual pixel dimensions of each vein box.\n\n"
            "Key visual features of veins in these scans:\n"
            "  • Nearly BLACK interior (anechoic — much darker than surrounding grey tissue)\n"
            "  • Round or oval cross-section with a thin bright echogenic wall/border\n"
            "  • Sizes range from ~10px to ~100px in diameter\n"
            "  • Small veins look like tiny dark dots; large ones like clear black circles\n"
            "  • Found anywhere from just below the skin surface to deep in muscle\n\n"
            "Study all annotated examples carefully — note the wide range of sizes and positions."
        )}]
        for b64 in self._vein_ctx_b64:
            ref_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        self._vein_ctx_msgs = [
            {"role": "user", "content": ref_content},
            {"role": "assistant", "content": (
                "I have studied all the annotated reference frames carefully. I can see vessels of "
                "various sizes — from small dark dots (~10-20px) to large black circles (~60-100px) "
                "— all marked with cyan boxes. I will now detect every similar dark anechoic oval "
                "in new frames, including small ones, and output tight bounding boxes."
            )},
        ]
        print(f"PipelineB: vein few-shot context built — {len(self._vein_ctx_b64)} frames "
              f"({len(buckets['large'])} large / {len(buckets['medium'])} medium / "
              f"{len(buckets['small'])} small available).")

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
        Load EchoVLM from local weights at Task_3/EchoVLM/.
        Requires CUDA PyTorch — falls back to Groq VLM if unavailable.
        """
        try:
            import sys, torch
            from transformers import AutoProcessor

            # Local weights placed by user at Task_3/EchoVLM/
            weights_dir = HERE.parent / 'EchoVLM'
            if not weights_dir.exists():
                print(f"PipelineB: EchoVLM weights not found at {weights_dir}; Groq VLM active.")
                self._echovlm = None
                return

            # Patch ROPE_INIT_FUNCTIONS — transformers 5.x removed "default"
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if "default" not in ROPE_INIT_FUNCTIONS:
                def _default_rope_init(config, device=None, **kwargs):
                    dim  = kwargs.get('dim') or (config.hidden_size // config.num_attention_heads)
                    base = float(kwargs.get('base') or getattr(config, 'rope_theta', 1000000.0))
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                    return inv_freq, 1.0
                ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

            # Code repo already cloned to vendor/EchoVLM
            echovlm_dir = HERE / 'vendor' / 'EchoVLM'
            if str(echovlm_dir) not in sys.path:
                sys.path.insert(0, str(echovlm_dir))

            from EchoVLM import Qwen2VLMOEForConditionalGeneration

            dev   = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            attn  = "sdpa" if torch.cuda.is_available() else "eager"

            print(f"PipelineB: loading EchoVLM weights from {weights_dir} ...")
            model = Qwen2VLMOEForConditionalGeneration.from_pretrained(
                str(weights_dir),
                torch_dtype=dtype,
                attn_implementation=attn,
                device_map="auto",
            )
            model.config.output_router_logits = False
            proc = AutoProcessor.from_pretrained(str(weights_dir))
            self._echovlm = (model, proc)
            print(f"PipelineB: EchoVLM ready on {dev}.")
            t = threading.Thread(target=self._echovlm_worker, daemon=True)
            t.start()
            print("PipelineB: EchoVLM background inference thread started.")
        except Exception as e:
            print(f"PipelineB: EchoVLM unavailable ({e}); Groq VLM fallback active.")
            self._echovlm = None

    def _echovlm_worker(self):
        """Background thread: consumes frames from _echovlm_queue, updates _echovlm_cache."""
        while True:
            bgr = self._echovlm_queue.get()  # blocks until a frame arrives
            if bgr is None:
                break
            try:
                print(f"[EchoVLM worker] starting inference...")
                fy, vd = self.vlm_detect_classify_echo(bgr)
                with self._echovlm_lock:
                    self._echovlm_cache = (fy, vd)
                print(f"[EchoVLM worker] done: fascia_y={fy}, veins={len(vd)}")
            except Exception as e:
                print(f"[EchoVLM worker] error: {e}")
                import traceback; traceback.print_exc()

    def _load_few_shot(self):
        """
        Build ground-truth-annotated few-shot reference images from the training CSVs.
        Draws actual vein boxes (N1/N2/N3) and fascia mask onto saved frames so the
        VLM sees real examples rather than raw unannotated images.
        """
        self._few_shot_b64     = []
        self._vlm_context_msgs = []

        output_dir  = DATA_ROOT / 'output'
        frames_dir  = output_dir / 'fascia' / 'frames'
        masks_dir   = output_dir / 'fascia' / 'masks'
        vein_csv    = output_dir / 'classification' / 'vein_classified_vlm.csv'

        if not vein_csv.exists() or not frames_dir.exists():
            print(f"PipelineB: ground-truth data not found, skipping few-shot context.")
            return

        try:
            import csv as _csv
            # Load fascia frames that have valid masks
            fascia_meta_csv = output_dir / 'fascia' / 'metadata.csv'
            fascia_frames = set()
            if fascia_meta_csv.exists():
                with open(fascia_meta_csv, newline='') as f:
                    for row in _csv.DictReader(f):
                        if row.get('has_fascia', '').lower() == 'true' and row.get('valid_mask', '').lower() == 'true':
                            fascia_frames.add((row['video'], int(row['frame_idx'])))

            # Load vein annotations grouped by (video, frame_idx)
            vein_rows = {}
            with open(vein_csv, newline='') as f:
                for row in _csv.DictReader(f):
                    key = (row['video'], int(row['frame_idx']))
                    vein_rows.setdefault(key, []).append(row)

            # Score frames: strongly prefer frames that have both fascia AND veins with mixed labels
            def _score(key, rows):
                labels = {r['label'] for r in rows}
                has_fascia_bonus = 50 if key in fascia_frames else 0
                return has_fascia_bonus + len(labels) * 10 + len(rows)

            scored = sorted(vein_rows.items(), key=lambda kv: _score(kv[0], kv[1]), reverse=True)

            label_color = {'N1': (220, 0, 0), 'N2': (0, 200, 50), 'N3': (0, 140, 255)}

            for (video, fidx), rows in scored:
                frame_path = frames_dir / video / f'{fidx:05d}.png'
                if not frame_path.exists():
                    continue

                frm = cv2.imread(str(frame_path))
                if frm is None:
                    continue
                frm = cv2.resize(frm, (IMG_SIZE, IMG_SIZE))

                # Draw fascia mask if available
                mask_path = masks_dir / video / f'{fidx:05d}.png'
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                        ys = np.where(mask > 127)
                        if len(ys[0]):
                            top_y = int(ys[0].min())
                            bot_y = int(ys[0].max())
                            cv2.line(frm, (0, top_y), (IMG_SIZE, top_y), (0, 255, 255), 1)
                            cv2.line(frm, (0, bot_y), (IMG_SIZE, bot_y), (0, 255, 255), 1)

                # Draw ground-truth vein boxes
                for r in rows:
                    x, y = int(float(r['x'])), int(float(r['y']))
                    w, h = max(4, int(float(r['w']))), max(4, int(float(r['h'])))
                    lbl  = r.get('label', 'N2')
                    col  = label_color.get(lbl, (150, 150, 150))
                    cv2.rectangle(frm, (x, y), (x+w, y+h), col, 2)
                    cv2.putText(frm, lbl, (x, max(0, y-3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

                _, buf = cv2.imencode('.jpg', frm, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self._few_shot_b64.append(base64.b64encode(buf.tobytes()).decode())

                if len(self._few_shot_b64) >= 2:
                    break

        except Exception as e:
            print(f"PipelineB: failed to build annotated few-shot images ({e}); falling back.")

        if not self._few_shot_b64:
            print(f"PipelineB: no annotated reference frames built.")
            return

        ref_content = [{"type": "text", "text": (
            "These are ANNOTATED B-mode ultrasound frames from the same scanner.\n"
            "CYAN lines = fascia band (bright hyperechoic horizontal layer separating fat above from muscle below).\n"
            "Coloured boxes = veins (dark anechoic/hypoechoic oval structures):\n"
            "  RED box   = N1 (deep vein — BELOW fascia, inside muscle)\n"
            "  GREEN box = N2 (GSV — AT or right on the fascia line)\n"
            "  BLUE box  = N3 (superficial vein — ABOVE fascia, in fat layer)\n"
            "Note: veins vary greatly in size — some fill 10-40% of the image."
        )}]
        for b64 in self._few_shot_b64:
            ref_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        self._vlm_context_msgs = [
            {"role": "user", "content": ref_content},
            {"role": "assistant", "content": (
                "Understood. I can see the annotated reference frames with cyan fascia lines "
                "and colour-coded vein boxes (red=N1 deep, green=N2 GSV, blue=N3 superficial). "
                "I will use this anatomy to analyse new frames."
            )},
        ]
        print(f"PipelineB: VLM context pre-built with {len(self._few_shot_b64)} annotated reference images "
              f"({sum(len(b) for b in self._few_shot_b64) // 1024} KB encoded at startup).")

    def vlm_detect_classify_echo(self, bgr):
        """
        Combined fascia + vein detection + N1/N2/N3 classification via EchoVLM.
        Uses processor.apply_chat_template with tokenize=True per the HF model card.
        Returns (fascia_y_int_or_None, dets_list_with_vlm_label).
        """
        if self._echovlm is None:
            return None, []
        try:
            import torch
            from PIL import Image as PILImage

            model, proc = self._echovlm

            vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
            pil_img = PILImage.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": ECHOVLM_PROMPT},
            ]}]

            inputs = proc.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            inputs.pop('mm_token_type_ids', None)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=300)

            raw = proc.decode(
                generated_ids[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

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
        # CLAHE on luminance channel — makes bright fascia band and dark vein ovals more distinct
        lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        vis = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
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

            # If VLM returned null, retry once with a top-edge focused prompt
            if fascia_y is None:
                retry_msg = {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_b64}"}},
                        {"type": "text", "text": FOUNDATION_TOP_RETRY_PROMPT},
                    ],
                }
                try:
                    r2 = groq_client.chat.completions.create(
                        model=vlm_model_nm,
                        messages=[retry_msg],
                        max_tokens=60, temperature=0.0,
                    )
                    m2 = re.search(r'\{.*\}', r2.choices[0].message.content.strip(), re.DOTALL)
                    if m2:
                        v2 = json.loads(m2.group()).get('fascia_y')
                        if v2 is not None:
                            fascia_y = int(np.clip(int(v2), 0, IMG_SIZE - 1))
                            print(f"[VLM retry] fascia_y recovered: {fascia_y}")
                except Exception:
                    pass

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
        """Brightness gradient locates the fascia zone; per-column peak gives the exact curved band."""
        return self._fascia_from_brightness(bgr, scan_xl_m=scan_xl_m, scan_xr_m=scan_xr_m)

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

            # Pick the widest, flattest, shallowest box in the expected fascia zone (10–55% depth).
            # Penalise depth strongly so the deep fascia / muscle interface scores below the
            # superficial fascia even when it produces a slightly wider GDINO box.
            best_y, best_score = None, -1.0
            for x1, y1, x2, y2 in boxes:
                cy = (y1 + y2) / 2
                cy_norm = cy / h0
                if cy_norm < 0.10 or cy_norm > 0.55:
                    continue
                bw, bh = x2 - x1, max(y2 - y1, 1)
                score = (bw / w0) * (bw / bh) * (1.0 - cy_norm)  # prefer wide + flat + shallow
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

    def detect_veins_cv(self, bgr, max_area=4000):
        """
        Strict dark-blob detector. Two threshold passes (very dark only) to minimise
        false positives from speckle noise and muscle striation.
        """
        vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray_enh = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_enh, (7, 7), 0)

        dets = []
        seen_centers = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Only two passes at dark thresholds — the old pass at 80 captured muscle
        # striations and generated most of the false positives
        for thresh_val in (42, 60):
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh[:int(IMG_SIZE * 0.10), :] = 0   # skip probe coupling line

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 150 or area > max_area:   # was 55 — raised to reject speckle
                    continue
                x, y, w, h = cv2.boundingRect(cnt)

                # Minimum dimensions: raise to 12 to eliminate tiny noise blobs
                if w < 12 or h < 12:   # was 8
                    continue

                # Allow up to 5:1 for longitudinal veins
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > 5.0:
                    continue

                # Circularity — relaxed to handle longitudinal oval shapes
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                if 4 * np.pi * area / (perimeter * perimeter) < 0.28:
                    continue

                # Strict interior darkness — real vein lumens are near-black (~20-50).
                # Raising from 75 to 60 rejects muscle striation (mid-grey ~80-120).
                mask_cnt = np.zeros(gray_enh.shape, np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, cv2.FILLED)
                mean_inside = float(cv2.mean(gray_enh, mask=mask_cnt)[0])
                if mean_inside > 60:   # was 75 — fascia ~200, muscle ~100, vein ~20-55
                    continue

                cx_c, cy_c = x + w // 2, y + h // 2
                if any(abs(cx_c - sc[0]) < 16 and abs(cy_c - sc[1]) < 16
                       for sc in seen_centers):
                    continue
                seen_centers.append((cx_c, cy_c))

                dets.append({
                    'bbox':     (x, y, w, h),
                    'centroid': (float(cx_c), float(cy_c)),
                    'area':     int(area),
                })
        return nms(dets)

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

    def _gdino_detect(self, bgr, min_area, box_threshold=0.20, max_dim=None,
                      text=None, max_aspect=4, edge_skip_frac=0.08):
        try:
            import torch
            from PIL import Image as PILImage
            proc, model, dev = self._gdino
            h0, w0 = bgr.shape[:2]
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            inputs = proc(
                images=pil,
                text=text or "vein . dark oval blood vessel . anechoic structure",
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
                keep = results['scores'].cpu() >= box_threshold
                results = {'boxes': results['boxes'][keep]}
            except TypeError:
                results = proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=box_threshold, text_threshold=box_threshold * 0.75,
                    target_sizes=[(h0, w0)],
                )[0]

            edge_skip = int(IMG_SIZE * edge_skip_frac)
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
                # Reject full-scan false positives
                if area > IMG_SIZE * IMG_SIZE * 0.40:
                    continue
                # Reject scan-border artifacts
                if y < edge_skip:
                    continue
                # Reject extremely elongated shapes (caller sets max_aspect)
                if bh > 0 and bw > bh * max_aspect:
                    continue
                # Optional per-call max dimension filter
                if max_dim is not None and (bw > max_dim or bh > max_dim):
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
@app.route('/api/ping')
def ping():
    return jsonify({'ok': True, 'store_keys': list(_store.keys())})

@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files.get('video')
    if not f:
        return jsonify({'error': 'No file'}), 400
    uid = str(uuid.uuid4())
    ext = Path(f.filename).suffix or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name)
    tmp.close()  # release handle so OpenCV can open the file on Windows
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


def _make_generate_foundation(upload_id, target_fps):
    """
    Foundation Pipeline: GDINO+SAM for fascia (curved per-column boundary),
    GDINO with brightness filter for veins. N1/N2/N3 classification is purely
    geometric (position relative to fascia band). No VLM calls.
    """
    if _pipe_b._gdino is None:
        _pipe_b._load_gdino()
    if _pipe_b._sam is None:
        _pipe_b._load_sam()

    tmp_path = _store[upload_id]['tmp_path']

    GDINO_VEIN_TEXT = (
        "blood vessel . vein . dark anechoic oval . dark hypoechoic tube . "
        "dark elongated vessel . dark circular vessel . "
        "dark horizontal band . dark anechoic region . dark tube"
    )
    MAX_VEIN_AREA = IMG_SIZE * IMG_SIZE * 0.15

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
        fascia_buf     = []   # (top_y_arr, bot_y_arr) per processed frame
        last_vein_dets = []
        vein_prop_cnt  = 0
        MAX_PROP       = 4

        yield f"data: {json.dumps({'type':'meta','total':n_proc,'src_fps':src_fps,'skip':skip})}\n\n"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % skip == 0:
                w0  = frame.shape[1]
                vis = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

                if scan_xl_m is None:
                    xl_px, xr_px = get_scan_x_limits(frame)
                    scan_xl_m = max(0, round(xl_px * IMG_SIZE / w0))
                    scan_xr_m = min(IMG_SIZE, round(xr_px * IMG_SIZE / w0))

                # ── Fascia: GDINO → SAM ───────────────────────────────────────
                top_y_raw, bot_y_raw = _pipe_b.detect_fascia(vis, scan_xl_m, scan_xr_m)
                if top_y_raw is not None:
                    fascia_buf.append((top_y_raw, bot_y_raw))
                    if len(fascia_buf) > FASCIA_SMOOTH:
                        fascia_buf.pop(0)

                if fascia_buf:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        s_top = np.nanmedian(np.stack([f[0] for f in fascia_buf]), axis=0).astype(np.float32)
                        s_bot = np.nanmedian(np.stack([f[1] for f in fascia_buf]), axis=0).astype(np.float32)
                    # Extra smoothing to remove SAM per-column noise
                    valid = ~np.isnan(s_top)
                    if valid.sum() > 10:
                        xs_s = np.where(valid)[0].astype(np.float32)
                        x_lo_s, x_hi_s = int(xs_s.min()), int(xs_s.max())
                        rng_s = np.arange(x_lo_s, x_hi_s + 1, dtype=np.float32)
                        s_top[x_lo_s:x_hi_s+1] = uniform_filter1d(
                            np.interp(rng_s, xs_s, s_top[valid]), 100).astype(np.float32)
                        s_bot[x_lo_s:x_hi_s+1] = uniform_filter1d(
                            np.interp(rng_s, xs_s, s_bot[valid]), 100).astype(np.float32)
                else:
                    s_top = s_bot = None

                # ── Veins: GDINO with CLAHE + brightness filter ───────────────
                lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                vis_enh  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                gray_vis = cv2.cvtColor(vis_enh, cv2.COLOR_BGR2GRAY)

                dets_raw = _pipe_b._gdino_detect(
                    vis, min_area=60, box_threshold=0.06,
                    text=GDINO_VEIN_TEXT, max_aspect=15, edge_skip_frac=0.03,
                )
                dets = []
                for d in dets_raw:
                    x, y, w, h = [int(v) for v in d['bbox']]
                    if w * h > MAX_VEIN_AREA or y + h > IMG_SIZE - 6:
                        continue
                    margin = max(2, min(w, h) // 6)
                    roi = gray_vis[y+margin:y+h-margin, x+margin:x+w-margin]
                    if roi.size > 0 and 10 < float(roi.mean()) < 90:
                        cx_d = float(x + w / 2)
                        cy_d = float(y + h / 2)
                        dets.append({'bbox': (x, y, w, h), 'centroid': (cx_d, cy_d), 'area': w * h})
                dets = nms(dets)

                if dets:
                    last_vein_dets = dets;  vein_prop_cnt = 0
                elif last_vein_dets and vein_prop_cnt < MAX_PROP:
                    dets = last_vein_dets;  vein_prop_cnt += 1
                else:
                    vein_prop_cnt += 1

                # Drop detections below the fascia band
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

                # Classify geometrically (N1/N2/N3) once track is 2 frames old
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
    target_fps = float(request.args.get('fps', 5))
    generate   = _make_generate_foundation(upload_id, target_fps)
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

@app.route('/api/test/fascia/<upload_id>', methods=['GET'])
def test_fascia(upload_id):
    if upload_id not in _store:
        return jsonify({'error': 'Not found'}), 404
    target_fps   = float(request.args.get('fps', 5))
    groq_key_arg = request.args.get('groq_key', DEFAULT_GROQ_KEY)
    vlm_model_arg = request.args.get('vlm_model', 'meta-llama/llama-4-scout-17b-16e-instruct')

    def generate():
        # Pipeline B VLM client (Groq)
        try:
            from groq import Groq as _Groq
            gc = _Groq(api_key=groq_key_arg)
        except Exception:
            gc = None

        tmp_path = _store[upload_id]['tmp_path']
        cap      = cv2.VideoCapture(tmp_path)
        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip     = max(1, round(src_fps / target_fps))
        n_proc   = max(1, total // skip)

        yield f"data: {json.dumps({'type':'meta','total':n_proc,'src_fps':src_fps})}\n\n"

        raw_idx    = 0
        proc_idx   = 0
        scan_xl_m  = None
        scan_xr_m  = None
        fascia_buf = []          # rolling buffer for temporal smoothing
        SMOOTH     = 5           # median over last 5 non-null readings

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

                # Pipeline B: VLM-only fascia detection (no trained DL model)
                fy, _ = _pipe_b.vlm_detect_classify(frame, gc, vlm_model_arg)

                if fy is not None:
                    fascia_buf.append(fy)
                    if len(fascia_buf) > SMOOTH:
                        fascia_buf.pop(0)

                # Use median of buffer for smoothness; fall back to latest raw if buffer empty
                fy_smooth = int(round(float(np.median(fascia_buf)))) if fascia_buf else fy

                top_y = bot_y = None
                if fy_smooth is not None:
                    half  = 8
                    x_lo  = max(0, scan_xl_m)
                    x_hi  = min(IMG_SIZE, scan_xr_m)
                    fy_c  = int(np.clip(fy_smooth, half, IMG_SIZE - 1 - half))
                    top_y = np.full(IMG_SIZE, np.nan, np.float32)
                    bot_y = np.full(IMG_SIZE, np.nan, np.float32)
                    top_y[x_lo:x_hi] = float(fy_c - half)
                    bot_y[x_lo:x_hi] = float(fy_c + half)

                annotated = draw_frame(frame, [], top_y, bot_y)
                b64 = frame_to_b64(annotated)

                yield f"data: {json.dumps({'type':'frame','frame_idx':proc_idx,'total':n_proc,'image':b64,'fascia_y':fy_smooth})}\n\n"
                proc_idx += 1
            raw_idx += 1

        cap.release()
        yield f"data: {json.dumps({'type':'done','total_processed':proc_idx})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/test/veins/<upload_id>', methods=['GET'])
def test_veins(upload_id):
    if upload_id not in _store:
        return jsonify({'error': 'Not found'}), 404
    target_fps = float(request.args.get('fps', 5))

    def generate():
        cap = None
        try:
            # Ensure Grounding DINO is loaded (only DINO — not the full pipeline)
            if _pipe_b._gdino is None:
                _pipe_b._load_gdino()

            tmp_path = _store[upload_id]['tmp_path']
            cap      = cv2.VideoCapture(tmp_path)
            src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip     = max(1, round(src_fps / target_fps))
            n_proc   = max(1, total // skip)

            print(f"[vein test] start: total={total} skip={skip} n_proc={n_proc} "
                  f"gdino={'yes' if _pipe_b._gdino else 'NO'}", flush=True)
            yield f"data: {json.dumps({'type':'meta','total':n_proc,'src_fps':src_fps})}\n\n"

            raw_idx  = 0
            proc_idx = 0

            GDINO_VEIN_TEXT = (
                "blood vessel . vein . dark anechoic oval . dark hypoechoic tube . "
                "dark elongated vessel . dark circular vessel . "
                "dark horizontal band . dark anechoic region . dark tube"
            )

            # Temporal propagation state — carry forward last detection for up
            # to MAX_PROP frames when DINO finds nothing in the current frame.
            # Handles the common case where DINO confidence dips just below
            # threshold on a few frames of a steady vein.
            MAX_PROP    = 4
            last_dets   = []
            prop_count  = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if raw_idx % skip == 0:
                    vis = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

                    # CLAHE enhance for darkness filter
                    lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                    vis_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    gray_vis = cv2.cvtColor(vis_enh, cv2.COLOR_BGR2GRAY)

                    # ── Primary: Grounding DINO ───────────────────────────────
                    dets_raw = []
                    if _pipe_b._gdino is not None:
                        dets_raw = _pipe_b._gdino_detect(
                            vis, min_area=60, box_threshold=0.06,
                            text=GDINO_VEIN_TEXT,
                            max_aspect=15,
                            edge_skip_frac=0.03,
                        )

                    # ── Darkness + size/edge filter ───────────────────────────
                    MAX_VEIN_AREA = IMG_SIZE * IMG_SIZE * 0.15  # 9830 px²
                    dets = []
                    for d in dets_raw:
                        x, y, w, h = [int(v) for v in d['bbox']]
                        # Reject huge boxes — scan-border artifacts, not veins
                        if w * h > MAX_VEIN_AREA:
                            continue
                        # Reject boxes that kiss the bottom edge (machine UI panel)
                        if y + h > IMG_SIZE - 6:
                            continue
                        margin = max(2, min(w, h) // 6)
                        roi = gray_vis[y+margin:y+h-margin, x+margin:x+w-margin]
                        if roi.size > 0:
                            m = float(roi.mean())
                            if 10 < m < 90:
                                dets.append(d)
                    dets = nms(dets)

                    # ── Temporal propagation ──────────────────────────────────
                    # If DINO found nothing but we had a detection recently,
                    # carry the last known boxes forward (vein didn't disappear,
                    # DINO just lost confidence for a frame or two).
                    if dets:
                        last_dets  = dets
                        prop_count = 0
                        src = 'DINO'
                    elif last_dets and prop_count < MAX_PROP:
                        dets       = last_dets
                        prop_count += 1
                        src = f'prop({prop_count})'
                    else:
                        prop_count += 1
                        src = 'none'

                    print(f"[vein test] frame {proc_idx}: {len(dets_raw)} DINO raw → "
                          f"{len(dets)} [{src}]", flush=True)

                    # Draw clean cyan boxes
                    annotated = vis.copy()
                    vein_summary = []
                    for d in dets:
                        x, y, w, h = [int(v) for v in d['bbox']]

                        # Find the actual vein contour within the DINO box.
                        # Find the dark region within the DINO box and fit
                        # an ellipse to it — veins are oval/circular structures.
                        roi = gray_vis[y:y+h, x:x+w]
                        roi_mean = float(roi.mean())
                        thresh_val = max(20, min(int(roi_mean * 0.75), 90))
                        _, roi_bin = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
                        roi_bin = cv2.GaussianBlur(roi_bin, (7, 7), 0)
                        _, roi_bin = cv2.threshold(roi_bin, 127, 255, cv2.THRESH_BINARY)
                        cnts, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        drew_contour = False
                        if cnts:
                            cnt = max(cnts, key=cv2.contourArea)
                            if cv2.contourArea(cnt) > 30:
                                cnt_global = cnt + np.array([[[x, y]]])
                                if len(cnt_global) >= 5:
                                    # Fit a smooth ellipse — perfect for circular/oval veins
                                    ellipse = cv2.fitEllipse(cnt_global)
                                    cv2.ellipse(annotated, ellipse, (0, 255, 255), 2)
                                    # Derive summary bbox from ellipse params
                                    (ex, ey), (ea, eb), angle = ellipse
                                    ew, eh = int(max(ea, eb)), int(min(ea, eb))
                                    vein_summary.append({
                                        'x': max(0, int(ex - ew/2)),
                                        'y': max(0, int(ey - eh/2)),
                                        'w': ew, 'h': eh,
                                    })
                                else:
                                    cv2.drawContours(annotated, [cnt_global], -1, (0, 255, 255), 2)
                                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                                    vein_summary.append({'x': x+rx, 'y': y+ry, 'w': rw, 'h': rh})
                                drew_contour = True
                        if not drew_contour:
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            vein_summary.append({'x': x, 'y': y, 'w': w, 'h': h})

                    b64 = frame_to_b64(annotated)
                    yield f"data: {json.dumps({'type':'frame','frame_idx':proc_idx,'total':n_proc,'image':b64,'veins':vein_summary})}\n\n"
                    proc_idx += 1
                raw_idx += 1

            print(f"[vein test] done: {proc_idx} frames processed", flush=True)
            yield f"data: {json.dumps({'type':'done','total_processed':proc_idx})}\n\n"

        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            print(f"[vein test] EXCEPTION: {msg}", flush=True)
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"
            yield f"data: {json.dumps({'type':'done','total_processed':0})}\n\n"
        finally:
            if cap is not None:
                cap.release()

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/test/combined/<upload_id>', methods=['GET'])
def test_combined(upload_id):
    if upload_id not in _store:
        return jsonify({'error': 'Not found'}), 404
    target_fps = float(request.args.get('fps', 5))

    def generate():
        cap = None
        try:
            # GDINO for veins; GDINO + SAM for fascia curved boundary
            if _pipe_b._gdino is None:
                _pipe_b._load_gdino()
            if _pipe_b._sam is None:
                _pipe_b._load_sam()

            tmp_path = _store[upload_id]['tmp_path']
            cap      = cv2.VideoCapture(tmp_path)
            src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip     = max(1, round(src_fps / target_fps))
            n_proc   = max(1, total // skip)

            print(f"[combined test] start: n_proc={n_proc}", flush=True)
            yield f"data: {json.dumps({'type':'meta','total':n_proc,'src_fps':src_fps})}\n\n"

            GDINO_VEIN_TEXT = (
                "blood vessel . vein . dark anechoic oval . dark hypoechoic tube . "
                "dark elongated vessel . dark circular vessel . "
                "dark horizontal band . dark anechoic region . dark tube"
            )
            MAX_VEIN_AREA = IMG_SIZE * IMG_SIZE * 0.15
            MAX_PROP      = 4
            last_dets     = []
            prop_count    = 0
            fascia_buf    = []   # list of (top_y_arr, bot_y_arr)
            SMOOTH        = 5
            FASCIA_COLOR  = (255, 255, 0)
            VEIN_COLOR    = (0,   255, 255)
            scan_xl_m     = None
            scan_xr_m     = None

            raw_idx  = 0
            proc_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if raw_idx % skip == 0:
                    vis = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    w0  = frame.shape[1]

                    if scan_xl_m is None:
                        xl_px, xr_px = get_scan_x_limits(frame)
                        scan_xl_m = max(0, round(xl_px * IMG_SIZE / w0))
                        scan_xr_m = min(IMG_SIZE, round(xr_px * IMG_SIZE / w0))

                    # ── Fascia: GDINO → SAM (curved per-column boundary) ──────
                    top_y_raw, bot_y_raw = _pipe_b.detect_fascia(vis, scan_xl_m, scan_xr_m)
                    if top_y_raw is not None:
                        fascia_buf.append((top_y_raw, bot_y_raw))
                        if len(fascia_buf) > SMOOTH:
                            fascia_buf.pop(0)

                    if fascia_buf:
                        top_y = np.nanmedian(np.stack([f[0] for f in fascia_buf]), axis=0).astype(np.float32)
                        bot_y = np.nanmedian(np.stack([f[1] for f in fascia_buf]), axis=0).astype(np.float32)
                        # Extra smoothing pass — SAM per-column output is noisy even after temporal median
                        valid = ~np.isnan(top_y)
                        if valid.sum() > 10:
                            xs_s = np.where(valid)[0].astype(np.float32)
                            x_lo_s, x_hi_s = int(xs_s.min()), int(xs_s.max())
                            rng_s = np.arange(x_lo_s, x_hi_s + 1, dtype=np.float32)
                            top_y[x_lo_s:x_hi_s+1] = uniform_filter1d(
                                np.interp(rng_s, xs_s, top_y[valid]), 100).astype(np.float32)
                            bot_y[x_lo_s:x_hi_s+1] = uniform_filter1d(
                                np.interp(rng_s, xs_s, bot_y[valid]), 100).astype(np.float32)
                        valid_top = top_y[~np.isnan(top_y)]
                        fy_mid = int(np.mean(valid_top) + (np.nanmean(bot_y - top_y)) / 2) if len(valid_top) else None
                    else:
                        top_y = bot_y = None
                        fy_mid = None

                    # ── Vein: GDINO ───────────────────────────────────────────
                    lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                    vis_enh  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    gray_vis = cv2.cvtColor(vis_enh, cv2.COLOR_BGR2GRAY)

                    dets_raw = _pipe_b._gdino_detect(
                        vis, min_area=60, box_threshold=0.06,
                        text=GDINO_VEIN_TEXT, max_aspect=15, edge_skip_frac=0.03,
                    )

                    dets = []
                    for d in dets_raw:
                        x, y, w, h = [int(v) for v in d['bbox']]
                        if w * h > MAX_VEIN_AREA or y + h > IMG_SIZE - 6:
                            continue
                        margin = max(2, min(w, h) // 6)
                        roi = gray_vis[y+margin:y+h-margin, x+margin:x+w-margin]
                        if roi.size > 0:
                            m = float(roi.mean())
                            if 10 < m < 90:
                                dets.append(d)
                    dets = nms(dets)

                    if dets:
                        last_dets = dets;  prop_count = 0
                    elif last_dets and prop_count < MAX_PROP:
                        dets = last_dets;  prop_count += 1
                    else:
                        prop_count += 1

                    # ── Draw ─────────────────────────────────────────────────
                    annotated = vis.copy()

                    # Fascia: curved polylines from SAM per-column boundary
                    if top_y is not None:
                        xs_v = [x for x in range(IMG_SIZE) if not np.isnan(top_y[x])]
                        if len(xs_v) > 1:
                            pts_top = np.array([[x, int(top_y[x])] for x in xs_v], np.int32)
                            pts_bot = np.array([[x, int(bot_y[x])] for x in xs_v], np.int32)
                            cv2.polylines(annotated, [pts_top], False, FASCIA_COLOR, 2)
                            cv2.polylines(annotated, [pts_bot], False, FASCIA_COLOR, 2)

                    # Veins: yellow ellipses
                    vein_summary = []
                    for d in dets:
                        x, y, w, h = [int(v) for v in d['bbox']]
                        roi        = gray_vis[y:y+h, x:x+w]
                        roi_mean   = float(roi.mean())
                        thresh_val = max(20, min(int(roi_mean * 0.75), 90))
                        _, roi_bin = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
                        roi_bin    = cv2.GaussianBlur(roi_bin, (7, 7), 0)
                        _, roi_bin = cv2.threshold(roi_bin, 127, 255, cv2.THRESH_BINARY)
                        cnts, _    = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        drew = False
                        if cnts:
                            cnt = max(cnts, key=cv2.contourArea)
                            if cv2.contourArea(cnt) > 30:
                                cnt_g = cnt + np.array([[[x, y]]])
                                if len(cnt_g) >= 5:
                                    ellipse = cv2.fitEllipse(cnt_g)
                                    cv2.ellipse(annotated, ellipse, VEIN_COLOR, 2)
                                    (ex, ey), (ea, eb), _ = ellipse
                                    ew, eh = int(max(ea, eb)), int(min(ea, eb))
                                    vein_summary.append({'x': max(0, int(ex-ew/2)), 'y': max(0, int(ey-eh/2)), 'w': ew, 'h': eh})
                                else:
                                    cv2.drawContours(annotated, [cnt_g], -1, VEIN_COLOR, 2)
                                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                                    vein_summary.append({'x': x+rx, 'y': y+ry, 'w': rw, 'h': rh})
                                drew = True
                        if not drew:
                            cv2.rectangle(annotated, (x, y), (x+w, y+h), VEIN_COLOR, 2)
                            vein_summary.append({'x': x, 'y': y, 'w': w, 'h': h})

                    b64 = frame_to_b64(annotated)
                    yield f"data: {json.dumps({'type':'frame','frame_idx':proc_idx,'total':n_proc,'image':b64,'fascia_y':fy_mid,'veins':vein_summary})}\n\n"
                    proc_idx += 1
                raw_idx += 1

            print(f"[combined test] done: {proc_idx} frames", flush=True)
            yield f"data: {json.dumps({'type':'done','total_processed':proc_idx})}\n\n"

        except Exception as e:
            import traceback
            print(f"[combined test] EXCEPTION: {traceback.format_exc()}", flush=True)
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"
            yield f"data: {json.dumps({'type':'done','total_processed':0})}\n\n"
        finally:
            if cap is not None:
                cap.release()

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    print(f"==> App running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)