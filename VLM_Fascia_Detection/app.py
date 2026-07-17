import base64
import io
import os
import sys
import uuid
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'stubs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'BiomedParse'))
sys.path.insert(0, os.path.join(BASE_DIR, 'LISA'))

from modeling.BaseModel import BaseModel
from modeling import build_model
from modeling.language.loss import vl_similarity
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
print("Loading BiomedParse model...")
BIOMEDPARSE_DIR = os.path.join(BASE_DIR, 'BiomedParse')
opt = load_opt_from_config_files([os.path.join(BIOMEDPARSE_DIR, 'configs', 'biomedparse_inference.yaml')])
opt = init_distributed(opt)

LOCAL_WEIGHTS = os.path.join(BASE_DIR, 'pretrained', 'biomedparse_v1.pt')
pretrained_source = LOCAL_WEIGHTS if os.path.exists(LOCAL_WEIGHTS) else 'hf_hub:microsoft/BiomedParse'
print(f"  weights: {pretrained_source}")

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_source).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"], is_eval=True
    )
print("Model loaded.")

# Image transform matching BiomedParse training (1024x1024 bicubic)
_img_transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)
])

VEIN_COLOR   = (0, 210, 0)    # green (RGB)
FASCIA_COLOR = (0, 210, 210)  # cyan  (RGB)

# ---------------------------------------------------------------------------
# Reference image/mask paths
# 3 examples per structure, selected from Task_3_Experimental training data
# using has_fascia/has_vein=True rows spread across different videos.
# ---------------------------------------------------------------------------
TASK3_OUT = os.path.join(
    BASE_DIR, '..', 'Task_3_Experimental',
    'vein_detection_task_3_training', 'output'
)

FASCIA_REFS = [
    (
        os.path.join(TASK3_OUT, 'fascia', 'frames', '202207111318_38-Perf', '00050.png'),
        os.path.join(TASK3_OUT, 'fascia', 'masks',  '202207111318_38-Perf', '00050.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'fascia', 'frames', '202207191643_00-Moving', '00598.png'),
        os.path.join(TASK3_OUT, 'fascia', 'masks',  '202207191643_00-Moving', '00598.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'fascia', 'frames', 'sample_data', '00084.png'),
        os.path.join(TASK3_OUT, 'fascia', 'masks',  'sample_data', '00084.png'),
    ),
]

VEIN_REFS = [
    (
        os.path.join(TASK3_OUT, 'vein', 'frames', '202207111318_38-Perf_00004.png'),
        os.path.join(TASK3_OUT, 'vein', 'masks',  '202207111318_38-Perf_00004.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'vein', 'frames', '202207191643_00-Moving_00508.png'),
        os.path.join(TASK3_OUT, 'vein', 'masks',  '202207191643_00-Moving_00508.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'vein', 'frames', 'sample_data_00212.png'),
        os.path.join(TASK3_OUT, 'vein', 'masks',  'sample_data_00212.png'),
    ),
]


# ---------------------------------------------------------------------------
# Visual few-shot: encode reference → infer query
# ---------------------------------------------------------------------------

def _img_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Resize to 1024x1024 and convert to [C, H, W] CUDA tensor."""
    resized = _img_transform(pil_image)
    arr = np.asarray(resized).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def encode_reference(ref_image_pil: Image.Image, ref_mask_pil: Image.Image) -> dict:
    """
    Run a (reference image, binary mask) pair through BiomedParse's
    spatial→visual referencing pipeline (task='refimg') to produce
    visual query features that can be injected into any query image pass.

    This is BiomedParse's built-in few-shot visual prompting mechanism:
    the model sees ONE annotated example and learns what the target
    structure looks like, then segments it in unseen images.
    """
    img_tensor = _img_to_tensor(ref_image_pil)

    # Mask must be 1024x1024 boolean, shape [1, H, W]
    mask_arr = np.array(ref_mask_pil.resize((1024, 1024), Image.NEAREST))
    mask_bool = torch.from_numpy(mask_arr > 127).cuda().unsqueeze(0)

    data = {
        'image':         img_tensor,
        'spatial_query': {'rand_shape': mask_bool},
        'height':        ref_image_pil.size[1],
        'width':         ref_image_pil.size[0],
    }

    # spatial=True is required for evaluate_referring_image
    model.model.task_switch['spatial']  = True
    model.model.task_switch['visual']   = True
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio']    = False

    visual_features, _ = model.model.evaluate_referring_image([data])
    return visual_features   # dict: visual_query_pos/neg, src_visual_queries/maskings


def vlm_visual_infer_prob(
    query_image_pil: Image.Image,
    visual_features_list: list,
) -> np.ndarray:
    """
    Run BiomedParse visual few-shot inference and return the RAW probability
    map (float32 H×W, values 0–1) — pixel-wise max ensemble across all refs.
    Callers decide how to consume the probabilities (threshold / centreline).
    """
    img_tensor = _img_to_tensor(query_image_pil)
    W, H = query_image_pil.size

    model.model.task_switch['spatial']   = True
    model.model.task_switch['visual']    = True
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio']     = False

    best_prob = None

    for vis_feat in visual_features_list:
        data = {'image': img_tensor, 'visual': vis_feat, 'height': H, 'width': W}
        results, _, _ = model.model.evaluate_demo([data])

        if 'pred_masks' in results:
            pred_masks = results['pred_masks'][0]
        elif 'pred_smasks' in results:
            pred_masks = results['pred_smasks'][0]
        else:
            continue

        vis_pos      = results.get('pred_pvisuals')
        pred_maskemb = results.get('pred_maskembs')
        pred_caps    = results.get('pred_captions')
        pred_logits  = results.get('pred_logits')

        if vis_pos is not None and vis_pos.numel() > 0:
            vis_q = vis_pos[0, 0] / (vis_pos[0, 0].norm() + 1e-7)
            emb   = pred_maskemb[0] if pred_maskemb is not None else (
                    pred_caps[0]    if pred_caps    is not None else None)
            if emb is not None:
                matched = (emb / (emb.norm(dim=-1, keepdim=True) + 1e-7) @ vis_q).argmax()
            elif pred_logits is not None:
                matched = pred_logits[0].max(dim=-1)[0].argmax()
            else:
                matched = pred_masks.sigmoid().mean(dim=(-2, -1)).argmax()
        elif pred_logits is not None:
            matched = pred_logits[0].max(dim=-1)[0].argmax()
        else:
            matched = pred_masks.sigmoid().mean(dim=(-2, -1)).argmax()

        prob = F.interpolate(
            pred_masks[matched:matched+1, :, :][None], (H, W), mode='bilinear'
        )[0, 0, :H, :W].sigmoid().cpu().numpy()

        best_prob = prob if best_prob is None else np.maximum(best_prob, prob)

    return best_prob.astype(np.float32) if best_prob is not None else np.zeros((H, W), dtype=np.float32)


def vlm_text_infer_prob(
    query_image_pil: Image.Image,
    text_prompts: list,
) -> np.ndarray:
    """
    Text-grounded segmentation via BiomedParse.
    Encodes each text prompt with BiomedBERT, matches to mask proposals via
    vl_similarity, and returns the pixel-wise max probability across all prompts.
    """
    img_tensor = _img_to_tensor(query_image_pil)
    W, H = query_image_pil.size

    model.model.task_switch['spatial']   = False
    model.model.task_switch['visual']    = False
    model.model.task_switch['grounding'] = True
    model.model.task_switch['audio']     = False

    data = {"image": img_tensor, 'text': text_prompts, "height": H, "width": W}
    results, _, extra = model.model.evaluate_demo([data])

    pred_masks = results['pred_masks'][0]   # [N_proposals, H_enc, W_enc]
    v_emb      = results['pred_captions'][0]  # [N_proposals, D]
    t_emb      = extra['grounding_class']     # [N_texts, D]

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob    = vl_similarity(v_emb, t_emb, temperature=temperature)  # [N_proposals, N_texts]

    # Best-matching proposal per text prompt; ensemble by taking pixel-wise max
    matched_ids = out_prob.max(0)[1]  # [N_texts]
    best_prob   = np.zeros((H, W), dtype=np.float32)
    for idx in matched_ids:
        prob = F.interpolate(
            pred_masks[idx:idx+1, :, :][None], (H, W), mode='bilinear'
        )[0, 0, :H, :W].sigmoid().cpu().numpy()
        best_prob = np.maximum(best_prob, prob.astype(np.float32))

    return best_prob


def prob_to_vein_mask(prob: np.ndarray, threshold: float = 0.40) -> np.ndarray:
    """Hard-threshold the probability map to get a filled vein region."""
    return (prob > threshold).astype(np.uint8)


def prob_to_fascia_centreline(prob: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Convert VLM fascia probability map to a thin curvilinear line.

    Per column: peak-row argmax where prob > threshold.
    Noise rejection:
      - Require at least 40% of image columns to have signal (random noise
        never spans the full width of the image).
      - Slope filter: drop columns where y-position jumps >8px from the
        smoothed running position (fascia is nearly horizontal).
    Smooth y-coordinates, dilate 3px for visibility.
    """
    H, W = prob.shape
    centreline = np.zeros((H, W), dtype=np.uint8)
    valid_cols, y_vals = [], []

    for col in range(W):
        col_p = prob[:, col]
        if col_p.max() > threshold:
            valid_cols.append(col)
            y_vals.append(float(np.argmax(col_p)))

    # Must span ≥40% of image width to be a real fascia line
    if len(valid_cols) < 0.40 * W:
        return centreline

    y_arr = np.array(y_vals, dtype=np.float32)

    # Smooth first so the slope filter works on a clean signal
    k = min(31, len(y_arr))
    if k >= 3:
        y_smooth = np.convolve(y_arr, np.ones(k, np.float32) / k, mode='same')
    else:
        y_smooth = y_arr.copy()

    # Slope filter: max 8px jump between adjacent columns (fascia ≈ horizontal)
    keep = np.ones(len(y_smooth), dtype=bool)
    for i in range(1, len(y_smooth)):
        if abs(y_smooth[i] - y_smooth[i-1]) > 8:
            keep[i] = False

    for i, col in enumerate(valid_cols):
        if keep[i]:
            centreline[int(y_smooth[i]), col] = 255

    h_close    = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    centreline = cv2.morphologyEx(centreline, cv2.MORPH_CLOSE, h_close)
    v_dilate   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    centreline = cv2.dilate(centreline, v_dilate, iterations=1)
    return centreline


# ---------------------------------------------------------------------------
# Pre-compute visual reference features at startup
# ---------------------------------------------------------------------------

def _load_ref_pair(img_path, mask_path):
    img  = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    return img, mask


print("Encoding visual reference examples (few-shot setup)...")
with torch.no_grad():
    FASCIA_VIS_FEATURES = []
    for img_path, msk_path in FASCIA_REFS:
        if os.path.exists(img_path) and os.path.exists(msk_path):
            img, msk = _load_ref_pair(img_path, msk_path)
            FASCIA_VIS_FEATURES.append(encode_reference(img, msk))
        else:
            print(f"  [WARN] missing fascia ref: {img_path}")

    VEIN_VIS_FEATURES = []
    for img_path, msk_path in VEIN_REFS:
        if os.path.exists(img_path) and os.path.exists(msk_path):
            img, msk = _load_ref_pair(img_path, msk_path)
            VEIN_VIS_FEATURES.append(encode_reference(img, msk))
        else:
            print(f"  [WARN] missing vein ref: {img_path}")

print(f"  {len(FASCIA_VIS_FEATURES)} fascia refs, {len(VEIN_VIS_FEATURES)} vein refs encoded.")
print("Ready. Open http://localhost:5000")


# ---------------------------------------------------------------------------
# Classical CV — fallback / enhancement layer
# Runs on every frame and is unioned with the VLM result.
# ---------------------------------------------------------------------------

def cv_detect_vein(gray: np.ndarray) -> np.ndarray:
    """
    Veins = anechoic (very dark), oval/circular blobs.
    Thresholds for truly dark pixels → morphological clean-up →
    filter by circularity and area → keep top-3 largest round blobs.
    """
    h, w = gray.shape
    blurred  = cv2.GaussianBlur(gray, (15, 15), 5)
    mean_val = float(np.mean(blurred))
    std_val  = float(np.std(blurred))

    dark = (blurred < (mean_val - 0.70 * std_val)).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (0.008 * h * w < area < 0.25 * h * w):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.35:
            candidates.append((area, cnt))

    candidates.sort(key=lambda x: x[0], reverse=True)
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, cnt in candidates[:1]:   # keep only the single largest vessel
        cv2.drawContours(mask, [cnt], -1, 1, cv2.FILLED)
    return mask


def cv_detect_fascia(gray: np.ndarray, vein_mask: np.ndarray) -> np.ndarray:
    """
    Fascia = thin curvilinear horizontal lines.

    Uses WHITE TOP-HAT transform with a tall vertical SE (31px).
    Top-hat = image − open(image, SE).  Any feature BROADER than the SE is
    removed by the opening and therefore disappears from the top-hat output.
    The machine display border is a large uniform-bright region (>31px tall)
    so it is suppressed automatically — no manual border cropping needed.
    Real fascia bands are 2-8px thick and therefore appear as strong responses.

    Pipeline:
      tophat → threshold → horizontal close → centreline per column →
      box-smooth centreline → 3px vertical dilate → vein exclusion.
    """
    h, w = gray.shape
    top_margin   = int(0.10 * h)
    bot_margin   = int(0.15 * h)
    left_margin  = int(0.04 * w)
    right_margin = int(0.04 * w)
    roi = gray[top_margin: h - bot_margin, left_margin: w - right_margin]
    rh, rw = roi.shape

    blurred = cv2.GaussianBlur(roi, (5, 5), 2)

    # White top-hat: keep only features thinner than 31px vertically
    se     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, se)

    # Adaptive threshold on the top-hat response
    th_val = max(8, int(np.percentile(tophat[tophat > 0], 60)) if tophat.any() else 8)
    _, th_bin = cv2.threshold(tophat, th_val, 255, cv2.THRESH_BINARY)

    # Connect along each horizontal line, merge nearby rows
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1))
    th_bin   = cv2.morphologyEx(th_bin, cv2.MORPH_CLOSE, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    th_bin   = cv2.morphologyEx(th_bin, cv2.MORPH_CLOSE, v_kernel)

    cnts, _ = cv2.findContours(th_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Score by mean top-hat response; keep top-2 widest-spanning bands
    scored = []
    for cnt in cnts:
        x0, _, bw, _ = cv2.boundingRect(cnt)
        if bw < 0.30 * rw:
            continue
        region = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(region, [cnt], -1, 255, cv2.FILLED)
        mean_resp = float(tophat[region > 0].mean())
        scored.append((mean_resp, x0, bw, region))

    scored.sort(key=lambda x: x[0], reverse=True)

    roi_mask = np.zeros((rh, rw), dtype=np.uint8)
    for mean_resp, x0, bw, region in scored[:2]:
        valid_cols, y_vals = [], []
        for col in range(x0, x0 + bw):
            ys = np.where(region[:, col] > 0)[0]
            if ys.size:
                valid_cols.append(col)
                y_vals.append(float(np.mean(ys)))
        if not valid_cols:
            continue
        y_arr = np.array(y_vals, dtype=np.float32)
        k = min(31, len(y_arr))
        if k >= 3:
            y_arr = np.convolve(y_arr, np.ones(k, np.float32) / k, mode='same')
        for i, col in enumerate(valid_cols):
            roi_mask[int(y_arr[i]), col] = 255

    v_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    roi_mask = cv2.dilate(roi_mask, v_dilate, iterations=1)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top_margin: h - bot_margin, left_margin: w - right_margin] = roi_mask

    if vein_mask is not None:
        excl = cv2.dilate(vein_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
        mask[excl > 0] = 0
    return mask


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(image_rgb: np.ndarray, vein_mask: np.ndarray, fascia_mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()

    # Fascia: direct thin-line pixel overlay (mask already contains 5px lines)
    if fascia_mask is not None and fascia_mask.max() > 0:
        out[fascia_mask > 0] = FASCIA_COLOR

    # Vein: outline only, no fill — matching clinical annotation style
    if vein_mask is not None and vein_mask.max() > 0:
        cnts, _ = cv2.findContours(vein_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, VEIN_COLOR, 3)

    return out


def make_mask_viz(vein_mask: np.ndarray, fascia_mask: np.ndarray,
                  h: int, w: int) -> np.ndarray:
    """Black canvas with vein=green pixels, fascia=cyan pixels."""
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    if fascia_mask is not None and fascia_mask.max() > 0:
        viz[fascia_mask > 0] = FASCIA_COLOR
    if vein_mask is not None and vein_mask.max() > 0:
        viz[vein_mask > 0] = VEIN_COLOR
    return viz


def _arr_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    uid  = uuid.uuid4().hex
    input_path  = os.path.join(UPLOAD_DIR, f'{uid}_input.png')
    output_path = os.path.join(OUTPUT_DIR, f'{uid}_output.png')

    image = Image.open(file.stream).convert('RGB')
    image.save(input_path)

    image_np = np.array(image)

    with torch.no_grad():
        # Vein: visual few-shot + text grounding ensemble
        vein_vis_prob  = vlm_visual_infer_prob(image, VEIN_VIS_FEATURES)
        vein_txt_prob  = vlm_text_infer_prob(image, [
            "vein", "blood vessel", "anechoic vessel in ultrasound"
        ])
        vein_mask = prob_to_vein_mask(
            np.maximum(vein_vis_prob, vein_txt_prob), threshold=0.40
        )

        # Fascia: visual few-shot + text grounding ensemble → centreline
        fascia_vis_prob = vlm_visual_infer_prob(image, FASCIA_VIS_FEATURES)
        fascia_txt_prob = vlm_text_infer_prob(image, [
            "fascia", "hyperechoic fascial layer",
            "connective tissue layer", "fascial plane in ultrasound"
        ])
        fascia_prob = np.maximum(fascia_vis_prob, fascia_txt_prob)
        fascia_mask = prob_to_fascia_centreline(fascia_prob, threshold=0.15)

    annotated = annotate(image_np, vein_mask, fascia_mask)
    masks_viz = make_mask_viz(vein_mask, fascia_mask, image_np.shape[0], image_np.shape[1])

    Image.fromarray(annotated).save(output_path)
    return jsonify({
        'output': _arr_to_b64(annotated),
        'masks':  _arr_to_b64(masks_viz),
    })


# ---------------------------------------------------------------------------
# LISA model — loaded if VLM_Fascia_Detection/LISA repo is present.
# Model weights stored in pretrained/LISA-7B-v1/ (downloaded by setup).
# Override with LISA_MODEL_PATH env var to point to a different checkpoint.
# ---------------------------------------------------------------------------

LISA_AVAILABLE   = False
lisa_model       = None
lisa_tokenizer   = None
lisa_image_proc  = None

_local_lisa_weights = os.path.join(BASE_DIR, 'pretrained', 'LISA-7B-v1')
LISA_MODEL_PATH = os.environ.get(
    'LISA_MODEL_PATH',
    _local_lisa_weights if os.path.exists(_local_lisa_weights) else 'xinlai/LISA-7B-v1'
)

_lisa_dir = os.path.join(BASE_DIR, 'LISA')
if os.path.exists(_lisa_dir):
    try:
        from transformers import AutoTokenizer, CLIPImageProcessor
        from model.LISA import LISAForCausalLM

        print(f"Loading LISA model from '{LISA_MODEL_PATH}' …")

        lisa_tokenizer = AutoTokenizer.from_pretrained(
            LISA_MODEL_PATH,
            cache_dir=None,
            model_max_length=512,
            padding_side='right',
            use_fast=False,
        )
        lisa_tokenizer.pad_token = lisa_tokenizer.unk_token
        lisa_tokenizer.add_tokens(['[SEG]'])
        seg_token_idx = lisa_tokenizer('[SEG]', add_special_tokens=False).input_ids[0]

        lisa_model = LISAForCausalLM.from_pretrained(
            LISA_MODEL_PATH,
            low_cpu_mem_usage=True,
            vision_tower='openai/clip-vit-large-patch14',
            seg_token_idx=seg_token_idx,
            torch_dtype=torch.bfloat16,
        )
        lisa_model.config.eos_token_id = lisa_tokenizer.eos_token_id
        lisa_model.config.bos_token_id = lisa_tokenizer.bos_token_id
        lisa_model.config.pad_token_id = lisa_tokenizer.pad_token_id

        # Initialise CLIP vision tower (required before first forward pass)
        lisa_model.get_model().initialize_vision_modules(lisa_model.get_model().config)
        lisa_model.get_model().get_vision_tower().to(dtype=torch.bfloat16)

        lisa_model = lisa_model.bfloat16().cuda().eval()
        lisa_image_proc = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        LISA_AVAILABLE = True
        print("LISA model loaded.")
    except Exception as _lisa_err:
        print(f"[WARN] LISA load failed: {_lisa_err}")
        import traceback; traceback.print_exc()
else:
    print(f"[INFO] No LISA directory at {_lisa_dir}; LISA page will report unavailable.")


# ---------------------------------------------------------------------------
# LISA — highly descriptive text prompts for each anatomical structure
# ---------------------------------------------------------------------------

LISA_FASCIA_PROMPT = (
    "This is a medical ultrasound image of the leg for varicose vein examination. "
    "Segment the superficial fascia: a thin, bright (hyperechoic) horizontal line "
    "that separates the subcutaneous fat above from the muscle below."
)

LISA_VEIN_PROMPT = (
    "This is a medical ultrasound image of the leg for varicose vein examination. "
    "Segment all dark oval or tubular venous structures (veins): anechoic "
    "(black), compressible, with thin bright walls, located between the skin and fascia."
)


# ---------------------------------------------------------------------------
# LISA inference helper
# ---------------------------------------------------------------------------

def lisa_segment(image_pil: Image.Image, text_prompt: str) -> np.ndarray:
    """
    Run LISA reasoning-segmentation for one text prompt.
    Returns a binary uint8 mask [H, W] (0/1).
    """
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from model.segment_anything.utils.transforms import ResizeLongestSide  # type: ignore
    from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN)

    image_np = np.array(image_pil)   # RGB uint8
    H, W = image_np.shape[:2]

    # --- CLIP image (336×336, bfloat16) ---
    clip_img = lisa_image_proc.preprocess(image_np, return_tensors='pt')['pixel_values'][0]
    clip_img = clip_img.unsqueeze(0).cuda().to(torch.bfloat16)   # [1, 3, 336, 336]

    # --- SAM image: resize longest side to 1024, normalise, pad to 1024×1024 ---
    transform = ResizeLongestSide(1024)
    sam_np    = transform.apply_image(image_np)          # RGB numpy, longest side=1024
    resize_list = [sam_np.shape[:2]]                     # post-resize (h, w)

    pixel_mean = torch.tensor([123.675, 116.28,  103.53]).view(-1, 1, 1)
    pixel_std  = torch.tensor([58.395,  57.12,   57.375]).view(-1, 1, 1)
    sam_t = torch.from_numpy(sam_np).permute(2, 0, 1).contiguous().float()
    sam_t = (sam_t - pixel_mean) / pixel_std
    h, w  = sam_t.shape[-2:]
    sam_t = F.pad(sam_t, (0, 1024 - w, 0, 1024 - h))
    sam_t = sam_t.unsqueeze(0).cuda().to(torch.bfloat16)  # [1, 3, 1024, 1024]

    # --- Conversational prompt with <im_start>/<im_end> wrapping ---
    _tpl = 'llava_llama_2' if 'llama2' in LISA_MODEL_PATH.lower() else 'llava_v1'
    conv = conversation_lib.conv_templates[_tpl].copy()
    conv.messages = []

    img_str = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    user_msg = img_str + '\n' + text_prompt

    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], '')
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, lisa_tokenizer, return_tensors='pt'
    ).unsqueeze(0).cuda()

    with torch.no_grad():
        _, pred_masks = lisa_model.evaluate(
            clip_img,
            sam_t,
            input_ids,
            resize_list=resize_list,
            original_size_list=[(H, W)],
            max_new_tokens=32,
            tokenizer=lisa_tokenizer,
        )

    if pred_masks and len(pred_masks) > 0 and pred_masks[0] is not None \
            and pred_masks[0].shape[0] > 0:
        mask = pred_masks[0].detach().cpu().numpy()[0]   # float logits [H, W]
        return (mask > 0).astype(np.uint8)

    return np.zeros((H, W), dtype=np.uint8)


# ---------------------------------------------------------------------------
# LISA routes
# ---------------------------------------------------------------------------

@app.route('/lisa')
def lisa_page():
    return render_template('lisa.html', available=LISA_AVAILABLE)


@app.route('/predict_lisa', methods=['POST'])
def predict_lisa():
    if not LISA_AVAILABLE:
        return jsonify({
            'error': (
                'LISA model is not loaded. '
                'Clone https://github.com/dvlab-research/LISA to '
                'VLM_Fascia_Detection/LISA and restart the server.'
            )
        }), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    uid  = uuid.uuid4().hex
    input_path  = os.path.join(UPLOAD_DIR, f'{uid}_lisa_input.png')
    output_path = os.path.join(OUTPUT_DIR, f'{uid}_lisa_output.png')

    try:
        image = Image.open(file.stream).convert('RGB')
        image.save(input_path)
        image_np = np.array(image)

        with torch.no_grad():
            # Vein: LISA binary mask used directly
            vein_mask = lisa_segment(image, LISA_VEIN_PROMPT)

            # Fascia: LISA binary mask → thin centreline (reuse BiomedParse centreline logic)
            fascia_binary = lisa_segment(image, LISA_FASCIA_PROMPT)
            fascia_mask   = prob_to_fascia_centreline(
                fascia_binary.astype(np.float32), threshold=0.5
            )

        annotated  = annotate(image_np, vein_mask, fascia_mask)
        masks_viz  = make_mask_viz(vein_mask, fascia_mask, image_np.shape[0], image_np.shape[1])

        Image.fromarray(annotated).save(output_path)
        return jsonify({
            'output': _arr_to_b64(annotated),
            'masks':  _arr_to_b64(masks_viz),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Florence-2 model — microsoft/Florence-2-large (770 M params, ~1.5 GB)
# Supports REFERRING_EXPRESSION_SEGMENTATION natively; no SAM decoder needed.
# ---------------------------------------------------------------------------

FLORENCE_AVAILABLE  = False
florence_model      = None
florence_processor  = None

_local_florence = os.path.join(BASE_DIR, 'pretrained', 'Florence-2-large')
FLORENCE_MODEL_PATH = os.environ.get(
    'FLORENCE_MODEL_PATH',
    _local_florence if os.path.exists(_local_florence) else 'microsoft/Florence-2-large'
)

try:
    from transformers import AutoProcessor, AutoModelForCausalLM as _AutoMCLM

    print(f"Loading Florence-2 from '{FLORENCE_MODEL_PATH}' …")
    florence_processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_PATH, trust_remote_code=True
    )
    florence_model = _AutoMCLM.from_pretrained(
        FLORENCE_MODEL_PATH,
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation='eager',
    ).cuda()
    # safetensors only stores shared.weight; tie the three copies manually
    _lm = florence_model.language_model
    _lm.model.encoder.embed_tokens.weight = _lm.model.shared.weight
    _lm.model.decoder.embed_tokens.weight = _lm.model.shared.weight
    _lm.lm_head.weight                    = _lm.model.shared.weight
    florence_model.eval()
    FLORENCE_AVAILABLE = True
    print("Florence-2 loaded.")
except Exception as _fl_err:
    print(f"[WARN] Florence-2 load failed: {_fl_err}")
    import traceback; traceback.print_exc()


FLORENCE_FASCIA_PROMPT = "the bright white horizontal line running across the image"
FLORENCE_VEIN_PROMPT   = "the dark black circular or oval region"


def florence_segment(image_pil: Image.Image, prompt: str) -> np.ndarray:
    W, H = image_pil.size
    task  = '<REFERRING_EXPRESSION_SEGMENTATION>'
    inputs = florence_processor(
        text=task + prompt,
        images=image_pil,
        return_tensors='pt',
    ).to('cuda', torch.float16)

    with torch.no_grad():
        generated_ids = florence_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            use_cache=False,
        )

    generated_text = florence_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = florence_processor.post_process_generation(
        generated_text, task=task, image_size=(W, H)
    )

    mask = np.zeros((H, W), dtype=np.uint8)
    for instance_polys in parsed.get(task, {}).get('polygons', []):
        for polygon in instance_polys:
            if len(polygon) >= 6:
                pts = np.array(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
    return mask


# ---------------------------------------------------------------------------
# Florence-2 routes
# ---------------------------------------------------------------------------

@app.route('/florence')
def florence_page():
    return render_template('florence.html', available=FLORENCE_AVAILABLE)


@app.route('/predict_florence', methods=['POST'])
def predict_florence():
    if not FLORENCE_AVAILABLE:
        return jsonify({'error': 'Florence-2 model is not loaded.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    uid  = uuid.uuid4().hex
    input_path  = os.path.join(UPLOAD_DIR, f'{uid}_florence_input.png')
    output_path = os.path.join(OUTPUT_DIR, f'{uid}_florence_output.png')

    try:
        image = Image.open(file.stream).convert('RGB')
        image.save(input_path)
        image_np = np.array(image)

        with torch.no_grad():
            vein_mask     = florence_segment(image, FLORENCE_VEIN_PROMPT)
            fascia_binary = florence_segment(image, FLORENCE_FASCIA_PROMPT)
            fascia_mask   = prob_to_fascia_centreline(
                fascia_binary.astype(np.float32), threshold=0.5
            )

        annotated = annotate(image_np, vein_mask, fascia_mask)
        masks_viz = make_mask_viz(vein_mask, fascia_mask, image_np.shape[0], image_np.shape[1])

        Image.fromarray(annotated).save(output_path)
        return jsonify({
            'output': _arr_to_b64(annotated),
            'masks':  _arr_to_b64(masks_viz),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)