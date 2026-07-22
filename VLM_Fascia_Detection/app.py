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

import glob as _glob
_ckpt_dir = os.path.join(BASE_DIR, 'BiomedParse', 'output', 'fascia_finetuning_v3')
_ckpts = sorted(_glob.glob(os.path.join(_ckpt_dir, '**/model_state_dict.pt'), recursive=True),
                key=os.path.getmtime)
FINETUNED_WEIGHTS = _ckpts[-1] if _ckpts else None
LOCAL_WEIGHTS = os.path.join(BASE_DIR, 'pretrained', 'biomedparse_v1.pt')

if FINETUNED_WEIGHTS:
    pretrained_source = FINETUNED_WEIGHTS
elif os.path.exists(LOCAL_WEIGHTS):
    pretrained_source = LOCAL_WEIGHTS
else:
    pretrained_source = 'hf_hub:microsoft/BiomedParse'
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

VEIN_COLOR             = (0, 210, 0)    # green
FASCIA_SUP_COLOR       = (0, 230, 230)  # cyan  — superficial (skin-fascia interface)
FASCIA_DEEP_COLOR      = (0, 160, 230)  # blue  — deep (fascia-muscle interface)
FASCIA_COLOR           = FASCIA_SUP_COLOR  # legacy alias

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


def prob_to_vein_mask(prob: np.ndarray, threshold: float = 0.40,
                      image_gray: np.ndarray = None) -> np.ndarray:
    """
    Threshold + two filters:
      - min area: drops pixel-level noise
      - mean intensity: veins are anechoic (dark); rejects bright non-vein blobs
    """
    binary = (prob > threshold).astype(np.uint8)
    min_area = max(50, int(0.002 * prob.shape[0] * prob.shape[1]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if image_gray is not None:
            mean_val = float(image_gray[labels == i].mean())
            if mean_val > 90:   # too bright to be an anechoic vein
                continue
        out[labels == i] = 1
    return out


def prob_to_fascia_two_lines(prob: np.ndarray, threshold: float = 0.15):
    """
    Returns (superficial_mask, deep_mask).
    The model is trained on the full fascia zone (top edge = superficial line,
    bottom edge = deep line). Both edges are read directly from the predicted blob.
    """
    H, W = prob.shape
    LINE_HALF = 6                      # 13px thick per line

    col_max = prob.max(axis=0)
    valid   = col_max > threshold

    if valid.sum() < int(0.40 * W):
        return np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)

    above = prob > threshold
    # Top edge: first row with signal per column (superficial fascia line)
    sup_raw  = np.argmax(above, axis=0).astype(np.float64)
    # Bottom edge: last row with signal per column (deep fascia line)
    deep_raw = (H - 1 - np.argmax(above[::-1], axis=0)).astype(np.float64)

    valid_idx   = np.where(valid)[0]
    sup_filled  = np.interp(np.arange(W), valid_idx, sup_raw[valid_idx])
    deep_filled = np.interp(np.arange(W), valid_idx, deep_raw[valid_idx])

    # Edge-pad before convolving so boundary doesn't ramp toward zero
    k   = min(63, max(3, W // 16))
    pad = k // 2
    kernel = np.ones(k) / k
    sup_smooth  = np.convolve(np.pad(sup_filled,  pad, mode='edge'), kernel, mode='valid')[:W]
    deep_smooth = np.convolve(np.pad(deep_filled, pad, mode='edge'), kernel, mode='valid')[:W]

    sup_rows  = np.clip(sup_smooth.astype(int),  0, H - 1)
    deep_rows = np.clip(deep_smooth.astype(int), 0, H - 1)

    sup_mask  = np.zeros((H, W), dtype=np.uint8)
    deep_mask = np.zeros((H, W), dtype=np.uint8)
    cols = valid_idx
    for dr in range(-LINE_HALF, LINE_HALF + 1):
        sup_mask [np.clip(sup_rows[cols]  + dr, 0, H-1), cols] = 255
        deep_mask[np.clip(deep_rows[cols] + dr, 0, H-1), cols] = 255

    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    sup_mask  = cv2.morphologyEx(sup_mask,  cv2.MORPH_CLOSE, h_close)
    deep_mask = cv2.morphologyEx(deep_mask, cv2.MORPH_CLOSE, h_close)

    # Clip to valid column range — prevents MORPH_CLOSE edge bleed
    c0, c1 = int(valid_idx[0]), int(valid_idx[-1]) + 1
    if c0 > 0:
        sup_mask[:, :c0] = 0;  deep_mask[:, :c0] = 0
    if c1 < W:
        sup_mask[:, c1:] = 0;  deep_mask[:, c1:] = 0

    return sup_mask, deep_mask


def prob_to_fascia_centreline(prob: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Single deep-band line — used by LISA / Florence fallback paths."""
    _, deep = prob_to_fascia_two_lines(prob, threshold)
    return deep


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
    if fascia_mask is not None and fascia_mask.max() > 0:
        out[fascia_mask > 0] = FASCIA_COLOR
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

    image_np   = np.array(image)
    image_gray = np.array(image.convert('L'))   # grayscale for anechoic check

    with torch.no_grad():
        # Vein: visual few-shot + text grounding ensemble
        vein_vis_prob  = vlm_visual_infer_prob(image, VEIN_VIS_FEATURES)
        vein_txt_prob  = vlm_text_infer_prob(image, [
            "vein", "blood vessel", "anechoic vessel in ultrasound"
        ])
        vein_mask = prob_to_vein_mask(
            np.maximum(vein_vis_prob, vein_txt_prob),
            threshold=0.40, image_gray=image_gray
        )

        # Fascia: visual few-shot + text grounding ensemble → centreline
        fascia_vis_prob = vlm_visual_infer_prob(image, FASCIA_VIS_FEATURES)
        fascia_txt_prob = vlm_text_infer_prob(image, [
            "fascia", "hyperechoic fascial layer",
            "connective tissue layer", "fascial plane in ultrasound"
        ])
        fascia_prob = np.maximum(fascia_vis_prob, fascia_txt_prob)

    # The model predicts the full fascia zone (superficial to deep line).
    # Top edge of blob = superficial fascia line; bottom edge = deep fascia line.
    h, w = image_np.shape[:2]
    above     = fascia_prob > 0.30
    col_max   = fascia_prob.max(axis=0)
    valid     = col_max > 0.30
    valid_idx = np.where(valid)[0]

    fascia_boundary = np.zeros((h, w), dtype=np.uint8)
    if valid_idx.size >= int(0.40 * w):
        # Superficial line: first row with signal per column (top edge of fascial zone)
        sup_raw  = np.argmax(above, axis=0).astype(np.float64)
        # Deep line: last row with signal per column (bottom edge of fascial zone)
        deep_raw = (h - 1 - np.argmax(above[::-1], axis=0)).astype(np.float64)

        k   = min(255, max(3, w // 4))   # wide kernel for smooth curve
        pad = k // 2
        kernel = np.ones(k) / k
        sup_filled  = np.interp(np.arange(w), valid_idx, sup_raw[valid_idx])
        deep_filled = np.interp(np.arange(w), valid_idx, deep_raw[valid_idx])
        # Two-pass box filter — rounds sharp dips into smooth curves
        for _ in range(2):
            sup_filled  = np.convolve(np.pad(sup_filled,  pad, mode='edge'), kernel, mode='valid')[:w]
            deep_filled = np.convolve(np.pad(deep_filled, pad, mode='edge'), kernel, mode='valid')[:w]

        sup_rows  = np.clip(sup_filled.astype(int),  0, h - 1)
        deep_rows = np.clip(deep_filled.astype(int), 0, h - 1)

        LINE_HALF = 7
        cols = valid_idx
        for dr in range(-LINE_HALF, LINE_HALF + 1):
            fascia_boundary[np.clip(deep_rows[cols] + dr, 0, h-1), cols] = 255
            fascia_boundary[np.clip(sup_rows[cols]  + dr, 0, h-1), cols] = 255

        h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
        fascia_boundary = cv2.morphologyEx(fascia_boundary, cv2.MORPH_CLOSE, h_close)
        c0, c1 = int(valid_idx[0]), int(valid_idx[-1]) + 1
        if c0 > 0: fascia_boundary[:, :c0] = 0
        if c1 < w: fascia_boundary[:, c1:] = 0

    annotated = image_np.copy()
    if fascia_boundary.max() > 0:
        annotated[fascia_boundary > 0] = FASCIA_SUP_COLOR
    if vein_mask.max() > 0:
        # Blur mask slightly before contouring → smooth natural outline, not ellipse
        vein_blur = cv2.GaussianBlur(vein_mask.astype(np.float32) * 255, (0, 0), sigmaX=3)
        vein_smooth_mask = (vein_blur > 100).astype(np.uint8)
        cnts, _ = cv2.findContours(vein_smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, cnts, -1, VEIN_COLOR, 3)

    masks_viz = np.zeros((h, w, 3), dtype=np.uint8)
    if fascia_boundary.max() > 0: masks_viz[fascia_boundary > 0] = FASCIA_SUP_COLOR
    if vein_mask.max() > 0:       masks_viz[vein_mask       > 0] = VEIN_COLOR

    Image.fromarray(annotated).save(output_path)
    return jsonify({
        'output': _arr_to_b64(annotated),
        'masks':  _arr_to_b64(masks_viz),
    })


# ---------------------------------------------------------------------------
# LISA — disabled
# ---------------------------------------------------------------------------

LISA_AVAILABLE = False


@app.route('/lisa')
def lisa_page():
    return render_template('lisa.html', available=False)


@app.route('/predict_lisa', methods=['POST'])
def predict_lisa():
    return jsonify({'error': 'LISA model is disabled.'}), 503


# ---------------------------------------------------------------------------
# Florence-2 — disabled
# ---------------------------------------------------------------------------

FLORENCE_AVAILABLE = False


@app.route('/florence')
def florence_page():
    return render_template('florence.html', available=False)


@app.route('/predict_florence', methods=['POST'])
def predict_florence():
    return jsonify({'error': 'Florence-2 model is disabled.'}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)