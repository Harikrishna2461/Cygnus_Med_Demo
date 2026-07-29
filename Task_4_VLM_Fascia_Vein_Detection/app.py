import base64
import concurrent.futures
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
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'stubs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'BiomedParse'))
sys.path.insert(0, os.path.join(BASE_DIR, 'LISA'))

from detectron2.structures import ImageList
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
import glob as _glob
BIOMEDPARSE_DIR = os.path.join(BASE_DIR, 'BiomedParse')

# Fascia model — fascia-only fine-tuned weights (proven working)
print("Loading fascia model...")
fascia_opt = load_opt_from_config_files([os.path.join(BIOMEDPARSE_DIR, 'configs', 'biomed_fascia_finetuning.yaml')])
fascia_opt = init_distributed(fascia_opt)
_ckpt_dir = os.path.join(BASE_DIR, 'BiomedParse', 'output', 'fascia_finetuning_v2_production')
_ckpts = sorted(_glob.glob(os.path.join(_ckpt_dir, '**/model_state_dict.pt'), recursive=True),
                key=os.path.getmtime)
LOCAL_WEIGHTS = os.path.join(BASE_DIR, 'pretrained', 'biomedparse_v1.pt')
fascia_weights = (_ckpts[-1] if _ckpts else None) or (LOCAL_WEIGHTS if os.path.exists(LOCAL_WEIGHTS) else 'hf_hub:microsoft/BiomedParse')
print(f"  fascia weights: {fascia_weights}")
model = BaseModel(fascia_opt, build_model(fascia_opt)).from_pretrained(fascia_weights).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"], is_eval=True
    )
print("Fascia model loaded.")

# Vein model — fascia+vein fine-tuned weights
print("Loading vein model...")
vein_opt = load_opt_from_config_files([os.path.join(BIOMEDPARSE_DIR, 'configs', 'biomed_fascia_finetuning.yaml')])
vein_opt = init_distributed(vein_opt)
_ckpt_dir = os.path.join(BASE_DIR, 'BiomedParse', 'output', 'fascia_vein_finetuning')
_ckpts = sorted(_glob.glob(os.path.join(_ckpt_dir, '**/model_state_dict.pt'), recursive=True),
                key=os.path.getmtime)
vein_weights = (_ckpts[-1] if _ckpts else None) or (LOCAL_WEIGHTS if os.path.exists(LOCAL_WEIGHTS) else 'hf_hub:microsoft/BiomedParse')
print(f"  vein weights: {vein_weights}")
vein_model = BaseModel(vein_opt, build_model(vein_opt)).from_pretrained(vein_weights).eval().cuda()
with torch.no_grad():
    vein_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"], is_eval=True
    )
print("Vein model loaded.")

# Image transform matching BiomedParse training (1024x1024 bicubic)
_img_transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)
])

VEIN_COLOR             = (0, 210, 0)    # green
FASCIA_SUP_COLOR       = (0, 230, 230)  # cyan  — superficial (skin-fascia interface)
FASCIA_DEEP_COLOR      = (0, 160, 230)  # blue  — deep (fascia-muscle interface)
FASCIA_COLOR           = FASCIA_SUP_COLOR  # legacy alias

# ---------------------------------------------------------------------------
# Groq evaluator — verifies each vein candidate blob using llama-4-scout vision
# ---------------------------------------------------------------------------
_GROQ_API_KEY      = ""
_GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
_groq_client = Groq(api_key=_GROQ_API_KEY)

_EVALUATOR_SYSTEM = (
    "You are a peripheral vascular sonographer reviewing a full B-mode ultrasound frame. "
    "One candidate structure is marked with a GREEN outline. "
    "Decide whether it is a real vein — which can appear anywhere: above the fascia (superficial veins), "
    "at the fascia, or deep inside the muscle compartment (deep veins like femoral or popliteal).\n\n"
    "A real vein has ALL of these:\n"
    "  - DISCRETE BOUNDARY — a clear, well-defined wall separating it from surrounding tissue\n"
    "  - DARK INTERIOR — anechoic or hypoechoic lumen (clearly darker than surrounding muscle speckle)\n"
    "  - COMPACT SHAPE — oval or round, not sprawling or irregular\n\n"
    "Say NO only if the structure is CLEARLY:\n"
    "  - Muscle tissue with internal speckle (not truly dark inside)\n"
    "  - A diffuse irregular blob with no clean boundary\n"
    "  - A horizontal band or sheet (fascia or artifact, not a vessel)\n\n"
    "When uncertain, say YES. Reply with exactly one word: YES or NO."
)


def _full_img_b64_highlighted(image_rgb: np.ndarray, blob_mask: np.ndarray) -> str:
    """Draw a bright green outline on the full image so the model sees full anatomical context."""
    annotated = image_rgb.copy()
    contours, _ = cv2.findContours(blob_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 3)
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _evaluate_blob(image_rgb: np.ndarray, blob_mask: np.ndarray) -> bool:
    """Returns True if Groq vision model confirms this blob is a real vein."""
    try:
        b64 = _full_img_b64_highlighted(image_rgb, blob_mask)
        resp = _groq_client.chat.completions.create(
            model=_GROQ_VISION_MODEL,
            messages=[
                {"role": "system", "content": _EVALUATOR_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text",
                     "text": (
                         "The GREEN outlined structure is the candidate. "
                         "Is it a vein? YES or NO. /no_think"
                     )},
                ]},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ''
        import re as _re
        clean = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL | _re.IGNORECASE).strip()
        answer = (clean.split()[0] if clean.split() else raw).upper()
        print(f"[evaluator] answer={answer!r}")
        return answer.startswith('YES')
    except Exception as e:
        print(f"[evaluator] error: {e}")
        return True


def evaluate_vein_mask(vein_mask: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """
    Split the model's vein prediction into individual blobs, evaluate each in
    parallel with Groq LLM, return a mask containing only confirmed veins.
    """
    n, labels, _, _ = cv2.connectedComponentsWithStats(vein_mask, connectivity=8)
    if n <= 1:
        return vein_mask

    h, w    = vein_mask.shape
    min_px  = max(50, int(0.0003 * h * w))   # ignore sub-50px specks — not anatomically meaningful
    blob_ids = [i for i in range(1, n) if int((labels == i).sum()) >= min_px]

    if not blob_ids:
        return np.zeros_like(vein_mask)

    def _check(i):
        return i, _evaluate_blob(image_rgb, (labels == i))

    verified = np.zeros_like(vein_mask)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(blob_ids))) as ex:
        for blob_id, is_vein in ex.map(_check, blob_ids):
            if is_vein:
                verified[labels == blob_id] = 1
    return verified

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
        )[0, 0, :H, :W].sigmoid().detach().cpu().numpy()

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
        )[0, 0, :H, :W].sigmoid().detach().cpu().numpy()
        best_prob = np.maximum(best_prob, prob.astype(np.float32))

    return best_prob


VEIN_PROMPT = (
    'small oval anechoic dark void vein lumen in cross-section '
    'peripheral vascular ultrasound below fascia'
)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _apply_clahe(arr_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(_clahe.apply(gray), cv2.COLOR_GRAY2RGB)


def _grounding_prob(mdl, query_image_pil, text, infer_size=512, top_bias=False, use_clahe=False):
    """Grounding inference on a given model instance. Returns float32 prob map."""
    m    = mdl.model
    pred = m.sem_seg_head.predictor
    W, H = query_image_pil.size

    resized = np.asarray(query_image_pil.resize((infer_size, infer_size), Image.BICUBIC))
    arr     = (_apply_clahe(resized) if use_clahe else resized).astype(np.float32)
    img_t   = torch.from_numpy(arr.copy()).permute(2, 0, 1).cuda()
    images = ImageList.from_tensors(
        [(img_t - m.pixel_mean) / m.pixel_std], m.size_divisibility
    )
    gtext    = pred.lang_encoder.get_text_token_embeddings(
        [text], name='grounding', token=False, norm=False
    )
    tok_emb  = gtext['token_emb']
    tok_mask = gtext['tokens']['attention_mask'].bool()
    q_emb    = tok_emb[tok_mask]
    nz_mask  = torch.zeros(q_emb[:, None].shape[:-1], dtype=torch.bool, device=q_emb.device)
    extra = {
        'grounding_tokens':       q_emb[:, None],
        'grounding_nonzero_mask': nz_mask.t(),
        'grounding_class':        gtext['class_emb'],
    }
    with torch.no_grad():
        feats     = m.backbone(images.tensor)
        mf, _, ms = m.sem_seg_head.pixel_decoder.forward_features(feats)
        outputs   = pred(ms, mf, extra=extra, task='grounding_eval')

    all_gm = outputs['pred_gmasks'][0]
    probs  = torch.sigmoid(all_gm)
    if top_bias:
        Hm = probs.shape[1]
        vert_weight = torch.linspace(2.0, 0.2, Hm, device=probs.device).view(1, Hm, 1)
        weighted = (probs * vert_weight).reshape(101, -1).max(dim=1).values
    else:
        weighted = probs.reshape(101, -1).max(dim=1).values
    best_q = weighted.argmax().item()
    return F.interpolate(
        all_gm[best_q:best_q + 1][None], (H, W), mode='bilinear', align_corners=False,
    )[0, 0].sigmoid().detach().cpu().numpy().astype(np.float32)


def prob_to_vein_mask(prob: np.ndarray, threshold: float = 0.25,
                      image_gray: np.ndarray = None) -> np.ndarray:
    """
    Keep only blobs that look like real vein cross-sections:
      - small (0.02–2.5 % of image)
      - anechoic (mean pixel < 65)
      - not wildly elongated (aspect ratio ≤ 4)
      - not completely irregular (circularity ≥ 0.15)
    """
    binary = (prob > threshold).astype(np.uint8)
    total  = prob.shape[0] * prob.shape[1]
    min_area = max(10, int(0.0002 * total))
    max_area = int(0.025 * total)   # 2.5 % max — veins are small
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        if image_gray is not None:
            mean_val = float(image_gray[labels == i].mean())
            if mean_val > 65:
                continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw >= 1 and bh >= 1 and max(bw, bh) / min(bw, bh) > 4.0:
            continue   # too elongated — not a vein cross-section
        mask_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            perim = cv2.arcLength(cnts[0], True)
            if perim > 0 and (4 * np.pi * area / perim ** 2) < 0.15:
                continue   # too irregular
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
    for _, cnt in candidates[:5]:
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

    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Image open failed: {e}'}), 400
    image.save(input_path)

    image_np = np.array(image)
    h_orig, w_orig = image_np.shape[:2]

    try:
        use_clahe     = request.form.get('use_clahe', '0') == '1'
        use_evaluator = request.form.get('use_evaluator', '1') == '1'
        with torch.no_grad():
            fascia_prob = _grounding_prob(
                model, image,
                text='fascia layer in PeripheralVascular Ultrasound',
                top_bias=True,
                use_clahe=use_clahe,
            )
            vein_prob = _grounding_prob(
                vein_model, image,
                text=VEIN_PROMPT,
                top_bias=False,
                use_clahe=use_clahe,
            )
        fascia_prob = cv2.resize(fascia_prob, (w_orig, h_orig))
        vein_prob   = cv2.resize(vein_prob,   (w_orig, h_orig))
        vein_mask_raw = (vein_prob > 0.5).astype(np.uint8)
        vein_mask     = evaluate_vein_mask(vein_mask_raw, image_np) if use_evaluator else vein_mask_raw
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    h, w = image_np.shape[:2]

    # Two thin smooth lines: top edge (superficial) + bottom edge (deep) of fascia zone
    fascia_prob_smooth = cv2.GaussianBlur(fascia_prob, (0, 0), sigmaX=5)
    above_half = np.zeros_like(fascia_prob_smooth)
    above_half[:h//2, :] = fascia_prob_smooth[:h//2, :]
    above     = above_half > 0.35
    col_max   = fascia_prob_smooth[:h//2, :].max(axis=0)
    valid     = col_max > 0.35
    valid_idx = np.where(valid)[0]

    fascia_mask = np.zeros((h, w), dtype=np.uint8)
    if valid_idx.size >= int(0.30 * w):
        sup_raw  = np.argmax(above, axis=0).astype(np.float64)
        deep_raw = (h - 1 - np.argmax(above[::-1], axis=0)).astype(np.float64)

        sup_filled  = np.interp(np.arange(w), valid_idx, sup_raw[valid_idx])
        deep_filled = np.interp(np.arange(w), valid_idx, deep_raw[valid_idx])

        # Three passes of wide box filter for very smooth curves
        k = min(127, max(5, w // 8))
        pad = k // 2
        kernel = np.ones(k) / k
        for _ in range(3):
            sup_filled  = np.convolve(np.pad(sup_filled,  pad, mode='edge'), kernel, mode='valid')[:w]
            deep_filled = np.convolve(np.pad(deep_filled, pad, mode='edge'), kernel, mode='valid')[:w]

        sup_rows  = np.clip(sup_filled.astype(int) - 8, 0, h - 1)
        deep_rows = np.clip(deep_filled.astype(int), 0, h - 1)

        LINE_HALF = 2   # 5px total per line
        cols = valid_idx
        for dr in range(-LINE_HALF, LINE_HALF + 1):
            fascia_mask[np.clip(sup_rows[cols]  + dr, 0, h - 1), cols] = 255
            fascia_mask[np.clip(deep_rows[cols] + dr, 0, h - 1), cols] = 255

        h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        fascia_mask = cv2.morphologyEx(fascia_mask, cv2.MORPH_CLOSE, h_close)
        c0, c1 = int(valid_idx[0]), int(valid_idx[-1]) + 1
        if c0 > 0: fascia_mask[:, :c0] = 0
        if c1 < w: fascia_mask[:, c1:] = 0

    annotated = image_np.copy()
    if fascia_mask.max() > 0:
        annotated[fascia_mask > 0] = FASCIA_SUP_COLOR
    if vein_mask.max() > 0:
        # Blur mask slightly before contouring → smooth natural outline, not ellipse
        vein_blur = cv2.GaussianBlur(vein_mask.astype(np.float32) * 255, (0, 0), sigmaX=3)
        vein_smooth_mask = (vein_blur > 100).astype(np.uint8)
        cnts, _ = cv2.findContours(vein_smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, cnts, -1, VEIN_COLOR, 3)

    masks_viz = np.zeros((h, w, 3), dtype=np.uint8)
    if fascia_mask.max() > 0: masks_viz[fascia_mask > 0] = FASCIA_SUP_COLOR
    if vein_mask.max() > 0:           masks_viz[vein_mask           > 0] = VEIN_COLOR

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