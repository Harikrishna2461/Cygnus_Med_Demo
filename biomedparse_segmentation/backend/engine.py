"""
Vein-only BiomedParse segmentation engine.

Vendored/adapted from Task_4_VLM_Fascia_Vein_Detection/app.py — keeps only the
vein model + vein postprocessing path (no fascia, no LISA/Florence, no Groq
blob-evaluator by default). See memory note `biomedparse_finetuned_model` for
the provenance of these functions.
"""
import glob as _glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # biomedparse_segmentation/
sys.path.insert(0, os.path.join(BASE_DIR, 'stubs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'BiomedParse'))

from detectron2.structures import ImageList
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

BIOMEDPARSE_DIR = os.path.join(BASE_DIR, 'BiomedParse')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints', 'vein')

VEIN_PROMPT = (
    'small oval anechoic dark void vein lumen in cross-section '
    'peripheral vascular ultrasound below fascia'
)
VEIN_COLOR = (0, 210, 0)  # green, RGB

_model = None


def load_model():
    """Load the finetuned vein-segmentation BiomedParse model once. GPU required."""
    global _model
    if _model is not None:
        return _model

    print("[engine] Loading vein model...")
    opt = load_opt_from_config_files([os.path.join(BIOMEDPARSE_DIR, 'configs', 'biomed_fascia_finetuning.yaml')])
    opt = init_distributed(opt)

    ckpts = sorted(_glob.glob(os.path.join(CKPT_DIR, '**', 'model_state_dict.pt'), recursive=True),
                    key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"No vein checkpoint found under {CKPT_DIR}")
    weights = ckpts[-1]
    print(f"[engine]   weights: {weights}")

    model = BaseModel(opt, build_model(opt)).from_pretrained(weights).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    print("[engine] Vein model loaded.")
    _model = model
    return _model


def _grounding_prob(mdl, query_image_pil, text, infer_size=512):
    """Grounding inference. Returns float32 [H,W] probability map in [0,1]."""
    m = mdl.model
    pred = m.sem_seg_head.predictor
    W, H = query_image_pil.size

    resized = np.asarray(query_image_pil.resize((infer_size, infer_size), Image.BICUBIC)).astype(np.float32)
    img_t = torch.from_numpy(resized.copy()).permute(2, 0, 1).cuda()
    images = ImageList.from_tensors(
        [(img_t - m.pixel_mean) / m.pixel_std], m.size_divisibility
    )
    gtext = pred.lang_encoder.get_text_token_embeddings(
        [text], name='grounding', token=False, norm=False
    )
    tok_emb = gtext['token_emb']
    tok_mask = gtext['tokens']['attention_mask'].bool()
    q_emb = tok_emb[tok_mask]
    nz_mask = torch.zeros(q_emb[:, None].shape[:-1], dtype=torch.bool, device=q_emb.device)
    extra = {
        'grounding_tokens': q_emb[:, None],
        'grounding_nonzero_mask': nz_mask.t(),
        'grounding_class': gtext['class_emb'],
    }

    with torch.no_grad():
        feats = m.backbone(images.tensor)
        mf, _, ms = m.sem_seg_head.pixel_decoder.forward_features(feats)
        outputs = pred(ms, mf, extra=extra, task='grounding_eval')

    all_gm = outputs['pred_gmasks'][0]
    probs = torch.sigmoid(all_gm)
    weighted = probs.reshape(101, -1).max(dim=1).values
    best_q = weighted.argmax().item()
    return F.interpolate(
        all_gm[best_q:best_q + 1][None], (H, W), mode='bilinear', align_corners=False,
    )[0, 0].sigmoid().detach().cpu().numpy().astype(np.float32)


def prob_to_vein_mask(prob: np.ndarray, threshold: float = 0.5, image_gray: np.ndarray = None) -> np.ndarray:
    """
    Keep only blobs that look like real vein cross-sections:
      - small (0.02-2.5% of image)
      - anechoic (mean pixel < 65)
      - not wildly elongated (aspect ratio <= 4)
      - not completely irregular (circularity >= 0.15)
    Returns a uint8 {0,1} mask.
    """
    binary = (prob > threshold).astype(np.uint8)
    total = prob.shape[0] * prob.shape[1]
    min_area = max(10, int(0.0002 * total))
    max_area = int(0.025 * total)
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
            continue
        mask_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            perim = cv2.arcLength(cnts[0], True)
            if perim > 0 and (4 * np.pi * area / perim ** 2) < 0.15:
                continue
        out[labels == i] = 1
    return out


def segment_frame(model, image_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Segment veins in a single RGB frame (numpy uint8 [H,W,3]).
    Returns a uint8 {0,1} mask, same H,W as input.
    """
    h, w = image_rgb.shape[:2]
    pil_img = Image.fromarray(image_rgb)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    with torch.no_grad():
        prob = _grounding_prob(model, pil_img, text=VEIN_PROMPT)
    prob = cv2.resize(prob, (w, h))
    mask = prob_to_vein_mask(prob, threshold=threshold, image_gray=gray)
    return mask


def draw_vein_contours(image_rgb: np.ndarray, vein_mask: np.ndarray) -> np.ndarray:
    """Draw smoothed green outline of the vein mask onto a copy of the frame."""
    out = image_rgb.copy()
    if vein_mask.max() > 0:
        vein_blur = cv2.GaussianBlur(vein_mask.astype(np.float32) * 255, (0, 0), sigmaX=3)
        vein_smooth = (vein_blur > 100).astype(np.uint8)
        cnts, _ = cv2.findContours(vein_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, VEIN_COLOR, 3)
    return out
