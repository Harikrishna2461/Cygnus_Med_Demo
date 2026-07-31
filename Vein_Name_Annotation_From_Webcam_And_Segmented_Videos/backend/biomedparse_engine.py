"""
Loads the two finetuned BioMedParse checkpoints from Task_4_VLM_Fascia_Vein_Detection
(referenced in place, never copied) and runs per-frame vein + fascia segmentation.

Inference logic (grounding_prob, prob_to_vein_mask, prob_to_fascia_two_lines) is ported
from Task_4_VLM_Fascia_Vein_Detection/app.py, which is the proven production path (not
BiomedParse's own stock example_prediction.py wrapper).
"""
import os
import sys
import glob
import threading
import contextlib
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

import config

# --- Make BiomedParse importable without a real detectron2/fvcore/mpi4py install ---
sys.path.insert(0, config.STUBS_DIR)
sys.path.insert(0, config.BIOMEDPARSE_DIR)

from detectron2.structures import ImageList
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES


@contextlib.contextmanager
def _cwd(path):
    """BiomedParse's config has CWD-relative fields (base_path/RESUME_FROM) — Task_4's own
    run.bat 'cd /d' into its project root before running for exactly this reason. Anything
    that reads a path from *our* config must be resolved to an absolute path before entering
    this context, since CWD is not what callers of this module normally expect."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@dataclass
class VeinBlob:
    blob_id: int                       # 1..N within this tick, sorted by (centroid_y, centroid_x)
    contour: np.ndarray                # cv2 contour points, CHAIN_APPROX_SIMPLE
    centroid: tuple                    # (cx, cy) in original-frame pixel coords
    bbox: tuple                        # (x, y, w, h)
    area_px: int
    n_class: str = None                # "N1"|"N2"|"N3" — filled by stage2, never by this module
    n_class_reasoning: str = None


@dataclass
class FasciaBoundary:
    sup_row_at_col: np.ndarray         # float[W], NaN where invalid
    deep_row_at_col: np.ndarray        # float[W], NaN where invalid


_lock = threading.Lock()
_fascia_model = None
_vein_model = None


def _newest_ckpt(ckpt_dir: str) -> str:
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "**", "model_state_dict.pt"), recursive=True),
        key=os.path.getmtime,
    )
    if ckpts:
        return ckpts[-1]
    if os.path.exists(config.LOCAL_FALLBACK_WEIGHTS):
        return config.LOCAL_FALLBACK_WEIGHTS
    return "hf_hub:microsoft/BiomedParse"


def _load_one(ckpt_dir: str):
    opt = load_opt_from_config_files([config.BIOMEDPARSE_CONFIG])
    opt = init_distributed(opt)
    weights = _newest_ckpt(ckpt_dir)
    print(f"[biomedparse_engine] loading weights: {weights}")
    model = BaseModel(opt, build_model(opt)).from_pretrained(weights).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    return model


def get_models():
    """Lazy-loaded singleton pair (fascia_model, vein_model). Thread-safe."""
    global _fascia_model, _vein_model
    if _fascia_model is not None and _vein_model is not None:
        return _fascia_model, _vein_model
    with _lock:
        if _fascia_model is None or _vein_model is None:
            with _cwd(config.TASK4_DIR):
                print("[biomedparse_engine] loading fascia model...")
                _fascia_model = _load_one(config.FASCIA_CKPT_DIR)
                print("[biomedparse_engine] loading vein model...")
                _vein_model = _load_one(config.VEIN_CKPT_DIR)
            print("[biomedparse_engine] both models ready.")
    return _fascia_model, _vein_model


def grounding_prob(model, image_pil: Image.Image, text: str, infer_size: int = 512,
                    top_bias: bool = False) -> np.ndarray:
    """Ported from Task_4_VLM_Fascia_Vein_Detection/app.py::_grounding_prob. Returns a
    dense float32 [0,1] probability map at the original (H, W) of image_pil."""
    m = model.model
    pred = m.sem_seg_head.predictor
    W, H = image_pil.size

    resized = np.asarray(image_pil.resize((infer_size, infer_size), Image.BICUBIC)).astype(np.float32)
    img_t = torch.from_numpy(resized.copy()).permute(2, 0, 1).cuda()
    images = ImageList.from_tensors([(img_t - m.pixel_mean) / m.pixel_std], m.size_divisibility)

    gtext = pred.lang_encoder.get_text_token_embeddings([text], name="grounding", token=False, norm=False)
    tok_emb = gtext["token_emb"]
    tok_mask = gtext["tokens"]["attention_mask"].bool()
    q_emb = tok_emb[tok_mask]
    nz_mask = torch.zeros(q_emb[:, None].shape[:-1], dtype=torch.bool, device=q_emb.device)
    extra = {
        "grounding_tokens": q_emb[:, None],
        "grounding_nonzero_mask": nz_mask.t(),
        "grounding_class": gtext["class_emb"],
    }
    with torch.no_grad():
        feats = m.backbone(images.tensor)
        mf, _, ms = m.sem_seg_head.pixel_decoder.forward_features(feats)
        outputs = pred(ms, mf, extra=extra, task="grounding_eval")

    all_gm = outputs["pred_gmasks"][0]
    probs = torch.sigmoid(all_gm)
    if top_bias:
        Hm = probs.shape[1]
        vert_weight = torch.linspace(2.0, 0.2, Hm, device=probs.device).view(1, Hm, 1)
        weighted = (probs * vert_weight).reshape(101, -1).max(dim=1).values
    else:
        weighted = probs.reshape(101, -1).max(dim=1).values
    best_q = weighted.argmax().item()
    return F.interpolate(
        all_gm[best_q:best_q + 1][None], (H, W), mode="bilinear", align_corners=False,
    )[0, 0].sigmoid().detach().cpu().numpy().astype(np.float32)


def prob_to_vein_mask(prob: np.ndarray, image_gray: np.ndarray = None,
                       threshold: float = config.VEIN_PROB_THRESHOLD) -> np.ndarray:
    """Ported verbatim from Task_4_VLM_Fascia_Vein_Detection/app.py::prob_to_vein_mask.
    Keeps only blobs that look like real vein cross-sections: small, anechoic, not
    elongated, not irregular. This filtering is tuned to this segmentation model's
    output characteristics, not a medical classification rule."""
    binary = (prob > threshold).astype(np.uint8)
    total = prob.shape[0] * prob.shape[1]
    min_area = max(10, int(config.VEIN_MIN_AREA_FRAC * total))
    max_area = int(config.VEIN_MAX_AREA_FRAC * total)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        if image_gray is not None:
            mean_val = float(image_gray[labels == i].mean())
            if mean_val > config.VEIN_MAX_ANECHOIC_MEAN:
                continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw >= 1 and bh >= 1 and max(bw, bh) / min(bw, bh) > config.VEIN_MAX_ASPECT_RATIO:
            continue
        mask_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            perim = cv2.arcLength(cnts[0], True)
            if perim > 0 and (4 * np.pi * area / perim ** 2) < config.VEIN_MIN_CIRCULARITY:
                continue
        out[labels == i] = 1
    return out


def prob_to_fascia_two_lines(prob: np.ndarray, threshold: float = config.FASCIA_PROB_THRESHOLD):
    """Ported from Task_4_VLM_Fascia_Vein_Detection/fascia_helpers.py::prob_to_fascia_two_lines,
    extended to also return the per-column row-index arrays (not just the rasterized mask) —
    Stage 2's geometric hints need numeric row positions, not just pixels to draw.
    Returns (sup_mask, deep_mask, sup_row_at_col, deep_row_at_col)."""
    H, W = prob.shape
    LINE_HALF = 6

    col_max = prob.max(axis=0)
    valid = col_max > threshold
    if valid.sum() < int(0.40 * W):
        nan_row = np.full(W, np.nan)
        return np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8), nan_row, nan_row.copy()

    above = prob > threshold
    sup_raw = np.argmax(above, axis=0).astype(np.float64)
    deep_raw = (H - 1 - np.argmax(above[::-1], axis=0)).astype(np.float64)

    valid_idx = np.where(valid)[0]
    sup_filled = np.interp(np.arange(W), valid_idx, sup_raw[valid_idx])
    deep_filled = np.interp(np.arange(W), valid_idx, deep_raw[valid_idx])

    k = min(63, max(3, W // 16))
    pad = k // 2
    kernel = np.ones(k) / k
    sup_smooth = np.convolve(np.pad(sup_filled, pad, mode="edge"), kernel, mode="valid")[:W]
    deep_smooth = np.convolve(np.pad(deep_filled, pad, mode="edge"), kernel, mode="valid")[:W]

    sup_rows = np.clip(sup_smooth.astype(int), 0, H - 1)
    deep_rows = np.clip(deep_smooth.astype(int), 0, H - 1)

    sup_mask = np.zeros((H, W), dtype=np.uint8)
    deep_mask = np.zeros((H, W), dtype=np.uint8)
    cols = valid_idx
    for dr in range(-LINE_HALF, LINE_HALF + 1):
        sup_mask[np.clip(sup_rows[cols] + dr, 0, H - 1), cols] = 255
        deep_mask[np.clip(deep_rows[cols] + dr, 0, H - 1), cols] = 255

    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    sup_mask = cv2.morphologyEx(sup_mask, cv2.MORPH_CLOSE, h_close)
    deep_mask = cv2.morphologyEx(deep_mask, cv2.MORPH_CLOSE, h_close)

    c0, c1 = int(valid_idx[0]), int(valid_idx[-1]) + 1
    if c0 > 0:
        sup_mask[:, :c0] = 0
        deep_mask[:, :c0] = 0
    if c1 < W:
        sup_mask[:, c1:] = 0
        deep_mask[:, c1:] = 0

    sup_row_at_col = np.full(W, np.nan)
    deep_row_at_col = np.full(W, np.nan)
    sup_row_at_col[c0:c1] = sup_smooth[c0:c1]
    deep_row_at_col[c0:c1] = deep_smooth[c0:c1]

    return sup_mask, deep_mask, sup_row_at_col, deep_row_at_col


def _blobs_from_mask(mask: np.ndarray) -> list[VeinBlob]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        bbox = cv2.boundingRect(c)
        raw.append((cy, cx, c, (cx, cy), bbox, int(cv2.contourArea(c))))
    raw.sort(key=lambda t: (t[0], t[1]))  # deterministic tick-local numbering: top-to-bottom, then left-to-right
    return [
        VeinBlob(blob_id=i, contour=c, centroid=centroid, bbox=bbox, area_px=area)
        for i, (_, _, c, centroid, bbox, area) in enumerate(raw, start=1)
    ]


def segment_frame(frame_bgr: np.ndarray):
    """Run both models on one OpenCV (BGR) frame. Returns (list[VeinBlob], FasciaBoundary)."""
    fascia_model, vein_model = get_models()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    fascia_prob = grounding_prob(fascia_model, image_pil, config.FASCIA_PROMPT, config.INFER_SIZE, top_bias=True)
    vein_prob = grounding_prob(vein_model, image_pil, config.VEIN_PROMPT, config.INFER_SIZE, top_bias=False)

    sup_mask, deep_mask, sup_rows, deep_rows = prob_to_fascia_two_lines(fascia_prob)
    vein_mask = prob_to_vein_mask(vein_prob, image_gray=gray)

    blobs = _blobs_from_mask(vein_mask)
    fascia = FasciaBoundary(sup_row_at_col=sup_rows, deep_row_at_col=deep_rows)
    return blobs, fascia
