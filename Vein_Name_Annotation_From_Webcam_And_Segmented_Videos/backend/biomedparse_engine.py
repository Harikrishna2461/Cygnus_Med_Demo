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
    is_valid: bool = True              # False if stage2 judges this a non-anatomical false
                                        # positive (text/watermark/logo, not a real vein) —
                                        # e.g. a recording-software watermark burned into the
                                        # video, which ROI cropping can't remove since it can
                                        # sit inside the scan region itself, not just the
                                        # surrounding machine-UI chrome ROI cropping targets


@dataclass
class FasciaBoundary:
    sup_row_at_col: np.ndarray         # float[W], NaN where invalid
    deep_row_at_col: np.ndarray        # float[W], NaN where invalid


_lock = threading.Lock()
_fascia_model = None
_vein_model = None

# Guards every forward pass through either model, not just loading. BiomedParse's
# AttentionDataStruct (modeling/interface/prototype/attention_data_struct_seemv1.py)
# mutates shared in-place state (self.query_index, self.attn_variables, ...) across the
# reset() -> set() -> cross_attn_variables() -> update_variables() sequence inside one
# forward() call. If a second thread's forward() call interleaves with that sequence
# (e.g. two jobs/videos being processed concurrently -- jobs.py starts one unthrottled
# thread per job with no concurrency cap), one call's reset() can wipe or a stale
# cross_attn_name list can outlive the query_index it was paired with, producing
# intermittent `KeyError: 'queries_object'`-style crashes deep in vendor code. Confirmed
# empirically: the exact same 120s video that crashed under the running Flask server
# processed clean end-to-end (220+ ticks) when run standalone/single-threaded -- not a
# per-frame data bug, a concurrency one. Model inference was never going to be safely
# concurrent on one shared instance anyway (also avoids two videos fighting over VRAM).
_infer_lock = threading.Lock()


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
    """Ported from Task_4_VLM_Fascia_Vein_Detection/app.py::_grounding_prob, with one
    deliberate change: letterbox (aspect-ratio-preserving resize + pad) instead of a
    direct non-uniform stretch to infer_size x infer_size.

    Confirmed the actual fascia/vein fine-tuning data pipeline (BiomedParse/datasets/
    dataset_mappers/biomed_dataset_mapper.py::build_transform_gen, driven by
    configs/biomed_fascia_finetuning.yaml's MIN/MAX_SIZE_TRAIN=460/560, IMAGE_SIZE=512)
    uses detectron2's T.ResizeScale + T.FixedSizeCrop -- aspect-preserving resize to fit
    within 512x512, then pad (or crop, only if training's 0.9-1.1x scale jitter pushed it
    over) to exactly 512x512. This is NOT what app.py's direct `.resize((512,512))` stretch
    does, and app.py's stretch is what got ported here originally. Real ROI-cropped frames
    are often distinctly non-square (confirmed: 364x598, aspect ~0.61) so this mismatch is
    not cosmetic -- it measurably distorts proportions the model never trained on.

    Matches detectron2's FixedSizeCrop specifically: padding added only to the right/
    bottom (content anchored top-left, not centered -- confirmed via CropTransform/
    PadTransform's (0,0) origin convention), with the same pad_value=128.0 (mid-gray)
    default, not black -- chosen in detectron2 to be visually distinct from real content,
    whereas black would blend into ultrasound's naturally near-black background regions
    and read as "more background" rather than "padding" to the model. No real detectron2
    install is available on this machine to read its source directly (stubbed, see
    Task_4_VLM_Fascia_Vein_Detection/stubs/), so this is reconstructed from detectron2's
    public API/documented defaults, not verified byte-for-byte -- flagging that
    explicitly rather than presenting it as fully confirmed.

    Returns a dense float32 [0,1] probability map at the original (H, W) of image_pil.
    """
    m = model.model
    pred = m.sem_seg_head.predictor
    W, H = image_pil.size

    scale = min(infer_size / W, infer_size / H)
    new_w, new_h = max(1, round(W * scale)), max(1, round(H * scale))
    resized_content = image_pil.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (infer_size, infer_size), (128, 128, 128))  # detectron2 FixedSizeCrop default
    canvas.paste(resized_content, (0, 0))

    arr = np.asarray(canvas).astype(np.float32)
    img_t = torch.from_numpy(arr.copy()).permute(2, 0, 1).cuda()
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

    # Upsample to canvas size, crop out the padding, then resize the real-content
    # region up to the true original (H, W) -- the inverse of the letterbox above.
    canvas_pred = F.interpolate(
        all_gm[best_q:best_q + 1][None], (infer_size, infer_size), mode="bilinear", align_corners=False,
    )[0, 0]
    content_pred = canvas_pred[:new_h, :new_w]
    final = F.interpolate(
        content_pred[None, None], (H, W), mode="bilinear", align_corners=False,
    )[0, 0].sigmoid().detach().cpu().numpy().astype(np.float32)
    return final


def prob_to_vein_mask(prob: np.ndarray, image_gray: np.ndarray = None,
                       threshold: float = config.VEIN_PROB_THRESHOLD) -> np.ndarray:
    """Ported from Task_4_VLM_Fascia_Vein_Detection/app.py::prob_to_vein_mask (one change
    below). Keeps only blobs that look like real vein cross-sections: small, anechoic, not
    elongated, not irregular. This filtering is tuned to this segmentation model's output
    characteristics, not a medical classification rule.

    Area bounds are fractions of a FIXED reference pixel count (config.VEIN_AREA_REFERENCE_PX,
    Task_4's own ~802x805 validated test-frame size), not of prob.shape itself. Confirmed on
    real reference footage: after ROI-cropping to a much smaller frame (e.g. 364x598), the
    same real vein covers a much larger *fraction* of the smaller frame, so a fraction-of-
    current-image cap was rejecting genuinely large, obvious veins as "too big" purely
    because the frame got tighter — not because the vein itself was unusual.
    """
    binary = (prob > threshold).astype(np.uint8)
    total = config.VEIN_AREA_REFERENCE_PX
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


def _fit_fascia_curve(valid_cols: np.ndarray, raw_rows: np.ndarray, W: int) -> np.ndarray:
    """Least-squares polynomial fit (degree config.FASCIA_POLY_DEGREE) through the raw
    per-column fascia row readings, evaluated at every column 0..W-1.

    Replaces a moving-average smoothing pass that still left visible local wiggle in the
    line (a boxcar filter only damps noise within its own window, it doesn't produce a
    mathematically smooth curve). A single global polynomial has no per-point jaggedness
    by construction — fascia is a large-scale gently-curving anatomical plane, not a
    high-frequency signal, so a low-degree fit is the right amount of flexibility:
    degree 1 (straight line) is too rigid for the genuine curvature fascia can have along
    a frame's width; degree 3 (cubic) comfortably follows a gentle asymmetric hump/S-curve
    without the overfitting risk of a much higher degree chasing pixel-level noise.
    Falls back to a lower degree (or a flat mean) automatically if there simply aren't
    enough valid points to support a stable cubic fit.
    """
    if len(valid_cols) == 0:
        return np.zeros(W)
    degree = min(config.FASCIA_POLY_DEGREE, len(valid_cols) - 1)
    if degree < 1:
        return np.full(W, raw_rows.mean())
    coeffs = np.polyfit(valid_cols, raw_rows, degree)
    return np.polyval(coeffs, np.arange(W))


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
    sup_smooth = _fit_fascia_curve(valid_idx, sup_raw[valid_idx], W)
    deep_smooth = _fit_fascia_curve(valid_idx, deep_raw[valid_idx], W)

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

    with _infer_lock:
        fascia_prob = grounding_prob(fascia_model, image_pil, config.FASCIA_PROMPT, config.INFER_SIZE, top_bias=True)
        vein_prob = grounding_prob(vein_model, image_pil, config.VEIN_PROMPT, config.INFER_SIZE, top_bias=False)

    sup_mask, deep_mask, sup_rows, deep_rows = prob_to_fascia_two_lines(fascia_prob)
    vein_mask = prob_to_vein_mask(vein_prob, image_gray=gray)

    blobs = _blobs_from_mask(vein_mask)
    fascia = FasciaBoundary(sup_row_at_col=sup_rows, deep_row_at_col=deep_rows)
    return blobs, fascia
