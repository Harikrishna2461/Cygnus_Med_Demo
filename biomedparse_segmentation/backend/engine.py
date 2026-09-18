"""
Vein-only BiomedParse segmentation engine.

Ported EXACTLY from Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/
backend/biomedparse_engine.py (the vein half only — fascia dropped, not
needed here) — not from Task_4_VLM_Fascia_Vein_Detection/app.py, and with
no tuning/modification of our own on top of that source. That project
already diagnosed and fixed the two real bugs Task_4's version had:
  1. `.resize((512,512))` stretches a non-square ROI crop, distorting
     proportions the model never trained on. Fixed with a letterboxed
     resize (aspect-preserving + mid-gray pad) matching BiomedParse's own
     training-time transform (detectron2 ResizeScale + FixedSizeCrop).
  2. `prob_to_vein_mask`'s area filter used fractions of the CURRENT
     image's pixel count. After ROI-cropping to a smaller frame, the same
     real vein covers a much larger fraction of that smaller frame, so a
     fraction-of-current-image cap was rejecting genuinely large, obvious
     veins as "too big" purely because the frame got tighter. Fixed by
     making the area bounds fractions of a FIXED reference pixel count
     (Task_4's own validated ~802x805 test-frame size) instead.
No Groq/LLM blob-verification step — classical CV filtering only, so this
stays fast and has no external API dependency or rate limits.

A prior version of this file added its own top-K query ensemble and
tightened circularity/aspect-ratio thresholds on top of this ported
pipeline. That made results worse, not better (tighter circularity
rejected genuinely round veins; the ensemble let more noise through) and
was reverted — this file is intentionally a 1:1 port, nothing more.
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
INFER_SIZE = 512

# Vein blob filtering constants — ported verbatim from
# Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend/config.py.
# Do not retune these without a specific request; see module docstring.
VEIN_PROB_THRESHOLD = 0.25
VEIN_MIN_AREA_FRAC = 0.0002
VEIN_MAX_AREA_FRAC = 0.025
VEIN_MAX_ASPECT_RATIO = 4.0
VEIN_MIN_CIRCULARITY = 0.15
VEIN_MAX_ANECHOIC_MEAN = 65.0
VEIN_AREA_REFERENCE_PX = 802 * 805  # fixed reference — see module docstring

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


def grounding_prob(model, image_pil: Image.Image, text: str, infer_size: int = INFER_SIZE) -> np.ndarray:
    """
    Letterboxed grounding inference: aspect-ratio-preserving resize to fit
    within infer_size x infer_size, padded with mid-gray (128) to a square
    canvas (content anchored top-left) — matches BiomedParse's own
    training-time transform (detectron2 ResizeScale + FixedSizeCrop),
    unlike a naive `.resize((infer_size, infer_size))` stretch which
    distorts non-square ROI crops.

    Returns a dense float32 [0,1] probability map at the original (H, W) of image_pil.
    """
    m = model.model
    pred = m.sem_seg_head.predictor
    W, H = image_pil.size

    scale = min(infer_size / W, infer_size / H)
    new_w, new_h = max(1, round(W * scale)), max(1, round(H * scale))
    resized_content = image_pil.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (infer_size, infer_size), (128, 128, 128))
    canvas.paste(resized_content, (0, 0))

    arr = np.asarray(canvas).astype(np.float32)
    img_t = torch.from_numpy(arr.copy()).permute(2, 0, 1).cuda()
    images = ImageList.from_tensors([(img_t - m.pixel_mean) / m.pixel_std], m.size_divisibility)

    gtext = pred.lang_encoder.get_text_token_embeddings([text], name='grounding', token=False, norm=False)
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

    # Upsample to canvas size, crop out the padding, then resize the real-content
    # region up to the true original (H, W) -- the inverse of the letterbox above.
    canvas_pred = F.interpolate(
        all_gm[best_q:best_q + 1][None], (infer_size, infer_size), mode='bilinear', align_corners=False,
    )[0, 0]
    content_pred = canvas_pred[:new_h, :new_w]
    final = F.interpolate(
        content_pred[None, None], (H, W), mode='bilinear', align_corners=False,
    )[0, 0].sigmoid().detach().cpu().numpy().astype(np.float32)
    return final


def prob_to_vein_mask(prob: np.ndarray, image_gray: np.ndarray = None,
                       threshold: float = VEIN_PROB_THRESHOLD) -> np.ndarray:
    """
    Keep only blobs that look like real vein cross-sections: small,
    anechoic, not elongated, not irregular. Area bounds are fractions of a
    FIXED reference pixel count (VEIN_AREA_REFERENCE_PX), not of
    prob.shape itself — see module docstring for why.
    """
    binary = (prob > threshold).astype(np.uint8)
    total = VEIN_AREA_REFERENCE_PX
    min_area = max(10, int(VEIN_MIN_AREA_FRAC * total))
    max_area = int(VEIN_MAX_AREA_FRAC * total)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        if image_gray is not None:
            mean_val = float(image_gray[labels == i].mean())
            if mean_val > VEIN_MAX_ANECHOIC_MEAN:
                continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw >= 1 and bh >= 1 and max(bw, bh) / min(bw, bh) > VEIN_MAX_ASPECT_RATIO:
            continue
        mask_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            perim = cv2.arcLength(cnts[0], True)
            if perim > 0 and (4 * np.pi * area / perim ** 2) < VEIN_MIN_CIRCULARITY:
                continue
        out[labels == i] = 1
    return out


def segment_frame(model, image_rgb: np.ndarray) -> np.ndarray:
    """
    Segment veins in a single RGB frame (numpy uint8 [H,W,3]).
    Returns a uint8 {0,1} mask, same H,W as input.
    """
    pil_img = Image.fromarray(image_rgb)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    with torch.no_grad():
        prob = grounding_prob(model, pil_img, text=VEIN_PROMPT)
    mask = prob_to_vein_mask(prob, image_gray=gray)
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
