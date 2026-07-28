"""
eval_baseline.py
Evaluates the ORIGINAL pretrained BiomedParse model (biomedparse_v1.pt,
no fine-tuning) on the fascia/vein test set.

Outputs:
  - eval_baseline_results.json   — mean IoU & Dice per class
  - eval_baseline_log.txt        — stdout/stderr (when run via shell redirect)
"""

import sys
import os
import json
import traceback

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = r'c:\Users\Krish\Downloads\Cygnus_Med_Demo\VLM_Fascia_Detection'
sys.path.insert(0, os.path.join(BASE_DIR, 'stubs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'BiomedParse'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from detectron2.structures import ImageList
from modeling.BaseModel import BaseModel
from modeling import build_model
from modeling.language.loss import vl_similarity
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_FILE  = os.path.join(BASE_DIR, 'BiomedParse', 'configs', 'biomed_fascia_finetuning.yaml')
WEIGHTS_FILE = os.path.join(BASE_DIR, 'pretrained', 'biomedparse_v1.pt')
TEST_JSON    = os.path.join(BASE_DIR, 'BiomedParse', 'biomedparse_datasets',
                            'Fascia_Detection', 'test_with_veins_backup.json')
IMAGES_DIR   = os.path.join(BASE_DIR, 'BiomedParse', 'biomedparse_datasets',
                            'Fascia_Detection', 'test')
MASKS_DIR    = os.path.join(BASE_DIR, 'BiomedParse', 'biomedparse_datasets',
                            'Fascia_Detection', 'test_mask')
OUTPUT_JSON  = os.path.join(BASE_DIR, 'eval_baseline_results.json')

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print("Loading pretrained BiomedParse model (no fine-tuning)...")
print(f"  weights : {WEIGHTS_FILE}")
print(f"  config  : {CONFIG_FILE}")

opt = load_opt_from_config_files([CONFIG_FILE])
opt = init_distributed(opt)
model = (
    BaseModel(opt, build_model(opt))
    .from_pretrained(WEIGHTS_FILE)
    .eval()
    .cuda()
)
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"], is_eval=True
    )
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Inference function — copied exactly from app.py _grounding_prob
# (no top_bias, no clahe — baseline settings)
# ---------------------------------------------------------------------------

def _grounding_prob(mdl, query_image_pil, text, infer_size=512):
    """Grounding inference. Returns float32 H×W probability map (0-1)."""
    m    = mdl.model
    pred = m.sem_seg_head.predictor
    W, H = query_image_pil.size

    arr   = np.asarray(query_image_pil.resize((infer_size, infer_size), Image.BICUBIC)).astype(np.float32)
    img_t = torch.from_numpy(arr.copy()).permute(2, 0, 1).cuda()
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
    weighted = probs.reshape(101, -1).max(dim=1).values
    best_q = weighted.argmax().item()
    return F.interpolate(
        all_gm[best_q:best_q + 1][None], (H, W), mode='bilinear', align_corners=False,
    )[0, 0].sigmoid().detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_dice(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union        = (pred_mask | gt_mask).sum()
    iou  = intersection / (union + 1e-7)
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-7)
    return float(iou), float(dice)


# ---------------------------------------------------------------------------
# Load test annotations
# ---------------------------------------------------------------------------
print(f"Loading test JSON: {TEST_JSON}")
with open(TEST_JSON) as f:
    data = json.load(f)

annotations = data['annotations']
print(f"Total annotations: {len(annotations)}")

# Category mapping
CAT_FASCIA = 17
CAT_VEIN   = 18

# Per-class accumulators
results = {
    'fascia': {'ious': [], 'dices': [], 'skipped': 0},
    'vein':   {'ious': [], 'dices': [], 'skipped': 0},
}

CAT_TO_KEY = {CAT_FASCIA: 'fascia', CAT_VEIN: 'vein'}

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
print("\nStarting evaluation...\n")

for i, ann in enumerate(annotations):
    cat_id   = ann['category_id']
    cls_key  = CAT_TO_KEY.get(cat_id)
    if cls_key is None:
        print(f"[{i}] Unknown category_id={cat_id}, skipping.")
        continue

    img_file  = ann['file_name']
    mask_file = ann['mask_file']
    text      = ann['sentences'][0]['raw']

    img_path  = os.path.join(IMAGES_DIR, img_file)
    mask_path = os.path.join(MASKS_DIR,  mask_file)

    if i % 50 == 0:
        print(f"[{i}/{len(annotations)}] cls={cls_key}  img={img_file}")

    try:
        # Load image
        image_pil = Image.open(img_path).convert('RGB')

        # Load ground-truth mask (pixel value 1 = structure present)
        gt_mask = np.array(Image.open(mask_path).convert('L')) > 0

        # Run inference
        with torch.no_grad():
            prob = _grounding_prob(model, image_pil, text)

        # Threshold at 0.5
        pred_mask = prob > 0.5

        # Compute metrics
        iou, dice = iou_dice(pred_mask, gt_mask)

        results[cls_key]['ious'].append(iou)
        results[cls_key]['dices'].append(dice)

    except Exception as e:
        print(f"  [SKIP] {img_file}: {e}")
        traceback.print_exc()
        results[cls_key]['skipped'] += 1

print("\nEvaluation complete.\n")

# ---------------------------------------------------------------------------
# Aggregate and save
# ---------------------------------------------------------------------------
summary = {}
for cls_key, acc in results.items():
    ious  = acc['ious']
    dices = acc['dices']
    n     = len(ious)
    summary[cls_key] = {
        'mean_iou':  float(np.mean(ious))  if n > 0 else 0.0,
        'mean_dice': float(np.mean(dices)) if n > 0 else 0.0,
        'n':         n,
        'skipped':   acc['skipped'],
    }

with open(OUTPUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)

print("=" * 60)
print("BASELINE RESULTS (pretrained biomedparse_v1.pt, no fine-tuning)")
print("=" * 60)
for cls_key, s in summary.items():
    print(f"  {cls_key.upper():8s}  mean_IoU={s['mean_iou']:.4f}  mean_Dice={s['mean_dice']:.4f}"
          f"  (n={s['n']}, skipped={s['skipped']})")
print("=" * 60)
print(f"\nResults saved to: {OUTPUT_JSON}")
