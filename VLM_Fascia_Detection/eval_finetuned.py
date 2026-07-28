import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'stubs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'BiomedParse'))

from detectron2.structures import ImageList
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
import glob as _glob


def load_model(weights):
    opt = load_opt_from_config_files([os.path.join(BASE_DIR, 'BiomedParse', 'configs', 'biomed_fascia_finetuning.yaml')])
    opt = init_distributed(opt)
    m = BaseModel(opt, build_model(opt)).from_pretrained(weights).eval().cuda()
    with torch.no_grad():
        m.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ['background'], is_eval=True)
    return m


def _grounding_prob(mdl, img_pil, text, infer_size=512):
    m = mdl.model
    pred = m.sem_seg_head.predictor
    W, H = img_pil.size
    arr = np.asarray(img_pil.resize((infer_size, infer_size), Image.BICUBIC)).astype(np.float32)
    img_t = torch.from_numpy(arr.copy()).permute(2, 0, 1).cuda()
    images = ImageList.from_tensors([(img_t - m.pixel_mean) / m.pixel_std], m.size_divisibility)
    gtext = pred.lang_encoder.get_text_token_embeddings([text], name='grounding', token=False, norm=False)
    tok_emb = gtext['token_emb']
    tok_mask = gtext['tokens']['attention_mask'].bool()
    q_emb = tok_emb[tok_mask]
    nz_mask = torch.zeros(q_emb[:, None].shape[:-1], dtype=torch.bool, device=q_emb.device)
    extra = {'grounding_tokens': q_emb[:, None], 'grounding_nonzero_mask': nz_mask.t(), 'grounding_class': gtext['class_emb']}
    with torch.no_grad():
        feats = m.backbone(images.tensor)
        mf, _, ms = m.sem_seg_head.pixel_decoder.forward_features(feats)
        outputs = pred(ms, mf, extra=extra, task='grounding_eval')
    all_gm = outputs['pred_gmasks'][0]
    probs = torch.sigmoid(all_gm)
    best_q = probs.reshape(101, -1).max(dim=1).values.argmax().item()
    return F.interpolate(
        all_gm[best_q:best_q + 1][None], (H, W), mode='bilinear', align_corners=False
    )[0, 0].sigmoid().detach().cpu().numpy().astype(np.float32)


def iou_dice(pred, gt):
    i = (pred & gt).sum()
    u = (pred | gt).sum()
    return float(i / (u + 1e-7)), float(2 * i / (pred.sum() + gt.sum() + 1e-7))


DATASET = os.path.join(BASE_DIR, 'BiomedParse', 'biomedparse_datasets', 'Fascia_Detection')
with open(os.path.join(DATASET, 'test_with_veins_backup.json')) as f:
    anns = json.load(f)['annotations']

# ── Fascia fine-tuned ─────────────────────────────────────────────────────────
ckpts = sorted(_glob.glob(os.path.join(BASE_DIR, 'BiomedParse', 'output',
    'fascia_finetuning_v2_production', '**', 'model_state_dict.pt'), recursive=True), key=os.path.getmtime)
print('Loading fascia model:', ckpts[-1])
fmodel = load_model(ckpts[-1])

fascia_ious, fascia_dices = [], []
fascia_anns = [a for a in anns if a['category_id'] == 17]
for i, a in enumerate(fascia_anns):
    try:
        img = Image.open(os.path.join(DATASET, 'test', a['file_name'])).convert('RGB')
        gt  = np.array(Image.open(os.path.join(DATASET, 'test_mask', a['mask_file'])).convert('L')) > 0
        prob = _grounding_prob(fmodel, img, a['sentences'][0]['raw'])
        pred = prob > 0.5
        iou, dice = iou_dice(pred, gt)
        fascia_ious.append(iou); fascia_dices.append(dice)
    except Exception as e:
        print(f'  skip {a["file_name"]}: {e}')
    if i % 50 == 0:
        print(f'  fascia [{i}/{len(fascia_anns)}]')

print(f'\nFASCIA FT: IoU={np.mean(fascia_ious):.4f}  Dice={np.mean(fascia_dices):.4f}  n={len(fascia_ious)}')

del fmodel
torch.cuda.empty_cache()

# ── Vein fine-tuned ───────────────────────────────────────────────────────────
ckpts2 = sorted(_glob.glob(os.path.join(BASE_DIR, 'BiomedParse', 'output',
    'fascia_vein_finetuning', '**', 'model_state_dict.pt'), recursive=True), key=os.path.getmtime)
print('\nLoading vein model:', ckpts2[-1])
vmodel = load_model(ckpts2[-1])

vein_ious, vein_dices = [], []
vein_anns = [a for a in anns if a['category_id'] == 18]
for i, a in enumerate(vein_anns):
    try:
        img = Image.open(os.path.join(DATASET, 'test', a['file_name'])).convert('RGB')
        gt  = np.array(Image.open(os.path.join(DATASET, 'test_mask', a['mask_file'])).convert('L')) > 0
        prob = _grounding_prob(vmodel, img, a['sentences'][0]['raw'])
        pred = prob > 0.5
        iou, dice = iou_dice(pred, gt)
        vein_ious.append(iou); vein_dices.append(dice)
    except Exception as e:
        print(f'  skip {a["file_name"]}: {e}')
    if i % 50 == 0:
        print(f'  vein [{i}/{len(vein_anns)}]')

print(f'\nVEIN FT: IoU={np.mean(vein_ious):.4f}  Dice={np.mean(vein_dices):.4f}  n={len(vein_ious)}')

results = {
    'fascia_ft': {'mean_iou': float(np.mean(fascia_ious)), 'mean_dice': float(np.mean(fascia_dices)), 'n': len(fascia_ious)},
    'vein_ft':   {'mean_iou': float(np.mean(vein_ious)),   'mean_dice': float(np.mean(vein_dices)),   'n': len(vein_ious)},
}
with open('eval_finetuned_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*60)
print(f"FASCIA FT : IoU={results['fascia_ft']['mean_iou']:.4f}  Dice={results['fascia_ft']['mean_dice']:.4f}")
print(f"VEIN FT   : IoU={results['vein_ft']['mean_iou']:.4f}  Dice={results['vein_ft']['mean_dice']:.4f}")
print('='*60)
print('Saved to eval_finetuned_results.json')
