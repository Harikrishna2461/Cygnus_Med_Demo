"""Generates Vein_Segmentation_Training_v2.ipynb. Run once: python gen_notebook_v2.py"""
import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""# Vein Segmentation Training - v2 (target: 0.90 IoU)

v1 reached 0.81 test IoU. What v2 changes, and why:

- **Higher resolution (320x320, was 256x256)** from `segmentation_dataset_v2` - thin veins and edge pixels survive the downscale.
  Masks are resized with area-averaging + threshold (smoother boundaries than nearest-neighbour).
- **LR schedule actually finishes.** v1 early-stopped at epoch 13 of a 30-epoch one-cycle schedule, i.e. while LR was still high and
  before the low-LR fine-tuning phase where most IoU is gained. Early stopping is now disabled until 70% of the schedule has run.
- **Boundary-weighted BCE + Dice + Focal-Tversky** loss - pixels near the vein edge count more, and missed vein pixels are penalised
  more than extra ones (fights the "contour sits inside the vein" shrinkage).
- **Post-processing tuned on the validation set**: probability threshold + a *minimum predicted area*. About 22% of frames contain no vein,
  and a single stray predicted pixel scores IoU = 0 on those frames, so cleaning up small false positives is worth a lot.
- Flip test-time augmentation, EMA weights, gradient clipping, NaN-gradient skipping, bf16, everything on the GPU.
- `SPLIT_MODE = "block"` gives an honest score (whole 5-second blocks held out) - consecutive frames are near-duplicates, so the
  random frame split (`"random"`, comparable to v1) is optimistic.
"""))

cells.append(code(r'''
import json, time, math, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from IPython.display import clear_output

%matplotlib inline
'''))

cells.append(md("## 1. Config"))

cells.append(code(r'''
SEED = 42
DATA_DIR = Path("../segmentation_dataset_v2")

BACKBONE = "resnet34"          # "resnet50" for more capacity (needs more GPU memory + time)
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 8                   # early stopping only counts after 70% of MAX_EPOCHS
TRAIN_TIME_BUDGET_MIN = 60.0
LR_ENC, LR_DEC = 3e-4, 2e-3    # pretrained encoder slow, random-init decoder fast
WEIGHT_DECAY = 1e-4
LONG_OVERSAMPLE = 10           # repeat longitudinal train frames

SPLIT_MODE = "random"          # "random" = comparable to v1 | "block" = honest (whole 150-frame blocks held out)
BLOCK = 150

OUT_DIR = Path("outputs_v2")
OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
'''))

cells.append(md("## 2. Load dataset onto the GPU + train/val/test split"))

cells.append(code(r'''
def to_gpu(path):
    arr = np.load(path, mmap_mode="r")
    out = torch.empty(arr.shape, dtype=torch.uint8, device=device)
    for i in range(0, len(arr), 512):
        out[i:i + 512] = torch.from_numpy(np.ascontiguousarray(arr[i:i + 512])).to(device)
    return out

images_gpu = to_gpu(DATA_DIR / "images.npy")      # uint8 [N,3,R,R]
masks_gpu = to_gpu(DATA_DIR / "masks.npy")        # uint8 [N,R,R]  (0/1)
with open(DATA_DIR / "meta.json") as f:
    meta = json.load(f)
RES = images_gpu.shape[-1]

print("images:", tuple(images_gpu.shape), "| masks:", tuple(masks_gpu.shape))
print("total frames:", len(meta), "| with vein:", sum(m["has_vein"] for m in meta),
      "| longitudinal:", sum(m["view"] == "longitudinal" for m in meta))
print(f"GPU memory used by this notebook: {torch.cuda.memory_allocated()/1e9:.1f} GB")
'''))

cells.append(code(r'''
n = len(meta)
idx = np.arange(n)
has_vein = np.array([m["has_vein"] for m in meta], dtype=np.int64)
is_long = np.array([m["view"] == "longitudinal" for m in meta])

if SPLIT_MODE == "random":
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, has_vein, test_size=0.2, random_state=SEED, stratify=has_vein)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED, stratify=y_temp)
else:
    keys = np.array([f'{m["source"]}:{m["frame_index"] // BLOCK}' for m in meta])
    blocks = np.unique(keys)
    rng = np.random.RandomState(SEED); rng.shuffle(blocks)
    nb = len(blocks)
    tr_b, va_b, te_b = set(blocks[:int(.8 * nb)]), set(blocks[int(.8 * nb):int(.9 * nb)]), set(blocks[int(.9 * nb):])
    train_idx = idx[[k in tr_b for k in keys]]
    val_idx = idx[[k in va_b for k in keys]]
    test_idx = idx[[k in te_b for k in keys]]

long_train = train_idx[is_long[train_idx]]
if len(long_train):
    train_idx = np.concatenate([train_idx] + [long_train] * (LONG_OVERSAMPLE - 1))
print(f"split mode: {SPLIT_MODE} | train/val/test sizes: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
print(f"longitudinal: train(unique)={len(long_train)}, val={int(is_long[val_idx].sum())}, test={int(is_long[test_idx].sum())}")
print(f"empty frames: val={int((~has_vein[val_idx].astype(bool)).sum())}, test={int((~has_vein[test_idx].astype(bool)).sum())}")
'''))

cells.append(code(r'''
MEAN_G = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD_G = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def get_batch(ids, augment):
    ids_t = torch.as_tensor(ids, device=device)
    x = images_gpu[ids_t].float() / 255.0
    y = masks_gpu[ids_t].float().unsqueeze(1)
    if augment:
        B = x.shape[0]
        r = lambda: torch.rand(B, 1, 1, 1, device=device)
        flip = r() < 0.5
        x = torch.where(flip, x.flip(3), x); y = torch.where(flip, y.flip(3), y)
        c, b = 0.8 + 0.4 * r(), 0.75 + 0.5 * r()                       # contrast, brightness
        m = x.mean((1, 2, 3), keepdim=True)
        x = (((x - m) * c + m) * b).clamp(0, 1) ** (0.8 + 0.4 * r())   # + gamma
        x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)               # speckle-like noise
        ang = (torch.rand(B, device=device) - 0.5) * 2 * math.radians(15)
        sc = 1 + (torch.rand(B, device=device) - 0.5) * 0.4
        tx, ty = [(torch.rand(B, device=device) - 0.5) * 0.25 for _ in range(2)]
        cs, sn = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cs, -sn, tx], 1), torch.stack([sn, cs, ty], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        y = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = ((x - MEAN_G) / STD_G).contiguous(memory_format=torch.channels_last)
    return x, (y > 0.5).float()

def denorm_img(x):
    return (x.cpu() * STD_G.cpu()[0] + MEAN_G.cpu()[0]).clamp(0, 1).permute(1, 2, 0).numpy()
'''))

cells.append(md("## 3. Visualize a batch of (augmented) training examples"))

cells.append(code(r'''
xb, yb = get_batch(np.random.choice(train_idx, 12, replace=False), augment=True)
fig, axes = plt.subplots(3, 4, figsize=(13, 10))
for ax, x, y in zip(axes.reshape(-1), xb, yb):
    ax.imshow(denorm_img(x))
    m = y[0].cpu().numpy()
    ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.45)
    ax.axis("off")
fig.suptitle("Augmented training examples (image + GT mask)")
fig.tight_layout(); plt.show()
'''))

cells.append(md("""## 4. Model - pretrained U-Net (ResNet encoder)
Plain U-Net decoder on an ImageNet-pretrained ResNet, plus a full-resolution skip from the raw input so boundaries stay sharp.
Weights are tracked with an EMA - the EMA copy is what gets validated, saved and tested."""))

cells.append(code(r'''
CH = {"resnet34": (64, 64, 128, 256, 512), "resnet50": (64, 256, 512, 1024, 2048)}

class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))

class ResUNet(nn.Module):
    def __init__(self, backbone="resnet34"):
        super().__init__()
        c0, c1, c2, c3, c4 = CH[backbone]
        r = getattr(tvm, backbone)(weights="DEFAULT")
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool, self.l1, self.l2, self.l3, self.l4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4
        self.u4, self.u3, self.u2, self.u1 = Up(c4, c3, 256), Up(256, c2, 128), Up(128, c1, 64), Up(64, c0, 32)
        self.u0 = nn.Sequential(nn.Conv2d(32 + 3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
        self.head = nn.Conv2d(16, 1, 1)
        self.enc_modules = [self.stem, self.l1, self.l2, self.l3, self.l4]
        self.dec_modules = [self.u4, self.u3, self.u2, self.u1, self.u0, self.head]
    def forward(self, x):
        s0 = self.stem(x); s1 = self.l1(self.pool(s0)); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
        d = self.u4(s4, s3); d = self.u3(d, s2); d = self.u2(d, s1); d = self.u1(d, s0)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.u0(torch.cat([d, x], 1)))

model = ResUNet(BACKBONE).to(device).to(memory_format=torch.channels_last)
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)

def ema_update(decay):
    with torch.no_grad():
        for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
            if pe.dtype.is_floating_point:
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            else:
                pe.copy_(pm)

n_params = sum(p.numel() for p in model.parameters())
print(f"{BACKBONE} U-Net params: {n_params/1e6:.2f}M | input res {RES}")
with torch.no_grad():
    print("output shape:", model(torch.randn(2, 3, RES, RES, device=device)).shape)
'''))

cells.append(md("## 5. Loss and metrics"))

cells.append(code(r'''
def weighted_bce(logits, target):
    # pixels near the vein boundary get up to 6x weight
    w = 1 + 5 * torch.abs(F.avg_pool2d(target, 31, stride=1, padding=15) - target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (w * bce).sum() / w.sum()

def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
    inter = (p * t).sum(1)
    return 1 - ((2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()

def focal_tversky(logits, target, a=0.6, b=0.4, g=0.75, eps=1.0):
    p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
    tp = (p * t).sum(1); fn = ((1 - p) * t).sum(1); fp = (p * (1 - t)).sum(1)
    return ((1 - (tp + eps) / (tp + a * fn + b * fp + eps)) ** g).mean()   # a>b: missed vein pixels cost more than extra ones

def combined_loss(logits, target):
    return weighted_bce(logits, target) + dice_loss(logits, target) + focal_tversky(logits, target)

@torch.no_grad()
def compute_metrics(logits, target, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float().flatten(1)
    t = target.flatten(1)
    inter = (preds * t).sum(1)
    union = preds.sum(1) + t.sum(1) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (preds.sum(1) + t.sum(1) + eps)
    acc = (preds == t).float().mean(1)
    return iou.mean().item(), dice.mean().item(), acc.mean().item()
'''))

cells.append(md("""## 6. Train
One-cycle LR (warmup, then cosine decay to ~0), gradient clipping, non-finite gradient steps are skipped (so training can't collapse),
early stopping only after 70% of the schedule. The chart updates after every epoch."""))

cells.append(code(r'''
enc_params = [p for m in model.enc_modules for p in m.parameters()]
dec_params = [p for m in model.dec_modules for p in m.parameters()]
optimizer = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC},
                               {"params": dec_params, "lr": LR_DEC}], weight_decay=WEIGHT_DECAY)
steps_per_epoch = len(train_idx) // BATCH_SIZE
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[LR_ENC, LR_DEC], total_steps=MAX_EPOCHS * steps_per_epoch,
    pct_start=0.1, anneal_strategy="cos", div_factor=20, final_div_factor=200)

history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
best_val_iou, best_state, bad_epochs, step = -1.0, None, 0, 0
time_budget_s = TRAIN_TIME_BUDGET_MIN * 60
t_start = time.time()
epoch = 0

def live_plot():
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss")
    ax1.plot(history["epoch"], history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(history["epoch"], history["val_iou"], label="val IoU", color="green")
    ax2.plot(history["epoch"], history["val_dice"], label="val Dice", color="orange")
    ax2.axhline(0.90, color="red", linestyle="--", linewidth=1, label="0.90 target")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.legend(); ax2.set_title("Validation metrics (EMA weights, thr 0.5)")
    fig.tight_layout(); plt.show()
    e = history["epoch"][-1]
    print(f"epoch {e:3d} | {(time.time()-t_start)/60:5.1f} min | train_loss {history['train_loss'][-1]:.4f} | "
          f"val_loss {history['val_loss'][-1]:.4f} | val_iou {history['val_iou'][-1]:.4f} | best {best_val_iou:.4f}")

while epoch < MAX_EPOCHS and (time.time() - t_start) < time_budget_s:
    epoch += 1
    model.train()
    perm = np.random.permutation(train_idx)
    running, nb, skipped = 0.0, 0, 0
    for b in range(steps_per_epoch):
        x, y = get_batch(perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE], augment=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = combined_loss(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isfinite(gnorm):
            optimizer.step()
        else:
            skipped += 1                       # NaN/inf gradient -> skip the step instead of corrupting the weights
        scheduler.step(); step += 1
        ema_update(min(0.998, (1 + step) / (10 + step)))
        running += loss.item(); nb += 1
    train_loss = running / max(nb, 1)

    ema.eval()
    vl = vi = vd = 0.0; vb = 0
    with torch.no_grad():
        for i in range(0, len(val_idx), 64):
            x, y = get_batch(val_idx[i:i + 64], augment=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = ema(x).float()
            iou, dice, _ = compute_metrics(logits, y)
            vl += combined_loss(logits, y).item(); vi += iou; vd += dice; vb += 1
    val_loss, val_iou, val_dice = vl / vb, vi / vb, vd / vb

    for k, v in zip(history, [epoch, train_loss, val_loss, val_iou, val_dice]):
        history[k].append(v)

    if val_iou > best_val_iou:
        best_val_iou, bad_epochs = val_iou, 0
        best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
        torch.save(best_state, OUT_DIR / "best_model.pt")
    elif epoch >= 0.7 * MAX_EPOCHS:
        bad_epochs += 1
    live_plot()
    if skipped:
        print(f"  skipped {skipped} non-finite gradient steps this epoch")
    if bad_epochs >= PATIENCE:
        print("early stopping"); break

print(f"training finished after {epoch} epochs. best val IoU (thr 0.5): {best_val_iou:.4f}")
ema.load_state_dict(best_state)      # everything below evaluates the best EMA weights
'''))

cells.append(md("""## 7. Tune post-processing on the validation set, then evaluate on the test set
Searches (probability threshold x minimum predicted area) on **validation only**, using flip-TTA. Predictions smaller than
`min_area` pixels are treated as "no vein" - this removes stray false positives on empty frames."""))

cells.append(code(r'''
assert best_state is not None, "run the training cell (section 6) before evaluating"

@torch.no_grad()
def predict_probs(ids, tta=True, bs=64):
    ema.eval(); out = []
    for i in range(0, len(ids), bs):
        x, _ = get_batch(ids[i:i + bs], augment=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = ema(x).float()
            if tta:
                lg = 0.5 * (lg + ema(x.flip(3)).float().flip(3))
        out.append(torch.sigmoid(lg).half())
    return torch.cat(out)

def frame_scores(prob, gt, thr, min_area):
    pred = prob.float() > thr
    pred = pred & (pred.flatten(1).sum(1) >= min_area).view(-1, 1, 1, 1)
    g = gt > 0.5
    inter = (pred & g).flatten(1).sum(1).float()
    union = (pred | g).flatten(1).sum(1).float()
    tot = pred.flatten(1).sum(1).float() + g.flatten(1).sum(1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))     # empty vs empty = 1
    dice = torch.where(tot > 0, 2 * inter / tot.clamp(min=1), torch.ones_like(union))
    return iou, dice

def gt_of(ids):
    return masks_gpu[torch.as_tensor(ids, device=device)].float().unsqueeze(1)

val_probs, val_gt = predict_probs(val_idx), gt_of(val_idx)
thrs = np.arange(0.30, 0.71, 0.05)
areas = [0, 25, 50, 100, 200, 400, 800]
grid = {(round(float(t), 2), a): frame_scores(val_probs, val_gt, t, a)[0].mean().item() for t in thrs for a in areas}
(best_thr, best_area), best_val_tuned = max(grid.items(), key=lambda kv: kv[1])
print(f"val IoU  thr 0.5 / no filter: {grid[(0.5, 0)]:.4f}   ->   tuned (thr={best_thr}, min_area={best_area}): {best_val_tuned:.4f}")

test_probs, test_gt = predict_probs(test_idx), gt_of(test_idx)
iou, dice = frame_scores(test_probs, test_gt, best_thr, best_area)
has_v = test_gt.flatten(1).sum(1) > 0
test_iou, test_dice = iou.mean().item(), dice.mean().item()
test_iou_vein = iou[has_v].mean().item()
test_empty_ok = iou[~has_v].mean().item()
test_iou_plain = frame_scores(test_probs, test_gt, 0.5, 0)[0].mean().item()
print(f"TEST  IoU: {test_iou:.4f}   Dice: {test_dice:.4f}     (plain thr 0.5, no filter: {test_iou_plain:.4f})")
print(f"  frames WITH vein   : IoU {test_iou_vein:.4f}  ({int(has_v.sum())} frames)")
print(f"  frames WITHOUT vein: correctly empty {test_empty_ok:.4f}  ({int((~has_v).sum())} frames)")
lg_mask = torch.as_tensor(is_long[test_idx], device=device)
if lg_mask.any():
    print(f"  longitudinal frames: IoU {iou[lg_mask].mean().item():.4f}  ({int(lg_mask.sum())} frames)")
print("target reached (>= 0.90 IoU)" if test_iou >= 0.90 else f"below the 0.90 target by {0.90 - test_iou:.4f}")

summary = {
    "split_mode": SPLIT_MODE, "backbone": BACKBONE, "resolution": RES,
    "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
    "epochs_trained": epoch, "best_val_iou_thr05": best_val_iou, "tuned_thr": best_thr, "tuned_min_area": best_area,
    "val_iou_tuned": best_val_tuned, "test_iou": test_iou, "test_dice": test_dice,
    "test_iou_vein_frames": test_iou_vein, "test_empty_frames_correct": test_empty_ok,
    "test_iou_plain": test_iou_plain, "model_params_millions": n_params / 1e6,
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
summary
'''))

cells.append(md("## 8. Visualize test predictions (green = ground truth, red = prediction)"))

cells.append(code(r'''
def show_predictions(n=10):
    sel = np.random.choice(len(test_idx), size=min(n, len(test_idx)), replace=False)
    fig, axes = plt.subplots(len(sel), 3, figsize=(10, 3.3 * len(sel)))
    for r, j in enumerate(sel):
        x, y = get_batch(test_idx[j:j + 1], augment=False)
        p = test_probs[j:j + 1].float() > best_thr
        if p.sum() < best_area:
            p = torch.zeros_like(p)
        pred = p[0, 0].cpu().numpy()
        gt = y[0, 0].cpu().numpy(); im = denorm_img(x[0])
        i_ = iou[j].item()
        axes[r, 0].imshow(im); axes[r, 0].axis("off")
        axes[r, 1].imshow(im); axes[r, 1].contour(gt, levels=[0.5], colors="lime", linewidths=1.5); axes[r, 1].axis("off")
        axes[r, 2].imshow(im); axes[r, 2].contour(gt, levels=[0.5], colors="lime", linewidths=1)
        if pred.any(): axes[r, 2].contour(pred.astype(float), levels=[0.5], colors="red", linewidths=1.5)
        axes[r, 2].set_title(f"IoU {i_:.3f}", fontsize=9); axes[r, 2].axis("off")
        if r == 0:
            axes[r, 0].set_title("image"); axes[r, 1].set_title("ground truth")
    fig.tight_layout(); plt.show()

show_predictions(10)
'''))

cells.append(md("## 9. Save the trained model"))

cells.append(code(r'''
torch.save({"state_dict": ema.state_dict(), "backbone": BACKBONE, "threshold": best_thr, "min_area": best_area, "res": RES},
           OUT_DIR / "final_model.pt")
print("saved to", (OUT_DIR / "final_model.pt").resolve())
'''))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("Vein_Segmentation_Training_v2.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("wrote Vein_Segmentation_Training_v2.ipynb with", len(cells), "cells")
