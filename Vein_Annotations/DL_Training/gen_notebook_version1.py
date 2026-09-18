"""Generates Vein_Segmentation_Training_version1.ipynb. Run once: python gen_notebook_version1.py"""
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Vein Segmentation Training - version1 (pretrained ResNet34 backbone)

Goal: push test IoU up towards 0.85, using a faster-converging architecture than
the from-scratch Attention U-Net in the first notebook.

**What's different from the first notebook:**
- **Encoder is a pretrained ResNet34** (ImageNet weights) instead of a from-scratch CNN.
  Transfer learning converges in far fewer epochs, since the encoder already knows
  useful low/mid-level visual features - this is the main lever for reaching higher
  IoU within the same time budget.
- **Lighter attention**: dropped the 4x full attention-gates-on-every-skip (expensive,
  didn't clearly pay for itself last time). Kept a real multi-head self-attention block
  at the bottleneck (tiny at this resolution: 8x8 = 64 tokens with a 256px input) and a
  CBAM (channel+spatial attention) block right before the classification head.
- **Discriminative learning rates**: encoder fine-tunes slowly (it's already good),
  decoder trains fast (it's random-init and needs to learn from scratch).
- **Fixed the LR scheduler bug** from the first notebook (CosineAnnealingLR's `T_max`
  now matches the actual epoch cap, so it can't cycle back up and blow up training late
  in a run) + gradient clipping as a safety net.
- **Batch size 128** (up from 64) - ResNet34 at 256x256 is light for a 5090's 32GB.

Edit the model cell to swap in a different `torchvision` backbone (e.g. `resnet18` for
even more speed, `resnet50` for more capacity) if you want to explore further.
"""))

cells.append(code("""import json, time, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from IPython.display import clear_output, display

%matplotlib inline
"""))

cells.append(md("## 1. Config"))

cells.append(code("""SEED = 42
DATA_DIR = Path("../segmentation_dataset")

BATCH_SIZE = 128
ENCODER_LR = 1e-4      # pretrained backbone: fine-tune slowly
DECODER_LR = 1e-3      # random-init decoder/head: train fast
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

TRAIN_TIME_BUDGET_MIN = 50.0
MAX_EPOCHS = 150

OUT_DIR = Path("outputs_version1")
OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
"""))

cells.append(md("## 2. Load dataset tensors + train/val/test split"))

cells.append(code("""images = torch.load(DATA_DIR / "images.pt")   # uint8 [N,3,H,W]
masks  = torch.load(DATA_DIR / "masks.pt")    # uint8 [N,1,H,W]
with open(DATA_DIR / "meta.json") as f:
    meta = json.load(f)

print("images:", images.shape, images.dtype)
print("masks: ", masks.shape, masks.dtype)
print("total frames:", len(meta))
print("frames with a vein:", sum(m["has_vein"] for m in meta))
"""))

cells.append(code("""n = len(meta)
idx = np.arange(n)
has_vein = np.array([m["has_vein"] for m in meta], dtype=np.int64)

train_idx, temp_idx, y_train, y_temp = train_test_split(
    idx, has_vein, test_size=0.2, random_state=SEED, stratify=has_vein)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=SEED, stratify=y_temp)

print(f"train/val/test sizes: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
"""))

cells.append(code("""IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class VeinSegDataset(Dataset):
    def __init__(self, images, masks, indices, augment=False):
        self.images, self.masks, self.indices, self.augment = images, masks, indices, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.images[idx].float() / 255.0
        mask = self.masks[idx].float() / 255.0

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(1).item() < 0.3:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                img = torch.clamp(img * factor, 0, 1)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        mask = (mask > 0.5).float()
        return img, mask

train_ds = VeinSegDataset(images, masks, train_idx, augment=True)
val_ds   = VeinSegDataset(images, masks, val_idx, augment=False)
test_ds  = VeinSegDataset(images, masks, test_idx, augment=False)
"""))

cells.append(md("## 3. Visualize a batch of examples"))

cells.append(code("""def denorm_img(img_tensor):
    img = img_tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def show_examples(dataset, n=12, title=""):
    idxs = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idxs):
        img, mask = dataset[i]
        im = denorm_img(img)
        m = mask.squeeze(0).numpy()
        ax.imshow(im)
        ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.45)
        ax.set_title(f"idx {i}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

show_examples(train_ds, n=12, title="Sample training examples (image + GT mask)")
"""))

cells.append(md("""## 4. Model - pretrained ResNet34 U-Net + attention

- `use_bottleneck_mhsa` - real Transformer-style self-attention at the bottleneck (tiny: 8x8=64 tokens)
- `use_cbam_head` - channel+spatial attention right before the final 1x1 classification conv
- swap `tvm.resnet34` / `tvm.ResNet34_Weights` for `resnet18` (faster, less capacity) or `resnet50` (slower, more capacity) to try other backbones
"""))

cells.append(code("""class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MultiHeadSelfAttention2D(nn.Module):
    \"\"\"Standard Transformer self-attention (Vaswani et al.): softmax(QK^T/sqrt(d))V
    over flattened spatial positions, residual + pre-norm.\"\"\"
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        self.norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens_norm = self.norm(tokens)
        attn_out, _ = self.mha(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out
        return tokens.transpose(1, 2).reshape(B, C, H, W)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = torch.mean(x, dim=(2, 3))
        mx, _ = torch.max(x.view(b, c, -1), dim=2)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))
"""))

cells.append(code("""class ResNet34UNet(nn.Module):
    def __init__(self, out_ch=1, pretrained=True, use_bottleneck_mhsa=True, use_cbam_head=True, mhsa_heads=8):
        super().__init__()
        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = tvm.resnet34(weights=weights)

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # stride 2,  64ch
        self.pool = r.maxpool                                # stride 4
        self.layer1 = r.layer1                               # stride 4,  64ch
        self.layer2 = r.layer2                               # stride 8,  128ch
        self.layer3 = r.layer3                               # stride 16, 256ch
        self.layer4 = r.layer4                               # stride 32, 512ch  (bottleneck)

        self.bottleneck_mhsa = MultiHeadSelfAttention2D(512, num_heads=mhsa_heads) if use_bottleneck_mhsa else nn.Identity()

        self.dec4 = DecoderBlock(512, 256, 256)  # -> stride 16
        self.dec3 = DecoderBlock(256, 128, 128)  # -> stride 8
        self.dec2 = DecoderBlock(128, 64, 64)    # -> stride 4
        self.dec1 = DecoderBlock(64, 64, 32)     # -> stride 2   (skip = stem output)
        self.dec0 = nn.Sequential(               # -> stride 1
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.head_cbam = CBAM(16) if use_cbam_head else nn.Identity()
        self.head = nn.Conv2d(16, out_ch, kernel_size=1)

        # keep track of which params are "encoder" (pretrained) vs "decoder" (random init)
        self.encoder_modules = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        self.decoder_modules = [self.bottleneck_mhsa, self.dec4, self.dec3, self.dec2, self.dec1,
                                 self.dec0, self.head_cbam, self.head]

    def encoder_parameters(self):
        for m in self.encoder_modules:
            yield from m.parameters()

    def decoder_parameters(self):
        for m in self.decoder_modules:
            yield from m.parameters()

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s4 = self.bottleneck_mhsa(s4)

        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d)
        d = self.head_cbam(d)
        return self.head(d)

# --- instantiate + sanity check ---
model = ResNet34UNet(out_ch=1, pretrained=True, use_bottleneck_mhsa=True, use_cbam_head=True).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params/1e6:.2f}M")

with torch.no_grad():
    test_out = model(torch.randn(2, 3, 256, 256, device=device))
print("output shape:", test_out.shape)
"""))

cells.append(md("## 5. Loss and metrics"))

cells.append(code("""def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    inter = (probs * target).sum(1)
    union = probs.sum(1) + target.sum(1)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()

bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)

@torch.no_grad()
def compute_metrics(logits, target, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float().flatten(1)
    target_f = target.flatten(1)
    inter = (preds * target_f).sum(1)
    union = preds.sum(1) + target_f.sum(1) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (preds.sum(1) + target_f.sum(1) + eps)
    acc = (preds == target_f).float().mean(1)
    return iou.mean().item(), dice.mean().item(), acc.mean().item()
"""))

cells.append(md("""## 6. Train
Discriminative LRs (encoder slow / decoder fast), `CosineAnnealingLR` with `T_max=MAX_EPOCHS`
(fixed so it can't cycle back up late in the run like in the first notebook), and gradient
clipping. The chart below updates live after every epoch.
"""))

cells.append(code("""train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                           pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

optimizer = torch.optim.AdamW([
    {"params": model.encoder_parameters(), "lr": ENCODER_LR},
    {"params": model.decoder_parameters(), "lr": DECODER_LR},
], weight_decay=WEIGHT_DECAY)

# T_max = MAX_EPOCHS (not a guess like before) so the LR only ever decreases
# across the whole possible run, however many epochs the time budget allows.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
best_val_iou = -1.0
best_state = None

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
    ax2.axhline(0.85, color="red", linestyle="--", linewidth=1, label="0.85 target")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.legend(); ax2.set_title("Validation metrics")
    fig.tight_layout()
    plt.show()
    if history["epoch"]:
        e = history["epoch"][-1]
        print(f"epoch {e:3d} | {(time.time()-t_start)/60:5.1f} min | "
              f"train_loss {history['train_loss'][-1]:.4f} | val_loss {history['val_loss'][-1]:.4f} | "
              f"val_iou {history['val_iou'][-1]:.4f} | val_dice {history['val_dice'][-1]:.4f} | "
              f"best_val_iou {best_val_iou:.4f}")

while True:
    epoch += 1
    elapsed = time.time() - t_start
    if elapsed > time_budget_s or epoch > MAX_EPOCHS:
        print(f"stopping: elapsed={elapsed/60:.1f} min, epoch={epoch}")
        break

    model.train()
    running_loss, n_batches = 0.0, 0
    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = combined_loss(logits, msks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item(); n_batches += 1
        if time.time() - t_start > time_budget_s:
            break
    train_loss = running_loss / max(n_batches, 1)

    model.eval()
    val_loss_sum = val_iou_sum = val_dice_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = combined_loss(logits, msks)
            iou, dice, _ = compute_metrics(logits, msks)
            val_loss_sum += loss.item(); val_iou_sum += iou; val_dice_sum += dice
            val_batches += 1
    val_loss = val_loss_sum / max(val_batches, 1)
    val_iou = val_iou_sum / max(val_batches, 1)
    val_dice = val_dice_sum / max(val_batches, 1)
    scheduler.step()

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)
    history["val_dice"].append(val_dice)

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, OUT_DIR / "best_model.pt")

    live_plot()

print(f"training finished. best val IoU: {best_val_iou:.4f}")
if best_state is not None:
    model.load_state_dict(best_state)
"""))

cells.append(md("## 7. Evaluate on the held-out test set"))

cells.append(code("""test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model.eval()
test_iou_sum = test_dice_sum = test_acc_sum = 0.0
test_batches = 0
with torch.no_grad():
    for imgs, msks in test_loader:
        imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
        iou, dice, acc = compute_metrics(logits, msks)
        test_iou_sum += iou; test_dice_sum += dice; test_acc_sum += acc
        test_batches += 1

test_iou = test_iou_sum / max(test_batches, 1)
test_dice = test_dice_sum / max(test_batches, 1)
test_acc = test_acc_sum / max(test_batches, 1)
print(f"TEST  IoU: {test_iou:.4f}   Dice: {test_dice:.4f}   PixelAcc: {test_acc:.4f}")
if test_iou >= 0.85:
    print("target reached (>= 0.85 IoU)")
else:
    print(f"below 0.85 target by {0.85 - test_iou:.4f} - consider more epochs, a bigger backbone (resnet50), or higher input resolution")

summary = {
    "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
    "epochs_trained": epoch - 1, "best_val_iou": best_val_iou,
    "test_iou": test_iou, "test_dice": test_dice, "test_pixel_acc": test_acc,
    "model_params_millions": n_params / 1e6,
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
summary
"""))

cells.append(md("## 8. Visualize test predictions"))

cells.append(code("""def show_predictions(model, dataset, n=12):
    model.eval()
    idxs = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    with torch.no_grad():
        for r, i in enumerate(idxs):
            img, mask = dataset[i]
            logits = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()
            im = denorm_img(img)
            gt = mask.squeeze(0).numpy()

            axes[r, 0].imshow(im); axes[r, 0].axis("off")
            axes[r, 1].imshow(im); axes[r, 1].imshow(np.ma.masked_where(gt == 0, gt), cmap="autumn", alpha=0.5); axes[r, 1].axis("off")
            axes[r, 2].imshow(im); axes[r, 2].imshow(np.ma.masked_where(pred == 0, pred), cmap="winter", alpha=0.5); axes[r, 2].axis("off")
            if r == 0:
                axes[r, 0].set_title("image"); axes[r, 1].set_title("ground truth"); axes[r, 2].set_title("prediction")
    fig.tight_layout()
    plt.show()

show_predictions(model, test_ds, n=10)
"""))

cells.append(md("## 9. Save the trained model"))

cells.append(code("""torch.save(model.state_dict(), OUT_DIR / "final_model.pt")
print("saved to", (OUT_DIR / "final_model.pt").resolve())
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("Vein_Segmentation_Training_version1.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("wrote Vein_Segmentation_Training_version1.ipynb with", len(cells), "cells")
