"""
Train an Attention U-Net for binary vein segmentation on the tensors built
by build_dataset.py, capped to a wall-clock time budget so the whole run
(train + eval + visualization) stays under an hour on an RTX 5090.

Usage:
    python train.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_raw, make_splits, VeinSegDataset, IMAGENET_MEAN, IMAGENET_STD
from model import AttentionUNet

# ---------------------------------------------------------------- config ---
SEED = 42
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_TIME_BUDGET_MIN = 50.0  # hard cap on training loop; total script < 1hr
MAX_EPOCHS = 200
NUM_EXAMPLES_TO_VISUALIZE = 12
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")


# ------------------------------------------------------------- loss/eval ---
def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    inter = (probs * target).sum(1)
    union = probs.sum(1) + target.sum(1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


bce_loss = nn.BCEWithLogitsLoss()


def combined_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def compute_metrics(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds_f = preds.flatten(1)
    target_f = target.flatten(1)
    inter = (preds_f * target_f).sum(1)
    union = preds_f.sum(1) + target_f.sum(1) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (preds_f.sum(1) + target_f.sum(1) + eps)
    acc = (preds_f == target_f).float().mean(1)
    return iou.mean().item(), dice.mean().item(), acc.mean().item()


def denorm_img(img_tensor):
    img = img_tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ------------------------------------------------------ visualize samples ---
def visualize_examples(dataset, n, out_path, title):
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
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print("saved", out_path)


def visualize_predictions(model, dataset, n, out_path):
    model.eval()
    idxs = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    cols = 3
    rows = len(idxs)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
    with torch.no_grad():
        for r, i in enumerate(idxs):
            img, mask = dataset[i]
            logits = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()
            im = denorm_img(img)
            gt = mask.squeeze(0).numpy()

            axes[r, 0].imshow(im)
            axes[r, 0].set_title("image" if r == 0 else "")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(im)
            axes[r, 1].imshow(np.ma.masked_where(gt == 0, gt), cmap="autumn", alpha=0.5)
            axes[r, 1].set_title("ground truth" if r == 0 else "")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(im)
            axes[r, 2].imshow(np.ma.masked_where(pred == 0, pred), cmap="winter", alpha=0.5)
            axes[r, 2].set_title("prediction" if r == 0 else "")
            axes[r, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print("saved", out_path)


def plot_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss")
    ax1.plot(history["epoch"], history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(history["epoch"], history["val_iou"], label="val IoU", color="green")
    ax2.plot(history["epoch"], history["val_dice"], label="val Dice", color="orange")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.legend()
    ax2.set_title("Validation metrics")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print("saved", out_path)


# ---------------------------------------------------------------- main -----
def main():
    print("loading dataset tensors...")
    images, masks, meta = load_raw()
    print("images:", images.shape, "masks:", masks.shape, "n:", len(meta))

    train_idx, val_idx, test_idx = make_splits(meta, seed=SEED)
    print(f"train/val/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    train_ds = VeinSegDataset(images, masks, train_idx, augment=True)
    val_ds = VeinSegDataset(images, masks, val_idx, augment=False)
    test_ds = VeinSegDataset(images, masks, test_idx, augment=False)

    visualize_examples(train_ds, NUM_EXAMPLES_TO_VISUALIZE, OUT_DIR / "01_example_batch.png",
                        "Sample training examples (image + GT mask)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = AttentionUNet(in_ch=3, out_ch=1, base_ch=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_val_iou = -1.0
    best_state = None

    time_budget_s = TRAIN_TIME_BUDGET_MIN * 60
    t_start = time.time()
    epoch = 0

    print(f"training (time budget {TRAIN_TIME_BUDGET_MIN} min, max {MAX_EPOCHS} epochs)...")
    while True:
        epoch += 1
        elapsed = time.time() - t_start
        if elapsed > time_budget_s or epoch > MAX_EPOCHS:
            print(f"stopping: elapsed={elapsed/60:.1f}min epoch={epoch}")
            break

        model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, msks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = combined_loss(logits, msks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            if time.time() - t_start > time_budget_s:
                break

        train_loss = running_loss / max(n_batches, 1)

        model.eval()
        val_loss_sum, val_iou_sum, val_dice_sum, val_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                msks = msks.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = combined_loss(logits, msks)
                iou, dice, _ = compute_metrics(logits, msks)
                val_loss_sum += loss.item()
                val_iou_sum += iou
                val_dice_sum += dice
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

        elapsed_min = (time.time() - t_start) / 60
        print(f"epoch {epoch:3d} | {elapsed_min:5.1f}min | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_iou {val_iou:.4f} | val_dice {val_dice:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT_DIR / "best_model.pt")

    print(f"training finished. best val IoU: {best_val_iou:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    plot_curves(history, OUT_DIR / "02_training_curves.png")

    # -------------------------------------------------------------- eval ---
    print("evaluating on test set...")
    model.eval()
    test_iou_sum, test_dice_sum, test_acc_sum, test_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, msks in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
            iou, dice, acc = compute_metrics(logits, msks)
            test_iou_sum += iou
            test_dice_sum += dice
            test_acc_sum += acc
            test_batches += 1

    test_iou = test_iou_sum / max(test_batches, 1)
    test_dice = test_dice_sum / max(test_batches, 1)
    test_acc = test_acc_sum / max(test_batches, 1)
    print(f"TEST  IoU: {test_iou:.4f}  Dice: {test_dice:.4f}  PixelAcc: {test_acc:.4f}")

    visualize_predictions(model, test_ds, NUM_EXAMPLES_TO_VISUALIZE, OUT_DIR / "03_test_predictions.png")

    summary = {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "epochs_trained": epoch - 1,
        "training_minutes": (time.time() - t_start) / 60,
        "best_val_iou": best_val_iou,
        "test_iou": test_iou,
        "test_dice": test_dice,
        "test_pixel_acc": test_acc,
        "model_params_millions": n_params / 1e6,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("summary:", summary)
    print("done. outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
