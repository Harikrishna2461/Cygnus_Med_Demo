import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "segmentation_dataset"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_raw():
    images = torch.load(DATA_DIR / "images.pt")  # uint8 N,3,H,W
    masks = torch.load(DATA_DIR / "masks.pt")  # uint8 N,1,H,W
    with open(DATA_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return images, masks, meta


def make_splits(meta, seed=42):
    n = len(meta)
    idx = np.arange(n)
    has_vein = np.array([m["has_vein"] for m in meta], dtype=np.int64)

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, has_vein, test_size=0.2, random_state=seed, stratify=has_vein
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return train_idx, val_idx, test_idx


class VeinSegDataset(Dataset):
    def __init__(self, images, masks, indices, augment=False):
        self.images = images
        self.masks = masks
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.images[idx].float() / 255.0  # 3,H,W
        mask = self.masks[idx].float() / 255.0  # 1,H,W

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
