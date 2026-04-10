"""
Ultrasound dataset handler for fascia segmentation.
Works with public ultrasound datasets.
"""

import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
import tarfile
import zipfile
import os


class UltrasoundDataset(Dataset):
    """
    Generic ultrasound dataset for fascia segmentation.
    Expects directory structure:
        data/
            images/
                *.jpg, *.png, *.tiff
            masks/
                *.jpg, *.png, *.tiff (corresponding masks)
    """
    
    def __init__(self, data_dir, split='train', transform=None, test_split=0.1):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Get image files
        image_dir = self.data_dir / 'images'
        mask_dir = self.data_dir / 'masks'
        
        if not image_dir.exists() or not mask_dir.exists():
            raise RuntimeError(f"Expected directories: {image_dir}, {mask_dir}")
        
        # Load image paths
        image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.tiff', '.bmp']])
        
        # Split into train/test
        n_test = max(1, int(len(image_files) * test_split))
        test_indices = set(range(0, len(image_files), len(image_files) // n_test)[:n_test])
        
        if split == 'train':
            self.image_files = [f for i, f in enumerate(image_files) if i not in test_indices]
        else:
            self.image_files = [f for i, f in enumerate(image_files) if i in test_indices]
        
        self.mask_dir = mask_dir
        print(f"✓ {split} dataset: {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (should be grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Create zero mask if not found
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Normalize mask to 0-1
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            
            mask = torch.from_numpy(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return image, mask


class DatasetDownloader:
    """Download and prepare public ultrasound datasets."""
    
    @staticmethod
    def download_busi_dataset(output_dir='./data/busi'):
        """
        Download BUSI (Breast Ultrasound Images) dataset.
        Contains breast ultrasound images with segmentation masks.
        
        Source: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
        
        Note: Requires manual download from Kaggle due to license restrictions.
        Download and extract to output_dir/busi
        """
        print("BUSI dataset requires manual download from Kaggle:")
        print("1. Go to: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print("2. Download the dataset")
        print(f"3. Extract to: {output_dir}")
        print("\nDataset structure expected:")
        print(f"  {output_dir}/")
        print("    images/")
        print("    masks/")
        
        return output_dir
    
    @staticmethod
    def available_datasets():
        """List publicly available ultrasound datasets."""
        datasets = {
            'BUSI': 'https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset',
            'IEEE DataPort': 'https://ieee-dataport.org/',
            'Grand Challenge': 'https://grand-challenge.org/',
            'PhysioNet': 'https://physionet.org/',
        }
        return datasets


def create_dataloaders(data_dir, batch_size=8, num_workers=2, augment=True):
    """
    Create train and test dataloaders.
    
    Args:
        data_dir: path to dataset directory
        batch_size: batch size
        num_workers: number of workers for data loading
        augment: whether to apply augmentation
    
    Returns:
        train_loader, test_loader
    """
    try:
        import albumentations as A
        augmentation = A.Compose([
            A.Rotate(limit=20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2(),
        ], is_check_shapes=True)
    except ImportError:
        augmentation = None
        print("⚠ albumentations not installed, skipping augmentation")
    
    train_dataset = UltrasoundDataset(data_dir, split='train', transform=augmentation if augment else None)
    test_dataset = UltrasoundDataset(data_dir, split='test', transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


if __name__ == '__main__':
    # REAL DATA ONLY - NO SYNTHETIC DATA
    data_dir_path = Path('./data/ultrasound_fascia')
    
    if not data_dir_path.exists() or not (data_dir_path / 'images').exists():
        print("\n✗ FATAL: Real ultrasound dataset NOT FOUND")
        print("\nTo train on REAL data:")
        print("  1. Download BUSI: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print("  2. Run: python3 download_prepare_busi.py <path_to_dataset>")
        print("  3. Then: python3 setup_and_train.py")
        print("\nNo synthetic data will be created.")
        import sys
        sys.exit(1)
    
    # Create dataloaders from real data
    train_loader, test_loader = create_dataloaders(str(data_dir_path), batch_size=8)
    
    print(f"✓ Dataloaders created from REAL dataset")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
