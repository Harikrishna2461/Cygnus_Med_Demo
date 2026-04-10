#!/usr/bin/env python3
"""
Download and prepare BUSI (Breast Ultrasound Images) dataset for training.
This is a real public dataset with actual ultrasound images.

BUSI Dataset: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from urllib.request import urlopen
import zipfile
import shutil


def download_busi_instructions():
    """Print instructions for downloading BUSI dataset."""
    print("\n" + "="*70)
    print("BUSI DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nThe BUSI (Breast Ultrasound Images) dataset is the recommended")
    print("public ultrasound dataset for training the fascia detection model.")
    print("\n1. Visit Kaggle dataset:")
    print("   https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
    print("\n2. Create free Kaggle account (if needed)")
    print("\n3. Click 'Download' button")
    print("\n4. Extract the downloaded ZIP file")
    print("\n5. Run: python3 prepare_busi_dataset.py <path_to_extracted_folder>")
    print("\nExample:")
    print("   python3 prepare_busi_dataset.py ~/Downloads/archive/")
    print("\n" + "="*70)


def prepare_busi_dataset(busi_path, output_dir='./backend/vision/segmentation/data/ultrasound_fascia'):
    """
    Prepare BUSI dataset for training.
    
    Args:
        busi_path: path to extracted BUSI dataset
        output_dir: output directory for train/test split
    """
    busi_path = Path(busi_path)
    output_dir = Path(output_dir)
    
    if not busi_path.exists():
        print(f"\n✗ Error: path does not exist: {busi_path}")
        return False
    
    # Create output structure
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Preparing BUSI dataset")
    print(f"{'='*70}")
    print(f"Source: {busi_path}")
    print(f"Output: {output_dir}")
    
    # Find image files
    image_count = 0
    mask_count = 0
    
    # BUSI structure: <category>/images and <category>/masks folders
    for category_dir in busi_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        images_src = category_dir / 'images'
        masks_src = category_dir / 'masks'
        
        if not images_src.exists():
            continue
        
        print(f"\nProcessing: {category_dir.name}")
        
        # Copy images
        for img_file in images_src.glob('*.png'):
            dest = images_dir / f"{category_dir.name}_{img_file.name}"
            shutil.copy2(img_file, dest)
            image_count += 1
        
        # Copy masks (if available) or create placeholder
        if masks_src.exists():
            for mask_file in masks_src.glob('*.png'):
                dest = masks_dir / f"{category_dir.name}_{mask_file.name}"
                shutil.copy2(mask_file, dest)
                mask_count += 1
        else:
            # Create placeholder masks (user will need to annotate)
            for img_file in images_src.glob('*.png'):
                mask_path = masks_dir / f"{category_dir.name}_{img_file.name}"
                # Create black mask (user needs to annotate)
                mask = np.zeros((512, 640), dtype=np.uint8)
                cv2.imwrite(str(mask_path), mask)
                mask_count += 1
    
    print(f"\n✓ Dataset preparation complete:")
    print(f"  Images copied: {image_count}")
    print(f"  Masks: {mask_count}")
    
    if mask_count < image_count:
        print(f"\n⚠ Warning: BUSI masks are not available in original dataset")
        print(f"  You need to annotate the fascia boundaries in each image.")
        print(f"\n  Masks directory created at: {masks_dir}")
        print(f"  Annotation tools:")
        print(f"    - http://labelimg.csail.mit.edu/")
        print(f"    - https://www.makesense.ai/")
        print(f"    - https://www.cvat.ai/")
        print(f"\n  Steps:")
        print(f"    1. Open annotation tool")
        print(f"    2. Load images from: {images_dir}")
        print(f"    3. Draw fascia boundary (polygon or line)")
        print(f"    4. Export as binary masks to: {masks_dir}")
    
    return True


def get_alternative_datasets():
    """List alternative public ultrasound datasets."""
    print("\n" + "="*70)
    print("ALTERNATIVE PUBLIC ULTRASOUND DATASETS")
    print("="*70)
    print("\n1. IEEE DataPort")
    print("   - https://ieee-dataport.org/")
    print("   - Multiple ultrasound collections")
    print("   - Some are free/public")
    print("\n2. Grand Challenge Datasets")
    print("   - https://grand-challenge.org/")
    print("   - Many medical imaging datasets")
    print("   - Filter by 'ultrasound'")
    print("\n3. OpenNeuro")
    print("   - https://openneuro.org/")
    print("   - Open access neuroimaging data")
    print("   - Some ultrasound collections")
    print("\n4. PhysioNet")
    print("   - https://physionet.org/")
    print("   - Clinical datasets (ECG, etc)")
    print("   - Some ultrasound collections")
    print("\n5. GitHub Medical Datasets")
    print("   - Search: 'ultrasound dataset github'")
    print("   - Various research team datasets")
    print("\n" + "="*70)


def main():
    if len(sys.argv) < 2:
        download_busi_instructions()
        get_alternative_datasets()
        return 1
    
    busi_path = sys.argv[1]
    
    if not prepare_busi_dataset(busi_path):
        return 1
    
    print("\n" + "="*70)
    print("✓ BUSI dataset prepared successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. (Optional) Annotate fascia boundaries if masks are missing")
    print("  2. Run: python3 setup_and_train.py")
    print("\n" + "="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
