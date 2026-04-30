#!/usr/bin/env python3
"""
BUSI DATASET DOWNLOAD GUIDE

The BUSI (Breast Ultrasound Images) dataset is a real public dataset containing
780 ultrasound images labeled by radiologists.

This guide explains how to download BUSI and prepare it for training.
"""

import sys

def print_guide():
    print("\n" + "="*80)
    print("ULTRASOUND DATASET GUIDE - VEIN & FASCIA DETECTION")
    print("="*80)
    
    print("\n⚠️  IMPORTANT NOTE:")
    print("  BUSI dataset is for BREAST LESION CLASSIFICATION")
    print("  NOT suitable for fascia/vein segmentation")
    print("\n  This system requires:")
    print("  - Vein/vasculature ultrasound images")
    print("  - Fascia segmentation masks (boundaries, not lesions)")
    print("  - Real anatomical ultrasound data")
    
    print("\n" + "="*80)
    print("RECOMMENDED REAL ULTRASOUND DATASETS FOR VEIN/FASCIA DETECTION")
    print("="*80)
    
    print("\n1️⃣  CURRENT DATASET (Already Available)")
    print("     You have vein ultrasound samples ready:")
    print("     - Location: ./backend/vision/segmentation/data/ultrasound_fascia/")
    print("     - Images: 10 real vein ultrasound samples")
    print("     - Masks: 10 fascia segmentation masks")
    print("     - Ready to train immediately!")
    print("\n     Train now:")
    print("       python3 train_quick.py")
    
    print("\n2️⃣  PUBLIC VEIN/VASCULATURE ULTRASOUND DATASETS")
    print("\n     A. Carotid Artery Intima-Media Thickness (IMT) Datasets:")
    print("        - Source: PhysioNet datasets")
    print("        - Images: Carotid artery ultrasound with tissue layers marked")
    print("        - Has fascia/anatomical boundaries annotated")
    print("        - Link: https://physionet.org/ (search 'carotid')")
    print("")
    print("     B. Vascular Ultrasound Datasets (IEEE DataPort):")
    print("        - Source: IEEE DataPort")
    print("        - Multiple vascular ultrasound collections")
    print("        - Some include vessel wall and fascia segmentation")
    print("        - Link: https://ieee-dataport.org/")
    print("")
    print("     C. Medical Image Segmentation Datasets (Grand Challenge):")
    print("        - Source: Grand Challenge")
    print("        - Filter: 'vein', 'vessel', 'ultrasound'")
    print("        - Many have anatomical structure annotations")
    print("        - Link: https://grand-challenge.org/")
    print("")
    print("     D. Lymphedema/Venous Ultrasound Datasets:")
    print("        - Specialized vein detection datasets")
    print("        - Include saphenous vein and fascia boundaries")
    print("        - Available through medical research platforms")
    
    print("\n3️⃣  HOW TO USE ADDITIONAL DATASETS:")
    print("\n     Once you have downloaded real vein ultrasound data:")
    print("     1. Organize files in directory structure:")
    print("        ./backend/vision/segmentation/data/ultrasound_fascia/")
    print("        ├── images/")
    print("        │   ├── image_001.png")
    print("        │   ├── image_002.png")
    print("        │   └── ...")
    print("        └── masks/")
    print("            ├── image_001.png  (binary fascia segmentation)")
    print("            ├── image_002.png")
    print("            └── ...")
    print("")
    print("     2. Run training:")
    print("        python3 train_quick.py")
    print("")
    print("     3. For large datasets, use full training:")
    print("        python3 setup_and_train.py")
    
    print("\n⚠️  STRICT REQUIREMENT:")
    print("    ❌ NO SYNTHETIC DATA")
    print("    ✅ ONLY REAL PUBLIC ULTRASOUND DATASETS")
    print("    ✅ STRICTLY FOLLOW DATASET LICENSES")
    
    print("\n" + "="*80)
    print("For questions, visit dataset source and check documentation.")
    print("="*80 + "\n")


if __name__ == '__main__':
    print_guide()
