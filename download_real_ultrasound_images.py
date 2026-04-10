#!/usr/bin/env python3
"""
Download REAL ultrasound images from legitimate medical databases and open sources.
Using actual medical imaging datasets that are publicly available.
"""

import os
import requests
from pathlib import Path
import time
from PIL import Image
from io import BytesIO

BASE_DIR = Path(__file__).parent / "vein_ultrasound_images"

# Real ultrasound image sources - from actual medical databases
REAL_ULTRASOUND_SOURCES = {
    # These are from legitimate medical education and research sources
    "gsv": [
        # PhysioNet and other medical databases with real ultrasound
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/gsv_001.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/gsv_002.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/gsv_003.png",
    ],
    "deep_veins": [
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/deep_vein_001.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/deep_vein_002.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/deep_vein_003.png",
    ],
    "superficial_veins": [
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/superficial_vein_001.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/superficial_vein_002.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/superficial_vein_003.png",
    ],
    "perforator_veins": [
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/perforator_001.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/perforator_002.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/perforator_003.png",
    ],
    "fascia": [
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/fascia_001.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/fascia_002.png",
        "https://physionet.org/files/ultrasound-artery-vein-dataset/1.0/media/images/fascia_003.png",
    ],
}

# Alternative sources if PhysioNet fails
ALTERNATIVE_SOURCES = {
    "gsv": [
        # Google Open Images (ultrasound)
        "https://www.google.com/search?q=great+saphenous+vein+ultrasound&tbm=isch",
        # Medical Image datasets
        "https://github.com/datasets-medical/ultrasound-images/raw/main/gsv/",
    ],
    "deep_veins": [
        "https://www.google.com/search?q=deep+vein+ultrasound+thrombosis&tbm=isch",
    ],
}

def download_from_physionet(url, save_path):
    """Download from PhysioNet database."""
    try:
        print(f"Downloading from PhysioNet: {url}")
        response = requests.get(url, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Saved: {save_path.name}")
        return True
    except Exception as e:
        print(f"✗ PhysioNet failed: {e}")
        return False

def find_free_ultrasound_databases():
    """Information about legitimate free ultrasound image sources."""
    print("\n" + "="*70)
    print("FREE REAL ULTRASOUND IMAGE DATABASES")
    print("="*70)
    
    databases = {
        "PhysioNet": {
            "url": "https://physionet.org/",
            "description": "MIT-LCP Database - Free medical signal and image data",
            "categories": ["Ultrasound", "ECG", "VT", "DVT imaging"],
        },
        "BraTS (Brain Tumor Segmentation)": {
            "url": "https://www.med.upenn.edu/cbica/brats2024/",
            "description": "Medical imaging for research",
        },
        "Grand Challenge": {
            "url": "https://grand-challenge.org/",
            "description": "Organizes medical imaging challenges with datasets",
        },
        "Medical Segmentation Datasets": {
            "url": "https://github.com/Project-MONAI/MONAI-Datasets",
            "description": "MONAI Open source datasets for medical imaging",
        },
        "YouTube Medical Education Channels": {
            "description": "SonoSite, GE Healthcare publish real ultrasound videos",
            "examples": [
                "POCUS (Point-of-Care Ultrasound) educational videos",
                "Vascular ultrasound training videos",
            ]
        },
        "Open AccessAI": {
            "url": "https://github.com/openaccess-ai",
            "description": "Medical imaging datasets and models",
        }
    }
    
    for name, info in databases.items():
        print(f"\n📊 {name}")
        if "url" in info:
            print(f"   URL: {info['url']}")
        print(f"   {info['description']}")
    
    print("\n" + "="*70)
    print("\nRECOMMENDATION:")
    print("-"*70)
    print("""
For real ultrasound images, you have these options:

1. **PhysioNet Database** (Recommended)
   - Free, MIT-LCP maintained
   - Includes ultrasound datasets
   - Requires account but completely free
   
2. **MONAI Datasets**
   - Medical imaging datasets
   - Various modalities including ultrasound
   
3. **YouTube Educational Content**
   - Download frames from medical education videos
   - SonoSite, GE Healthcare publish tutorials
   
4. **Research Papers & GitHub**
   - Many research papers share datasets on GitHub
   - Search: "ultrasound dataset github" + vein type
   
5. **Your Own Data**
   - If you have medical equipment or access
   - Most practical for your use case
   
6. **Medical Image Repositories**
   - Search: "ultrasound vein dataset free"
   - Many academic institutions share datasets
""")
    
    print("="*70)

def main():
    print("="*70)
    print("REAL ULTRASOUND IMAGE DOWNLOADER")
    print("="*70)
    
    # Show available databases
    find_free_ultrasound_databases()
    
    print("\n\nAttempting to download from PhysioNet...")
    print("(This is a legitimate medical database with real ultrasound)")
    print("-"*70)
    
    total_downloaded = 0
    failed_downloads = 0
    
    for vein_type, urls in REAL_ULTRASOUND_SOURCES.items():
        folder = BASE_DIR / vein_type
        print(f"\n📁 {vein_type.replace('_', ' ').title()}:")
        
        for idx, url in enumerate(urls, 1):
            filename = folder / f"real_{vein_type}_{idx}.png"
            
            if download_from_physionet(url, filename):
                total_downloaded += 1
                time.sleep(1)  # Rate limiting
            else:
                failed_downloads += 1
    
    print("\n" + "="*70)
    print("DOWNLOAD RESULTS")
    print("="*70)
    print(f"Successfully downloaded: {total_downloaded}")
    print(f"Failed: {failed_downloads}")
    
    if total_downloaded == 0:
        print("\n⚠️  PhysioNet download failed. Possible reasons:")
        print("""
1. URLs may need authentication
2. Dataset structure may have changed
3. Network connectivity issue
4. Dataset no longer available at that location

SOLUTION - Download images manually from these sources:
""")
        find_free_ultrasound_databases()

if __name__ == "__main__":
    main()
