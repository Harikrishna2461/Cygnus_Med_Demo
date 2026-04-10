#!/usr/bin/env python3
"""
Download sample ultrasound images for testing the vein detection pipeline.
Uses publicly available medical images from Wikimedia Commons and other sources.
"""

import os
import requests
from pathlib import Path
import time

# Base directory for downloaded images
BASE_DIR = Path(__file__).parent / "vein_ultrasound_images"

# Image URLs for different vein types
# Using Wikimedia Commons and publicly available ultrasound images
IMAGE_URLS = {
    "gsv": [
        # Great Saphenous Vein samples (publicly available medical images)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Ultrasound_Femoral_Artery_%26_Vein.jpg/640px-Ultrasound_Femoral_Artery_%26_Vein.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Ultrasound_B_mode_GSV.jpg/640px-Ultrasound_B_mode_GSV.jpg",
    ],
    "deep_veins": [
        # Deep vein ultrasound images
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Ultrasound_iliofemoral_veins.jpg/640px-Ultrasound_iliofemoral_veins.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Ultrasound_femoral_vein_DVT_exam.jpg/640px-Ultrasound_femoral_vein_DVT_exam.jpg",
    ],
    "superficial_veins": [
        # Superficial vein samples
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Ultrasound_superficial_leg_veins.jpg/640px-Ultrasound_superficial_leg_veins.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Ultrasound_calf_veins.jpg/640px-Ultrasound_calf_veins.jpg",
    ],
    "perforator_veins": [
        # Perforator vein samples
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Ultrasound_perforator_veins.jpg/640px-Ultrasound_perforator_veins.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Ultrasound_thigh_perforators.jpg/640px-Ultrasound_thigh_perforators.jpg",
    ],
    "fascia": [
        # Fascial layer samples
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Ultrasound_leg_anatomy.jpg/640px-Ultrasound_leg_anatomy.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Ultrasound_fascial_layers.jpg/640px-Ultrasound_fascial_layers.jpg",
    ],
}

def download_image(url, save_path):
    """Download a single image from URL and save it."""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=10, allow_redirects=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Saved: {save_path.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False

def create_sample_images():
    """Create sample ultrasound-like images if downloads fail."""
    print("\nCreating synthetic sample images...")
    import numpy as np
    from PIL import Image, ImageDraw
    
    # Create sample ultrasound-like images (grayscale with texture)
    def create_ultrasound_image(vein_type, filename):
        # Create ultrasound-like appearance with noise and gradient
        img_array = np.random.randint(20, 80, (480, 640, 1), dtype=np.uint8)
        
        # Add some structure/artifacts
        y, x = np.ogrid[:480, :640]
        
        if vein_type == "gsv":
            # Circular vein pattern
            mask = ((y - 240)**2 + (x - 320)**2 < 4000) & ((y - 240)**2 + (x - 320)**2 > 2500)
            img_array[mask] = 150
            
        elif vein_type == "deep_veins":
            # Multiple vein patterns
            mask1 = ((y - 200)**2 + (x - 250)**2 < 3000) & ((y - 200)**2 + (x - 250)**2 > 2000)
            mask2 = ((y - 300)**2 + (x - 400)**2 < 3000) & ((y - 300)**2 + (x - 400)**2 > 2000)
            img_array[mask1] = 140
            img_array[mask2] = 135
            
        elif vein_type == "superficial_veins":
            # Elongated near-surface pattern
            mask = (np.abs(y - 150) < 30) & (x > 200) & (x < 600)
            img_array[mask] = 120
            
        elif vein_type == "perforator_veins":
            # Crossing/intersecting pattern
            mask1 = (np.abs(y - 240) < 20) & (x > 200) & (x < 600)
            mask2 = ((x - 400) > 0) & ((x - 400) < 120) & (y > 100) & (y < 380)
            img_array[mask1] = 130
            img_array[mask2] = 125
            
        elif vein_type == "fascia":
            # Horizontal fascial line
            mask = (np.abs(y - 200) < 10) & (x > 100) & (x < 600)
            img_array[mask] = 180
        
        # Convert to PIL Image and save
        img = Image.fromarray(img_array[:,:,0], mode='L')
        img.save(filename)
        return filename
    
    for vein_type in IMAGE_URLS.keys():
        folder = BASE_DIR / vein_type
        for i in range(2):
            filename = folder / f"sample_{vein_type}_{i+1}.png"
            create_ultrasound_image(vein_type, str(filename))
            print(f"✓ Created: {filename.name}")

def main():
    """Main download function."""
    print("=" * 70)
    print("Ultrasound Image Downloader for Vein Detection Pipeline")
    print("=" * 70)
    
    total_downloaded = 0
    failed_downloads = 0
    
    # Try to download images
    for vein_type, urls in IMAGE_URLS.items():
        folder = BASE_DIR / vein_type
        print(f"\n📁 Processing {vein_type.replace('_', ' ').title()}:")
        print(f"   Folder: {folder}")
        
        for idx, url in enumerate(urls, 1):
            filename = folder / f"{vein_type}_{idx}.png"
            
            if filename.exists():
                print(f"✓ Already exists: {filename.name}")
                total_downloaded += 1
            else:
                if download_image(url, filename):
                    total_downloaded += 1
                    time.sleep(0.5)  # Rate limiting
                else:
                    failed_downloads += 1
    
    # If some downloads failed, create synthetic samples
    if failed_downloads > 0:
        print(f"\n⚠️  {failed_downloads} downloads failed. Creating synthetic samples...")
        create_sample_images()
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully processed: {total_downloaded} images")
    print(f"✗ Failed downloads: {failed_downloads} images")
    print(f"\n📂 Image directory structure created:")
    for vein_type in IMAGE_URLS.keys():
        folder = BASE_DIR / vein_type
        image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        print(f"   {vein_type:.<30} {len(image_files)} images")
    
    print(f"\n🎯 Base directory: {BASE_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
