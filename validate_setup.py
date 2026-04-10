#!/usr/bin/env python3
"""
Quick validation script for the vision pipeline setup
Tests that all dependencies are installed and basic functionality works
"""

import sys
from pathlib import Path

print("=" * 70)
print("VISION PIPELINE SETUP VALIDATION")
print("=" * 70)

# Test 1: Python version
print("\n✓ Test 1: Python Version")
print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Check required packages
print("\n✓ Test 2: Required Packages")
required_packages = [
    "cv2",
    "numpy",
    "torch",
    "torchvision",
    "PIL",
    "scipy",
    "flask",
    "requests"
]

missing_packages = []
for package in required_packages:
    try:
        if package == "cv2":
            import cv2
        elif package == "PIL":
            from PIL import Image
        else:
            __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - NOT FOUND")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")
else:
    print("  All packages installed!")

# Test 3: Check test images
print("\n✓ Test 3: Test Images")
test_images_dir = Path(__file__).parent / "vein_ultrasound_images"
if test_images_dir.exists():
    categories = {
        "gsv": 0,
        "deep_veins": 0,
        "superficial_veins": 0,
        "perforator_veins": 0,
        "fascia": 0
    }
    
    for category in categories:
        images = list((test_images_dir / category).glob("*.png"))
        categories[category] = len(images)
        print(f"  {category:.<25} {len(images)} images")
    
    total = sum(categories.values())
    print(f"  {'Total':.<25} {total} images")
    
    if total > 0:
        print("  ✓ Test images found!")
    else:
        print("  ✗ No test images found - run download_ultrasound_samples.py")
else:
    print(f"  ✗ Test images directory not found: {test_images_dir}")

# Test 4: Vision module structure
print("\n✓ Test 4: Vision Module Structure")
vision_dir = Path(__file__).parent / "backend" / "vision"
required_modules = [
    "config.py",
    "vision_main.py",
    "video/frame_extractor.py",
    "segmentation/sam_wrapper.py",
    "geometry/spatial_analysis.py",
    "classification/rules.py",
    "classification/llm_interface.py",
    "utils/visualization.py",
]

all_modules_found = True
for module in required_modules:
    module_path = vision_dir / module
    if module_path.exists():
        print(f"  ✓ {module}")
    else:
        print(f"  ✗ {module} - NOT FOUND")
        all_modules_found = False

if not all_modules_found:
    print("  Some modules missing!")

# Test 5: Try importing vision module
print("\n✓ Test 5: Vision Module Import")
sys.path.insert(0, str(Path(__file__).parent / "backend"))

try:
    from vision import config
    print("  ✓ vision.config imported")
except Exception as e:
    print(f"  ✗ vision.config import failed: {e}")

try:
    from vision.video.frame_extractor import FrameExtractor
    print("  ✓ FrameExtractor imported")
except Exception as e:
    print(f"  ✗ FrameExtractor import failed: {e}")

try:
    from vision.geometry.spatial_analysis import SpatialAnalyzer
    print("  ✓ SpatialAnalyzer imported")
except Exception as e:
    print(f"  ✗ SpatialAnalyzer import failed: {e}")

try:
    from vision.classification.rules import VeinClassifier
    print("  ✓ VeinClassifier imported")
except Exception as e:
    print(f"  ✗ VeinClassifier import failed: {e}")

try:
    from vision.utils.visualization import UltrasoundVisualizer
    print("  ✓ UltrasoundVisualizer imported")
except Exception as e:
    print(f"  ✗ UltrasoundVisualizer import failed: {e}")

# SAM model note
print("\n⚠️  Note: SAM Model Download")
print("  The SAM (Segment Anything Model) requires manual download:")
print("  Run the following command:")
print("")
print("  mkdir -p backend/weights")
print("  wget -O backend/weights/sam_vit_b.pth \\")
print("    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b.pth")
print("")
print("  Or for faster processing (mobile_sam):")
print("  wget -O backend/weights/mobile_sam.pt \\")
print("    https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt")

# Final summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

if not missing_packages and all_modules_found:
    print("✅ Setup looks good! Ready to test the vision pipeline.")
    print("\nNext steps:")
    print("1. Download SAM model (see instructions above)")
    print("2. Run: python backend/vision/examples.py")
    print("3. Test API: curl http://localhost:5002/api/vision/health")
else:
    print("⚠️  Some issues detected. Please fix before proceeding.")

print("=" * 70)
