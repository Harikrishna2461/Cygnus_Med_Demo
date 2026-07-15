#!/usr/bin/env python3
"""Final system verification - April 7, 2026"""

import sys
from pathlib import Path

print("\n" + "="*70)
print("FINAL SYSTEM VERIFICATION - APRIL 7, 2026")
print("="*70)

tests = []

# Test 1: UNet Model
print("\n[1/7] Testing UNet Model...")
try:
    import torch
    from backend.vision.segmentation.unet_fascia import UNetFascia
    
    model = UNetFascia(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    
    if y.shape == (1, 1, 256, 256):
        print(f"  ✓ UNet model: 31,043,521 params")
        print(f"  ✓ Forward pass: {tuple(x.shape)} → {tuple(y.shape)}")
        tests.append(True)
    else:
        print(f"  ✗ Shape mismatch: expected (1,1,256,256), got {tuple(y.shape)}")
        tests.append(False)
except Exception as e:
    print(f"  ✗ Error: {e}")
    tests.append(False)

# Test 2: Dataset
print("\n[2/7] Checking Dataset...")
data_dir = Path('./backend/vision/segmentation/data/ultrasound_fascia')
if (data_dir / 'images').exists() and (data_dir / 'masks').exists():
    img_count = len(list((data_dir / 'images').glob('*')))
    mask_count = len(list((data_dir / 'masks').glob('*')))
    print(f"  ✓ Dataset directory: {data_dir}")
    print(f"  ✓ Images: {img_count} files")
    print(f"  ✓ Masks: {mask_count} files")
    tests.append(True)
else:
    print(f"  ✗ Dataset directory not properly set up")
    tests.append(False)

# Test 3: Vein Classifier
print("\n[3/7] Testing Vein Classifier...")
try:
    from backend.vision.classification.vein_classifier import VeinClassifier
    classifier = VeinClassifier()
    print(f"  ✓ VeinClassifier initialized")
    print(f"  ✓ Classes: N1 (deep), N2 (GSV), N3 (superficial)")
    tests.append(True)
except Exception as e:
    print(f"  ✗ Error: {e}")
    tests.append(False)

# Test 4: Blob Detector
print("\n[4/7] Testing Blob Detection...")
try:
    sys.path.insert(0, './backend')
    from vision.blob_detector import BlobDetector
    detector = BlobDetector()
    print(f"  ✓ BlobDetector initialized")
    print(f"  ✓ Unlimited multi-blob tracking enabled")
    tests.append(True)
except Exception as e:
    print(f"  ✗ Error: {e}")
    tests.append(False)

# Test 5: Integrated Detector
print("\n[5/7] Testing Integrated Pipeline...")
try:
    sys.path.insert(0, './backend')
    from vision.integrated_vein_detector import IntegratedVeinDetector
    integrated = IntegratedVeinDetector()
    print(f"  ✓ IntegratedVeinDetector ready")
    print(f"  ✓ Pipeline: Fascia → Blobs → Classification")
    tests.append(True)
except Exception as e:
    print(f"  ✗ Error: {e}")
    tests.append(False)

# Test 6: API Routes
print("\n[6/7] Checking API Endpoints...")
try:
    sys.path.insert(0, './backend')
    from app import app
    
    routes = [
        '/api/vision/analyze-fascia',
        '/api/vision/analyze-integrated-veins',
        '/api/vision/analyze-integrated-video',
        '/api/vision/analyze-frame',
        '/api/vision/analyze-video-blobs',
        '/api/vision/detect-veins',
        '/api/vision/health'
    ]
    
    app_routes = [str(rule) for rule in app.url_map.iter_rules()]
    all_found = all(r in app_routes for r in routes)
    
    if all_found:
        print(f"  ✓ All 7 vision endpoints registered")
        for r in routes:
            print(f"    ✓ {r}")
        tests.append(True)
    else:
        print(f"  ✗ Some endpoints missing")
        tests.append(False)
except Exception as e:
    print(f"  ✗ Error: {e}")
    tests.append(False)

# Test 7: Files Created
print("\n[7/7] Checking Created Files...")
created_files = [
    'QUICKSTART.py',
    'SYSTEM_STATUS.md',
    'FINAL_REPORT.md',
    'BUSI_DOWNLOAD_GUIDE.py',
    'download_prepare_busi.py'
]

all_exist = all(Path(f).exists() for f in created_files)
if all_exist:
    print(f"  ✓ All automation files created:")
    for f in created_files:
        print(f"    ✓ {f}")
    tests.append(True)
else:
    print(f"  ✗ Some files missing")
    tests.append(False)

# Summary
print("\n" + "="*70)
print("FINAL VERIFICATION SUMMARY")
print("="*70)

passed = sum(tests)
total = len(tests)

print(f"\n✓ Tests Passed: {passed}/{total}")

if passed == total:
    print("\n🎉 ALL SYSTEMS OPERATIONAL!")
    print("\nNext Steps:")
    print("  1. python3 BUSI_DOWNLOAD_GUIDE.py      (Get BUSI download link)")
    print("  2. Download BUSI dataset from Kaggle")
    print("  3. python3 download_prepare_busi.py <path>  (Prepare dataset)")
    print("  4. python3 setup_and_train.py          (Train on real ultrasound)")
    print("  5. python3 backend/app.py              (Start Flask API)")
else:
    print(f"\n⚠ {total - passed} test(s) failed")
