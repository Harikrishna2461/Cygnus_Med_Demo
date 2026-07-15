#!/usr/bin/env python3
"""Test the integrated vision pipeline with improved fascia detector."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/backend')

from vision.segmentation.unet_detector import PretrainedSegmentationDetector

# Load reference image
img = cv2.imread('/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png')
if img is None:
    print("Image not found")
    sys.exit(1)

print(f"Image shape: {img.shape}")

# Test the integrated detector
print("\nTesting PretrainedSegmentationDetector (now using EdgeFasciaDetector)...")
detector = PretrainedSegmentationDetector()
fascia_mask = detector.segment_fascia(img)

# Check mask
if fascia_mask.max() > 0:
    non_zero = np.count_nonzero(fascia_mask)
    print(f"\n✓ Fascia mask created:")
    print(f"  Non-zero pixels: {non_zero}")
    print(f"  Mask shape: {fascia_mask.shape}")
    
    # Save visualization
    img_color = img.copy()
    # Show fascia region in green
    fascia_region = np.where(fascia_mask > 0)
    img_color[fascia_region[0], fascia_region[1]] = [0, 255, 0]  # Green
    
    cv2.imwrite('/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/pipeline_test_fascia.png', img_color)
    print(f"\n✓ Visualization saved to pipeline_test_fascia.png")
else:
    print(f"✗ No fascia mask created")
