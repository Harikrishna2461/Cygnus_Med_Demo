#!/usr/bin/env python3
"""Test the updated fascia detector on reference image."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/backend')

from vision.segmentation.edge_fascia_detector import EdgeFasciaDetector

# Load the reference image
img = cv2.imread('/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Image not found")
    sys.exit(1)

print(f"Image shape: {img.shape}\n")

# Create detector and test
detector = EdgeFasciaDetector()
result = detector.detect(img)

if result['status'] == 'success':
    upper = result['upper_edge']
    lower = result['lower_edge']
    print(f"✓ Fascia detected successfully!")
    print(f"  Upper y-coord: {upper[0, 1]:.1f}")
    print(f"  Lower y-coord: {lower[0, 1]:.1f}")
    print(f"  Spacing: {lower[0, 1] - upper[0, 1]:.1f} pixels")
    print(f"  Confidence: {result['confidence']:.2f}")
else:
    print(f"✗ No fascia detected: {result['status']}")
