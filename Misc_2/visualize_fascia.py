#!/usr/bin/env python3
"""Visualize the detected fascia on the ultrasound image."""

import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/backend')

from vision.segmentation.edge_fascia_detector import EdgeFasciaDetector

# Load image
img = cv2.imread('/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png')

print(f"Image shape: {img.shape}")

# Detect fascia
detector = EdgeFasciaDetector()
result = detector.detect(img)

# Draw fascia lines
if result['status'] == 'success':
    upper = result['upper_edge']
    lower = result['lower_edge']
    
    # Draw lines on color image
    points_upper = [(int(upper[i, 0]), int(upper[i, 1])) for i in range(len(upper))]
    points_lower = [(int(lower[i, 0]), int(lower[i, 1])) for i in range(len(lower))]
    
    for i in range(len(points_upper) - 1):
        cv2.line(img_color, points_upper[i], points_upper[i+1], (0, 255, 255), 2)  # Yellow
    
    for i in range(len(points_lower) - 1):
        cv2.line(img_color, points_lower[i], points_lower[i+1], (0, 255, 255), 2)  # Yellow
    
    # Save result
    cv2.imwrite('/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/fascia_test_result.png', img_color)
    print(f"\n✓ Visualization saved to fascia_test_result.png")
    print(f"  Upper y: {upper[0, 1]:.0f}")
    print(f"  Lower y: {lower[0, 1]:.0f}")
    print(f"  Spacing: {lower[0, 1] - upper[0, 1]:.0f} pixels")
else:
    print(f"✗ Detection failed: {result['status']}")
