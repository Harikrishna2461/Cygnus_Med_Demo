#!/usr/bin/env python3
"""Debug edge-based fascia detection."""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, './backend')

from vision.segmentation.edge_fascia_detector import EdgeFasciaDetector


# Create test image that matches real ultrasound structure
img = np.ones((256, 256), dtype=np.uint8) * 85

# Bright tissue region
img[40:95, :] = np.random.randint(130, 160, (55, 256), dtype=np.uint8)

# Upper fascia - thin bright line (2 pixels thick)
img[100:102, 20:240] = 210
for i in range(20, 240):
    img[100:102, i] = 210 + np.random.randint(-10, 10)

# Fascia tissue region (slightly bright)
img[102:110, :] = np.random.randint(90, 120, (8, 256), dtype=np.uint8)

# Lower fascia - thin bright line (2 pixels thick, ~8 pixels below upper)
img[110:112, 20:240] = 210
for i in range(20, 240):
    img[110:112, i] = 210 + np.random.randint(-10, 10)

# Dark muscle below
img[115:, :] = np.random.randint(50, 80, (141, 256), dtype=np.uint8)

# Add speckle noise
noise = np.random.normal(0, 8, img.shape)
img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

print(f"Test image created:")
print(f"  Tissue above: y=40-95, intensity ~145")
print(f"  Upper fascia: y=100-102, intensity ~210")
print(f"  Fascia band: y=102-110, intensity ~105")
print(f"  Lower fascia: y=110-112, intensity ~210")
print(f"  Muscle below: y=115+, intensity ~65")
print(f"  Expected spacing: ~10 pixels")

# Convert to BGR (as if from camera)
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

detector = EdgeFasciaDetector()

# Debug step 1: Enhance
enhanced = detector.enhance_image(img_bgr)
print(f"\n1. Enhanced image: range=[{enhanced.min()}, {enhanced.max()}], mean={enhanced.mean():.1f}")

# Check intensity at key rows
print(f"\n   Intensity profile:")
for y in [50, 95, 100, 101, 110, 111, 115]:
    intensity = enhanced[y, 100]
    print(f"   y={y}: {intensity}")

# Debug step 2: Band region
y_start, y_end = detector.find_fascia_band_region(enhanced)
print(f"\n2. Band region: y=[{y_start}, {y_end}], height={y_end-y_start}")

# Check texture profile
laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
laplacian = np.abs(laplacian)
texture_per_row = np.mean(laplacian, axis=1)
print(f"   Texture (Laplacian) profile:")
for y in [50, 95, 100, 105, 110, 115]:
    print(f"   y={y}: {texture_per_row[y]:.1f}")

# Debug step 3: Find edges
upper = detector.find_boundary_line(enhanced, y_start, y_end, find_upper=True)
lower = detector.find_boundary_line(enhanced, y_start, y_end, find_upper=False)

print(f"\n3. Boundary lines:")
if upper is not None:
    print(f"   Upper: {len(upper)} points, y=[{upper[:, 1].min()}, {upper[:, 1].max()}]")
    print(f"           mean y={upper[:, 1].mean():.1f}")
else:
    print(f"   Upper: NOT FOUND")

if lower is not None:
    print(f"   Lower: {len(lower)} points, y=[{lower[:, 1].min()}, {lower[:, 1].max()}]")
    print(f"           mean y={lower[:, 1].mean():.1f}")
else:
    print(f"   Lower: NOT FOUND")

# Debug step 4: Validation
if upper is not None and lower is not None:
    spacing = lower[:, 1].astype(float) - upper[:, 1].astype(float)
    print(f"\n4. Spacing analysis:")
    print(f"   Mean: {spacing.mean():.1f} (expected ~10)")
    print(f"   Std: {spacing.std():.1f}")
    print(f"   Range: [{spacing.min():.1f}, {spacing.max():.1f}]")
    
    confidence = detector.validate_boundaries(enhanced, upper, lower)
    print(f"   Confidence: {confidence:.3f}")

# Full detection
print(f"\n5. Full detection result:")
result = detector.detect(img_bgr)
print(f"   Status: {result['status']}")
print(f"   Confidence: {result['confidence']:.3f}")

if result['upper_edge'] is not None:
    up = result['upper_edge']
    lo = result['lower_edge']
    sp = lo[:, 1] - up[:, 1]
    print(f"   Final upper y: [{up[:, 1].min()}, {up[:, 1].max()}]")
    print(f"   Final lower y: [{lo[:, 1].min()}, {lo[:, 1].max()}]")
    print(f"   Final spacing: {sp.mean():.1f} ± {sp.std():.1f}")
