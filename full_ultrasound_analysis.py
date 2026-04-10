#!/usr/bin/env python3
"""
Analyze the entire ultrasound image to find all bright structures.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

image = cv2.imread('real_ultrasound.png')
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

h, w = gray.shape
print(f"Image: {w}x{h}, range [{gray.min()}, {gray.max()}]")

# Enhance
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
enhanced = cv2.bilateralFilter(enhanced.astype(np.uint8), 5, 75, 75)

print(f"Enhanced: range [{enhanced.min()}, {enhanced.max()}]")

# Analyze brightness profile across entire image
brightness_per_row = np.mean(enhanced, axis=1)

# Find peaks
brightness_smooth = gaussian_filter1d(brightness_per_row.astype(float), sigma=2)
gradient = np.diff(brightness_smooth)
sign_change = np.diff(np.sign(gradient))
peaks = np.where(sign_change < 0)[0]

print(f"\nFound {len(peaks)} total peaks:")
if len(peaks) > 0:
    peak_brightness = brightness_per_row[peaks]
    sorted_indices = np.argsort(peak_brightness)[::-1]
    for idx in sorted_indices[:10]:
        peak_idx = peaks[idx]
        print(f"  Row {peak_idx:3d}: brightness={brightness_per_row[peak_idx]:7.1f}")

# Find pairs of peaks with typical fascia spacing
if len(peaks) > 1:
    print(f"\nFascia candidates (3-20 pixel spacing):")
    for i in range(len(peaks)-1):
        spacing = peaks[i+1] - peaks[i]
        if 3 <= spacing <= 20:
            b1 = brightness_per_row[peaks[i]]
            b2 = brightness_per_row[peaks[i+1]]
            print(f"  y=[{peaks[i]}, {peaks[i+1]}], spacing={spacing}, brightness=[{b1:.0f}, {b2:.0f}]")

# Center region analysis
print(f"\nCenter region (y={h//3} to y={2*h//3}):")
center_start = h // 3
center_end = 2 * h // 3
center_peaks = peaks[(peaks >= center_start) & (peaks < center_end)]
print(f"  Peaks in center: {len(center_peaks)}")
for peak_idx in center_peaks:
    print(f"    y={peak_idx}: brightness={brightness_per_row[peak_idx]:.1f}")
