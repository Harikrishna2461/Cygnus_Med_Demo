#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

image = cv2.imread('real_ultrasound.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Analyze ENTIRE image
brightness_per_row = np.mean(gray, axis=1)
brightness_smooth = gaussian_filter1d(brightness_per_row.astype(float), sigma=2)

# Find all peaks
gradient = np.diff(brightness_smooth)
sign_change = np.diff(np.sign(gradient))
peaks = np.where(sign_change < 0)[0]

print(f"All peaks in entire image:")
print(f"  Total peaks: {len(peaks)}")
for i, peak in enumerate(peaks):
    print(f"  Peak {i}: y={peak}, brightness={brightness_per_row[peak]:.1f}")

print(f"\nAll peak pairs with spacing >= 20 pixels:")
for i in range(len(peaks)-1):
    spacing = peaks[i+1] - peaks[i]
    if spacing >= 20:
        b1 = brightness_per_row[peaks[i]]
        b2 = brightness_per_row[peaks[i+1]]
        print(f"  y=[{peaks[i]}, {peaks[i+1]}] spacing={spacing} brightness=[{b1:.1f}, {b2:.1f}]")
