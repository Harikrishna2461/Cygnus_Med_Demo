#!/usr/bin/env python3
"""Debug curved fascia detection."""

import sys
import numpy as np
import cv2

sys.path.insert(0, './backend')

from vision.segmentation.curved_fascia_detector import CurvedFasciaDetector


def debug_detection():
    """Debug each step of detection."""
    
    # Create test image
    img = np.ones((256, 256, 3), dtype=np.uint8) * 70
    img[:85, :] = np.random.randint(140, 170, (85, 256, 3), dtype=np.uint8)
    
    # Create curved fascia
    x = np.arange(256)
    upper_y_base = 95
    upper_wave = 8 * np.sin(x / 60) + 4 * np.cos(x / 30)
    upper_y = upper_y_base + upper_wave
    
    for i, y in enumerate(upper_y):
        y_int = int(y)
        if 0 <= y_int < 256:
            img[max(0, y_int-1):min(256, y_int+2), i] = [220, 220, 220]
    
    lower_y_base = upper_y_base + 18
    lower_y = lower_y_base + upper_wave
    
    for i, y in enumerate(lower_y):
        y_int = int(y)
        if 0 <= y_int < 256:
            img[max(0, y_int-1):min(256, y_int+2), i] = [220, 220, 220]
    
    img[120:, :] = np.random.randint(40, 70, (136, 256, 3), dtype=np.uint8)
    
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    
    detector = CurvedFasciaDetector()
    
    # Step 1
    enhanced = detector.enhance_image(img)
    print(f"1. Enhanced: min={enhanced.min()}, max={enhanced.max()}, mean={enhanced.mean():.1f}")
    
    # Check intensities at specific locations
    print(f"\n   Intensity checks:")
    print(f"   y=50 (above fascia): mean={enhanced[50, :].mean():.1f}")
    print(f"   y=95 (upper fascia): mean={enhanced[95, :].mean():.1f}")
    print(f"   y=110 (lower fascia): mean={enhanced[110, :].mean():.1f}")
    print(f"   y=130 (below fascia): mean={enhanced[130, :].mean():.1f}")
    
    # Step 2
    print(f"\n2. Testing bright boundary detection...")
    for threshold in [100, 120, 140, 160]:
        _, bright = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)
        bright_pixels = np.count_nonzero(bright)
        print(f"   threshold={threshold}: {bright_pixels} pixels")
    
    bright_mask = detector.detect_bright_boundaries(enhanced, brightness_threshold=160)
    bright_pixels = np.count_nonzero(bright_mask)
    print(f"   Selected threshold=160: {bright_pixels} pixels")
    
    # Step 3
    y_start, y_end = detector.find_horizontal_band(bright_mask)
    print(f"\n3. Band found: y=[{y_start}, {y_end}]")
    
    if y_start is not None:
        band_projection = np.sum(bright_mask[y_start:y_end+1, :], axis=1)
        print(f"   Band projection (row sums):")
        print(f"   min={band_projection.min()}, max={band_projection.max()}, mean={band_projection.mean():.1f}")
    
    # Step 4
    print(f"\n4. Extracting curves...")
    upper = detector.extract_boundary_curve(enhanced, y_start, y_end, 'upper')
    lower = detector.extract_boundary_curve(enhanced, y_start, y_end, 'lower')
    
    if upper is not None:
        print(f"   Upper: {len(upper)} points, y=[{upper[:, 1].min()}, {upper[:, 1].max()}]")
        print(f"          mean y={upper[:, 1].mean():.1f}")
    
    if lower is not None:
        print(f"   Lower: {len(lower)} points, y=[{lower[:, 1].min()}, {lower[:, 1].max()}]")
        print(f"          mean y={lower[:, 1].mean():.1f}")
    
    # Step 5
    if upper is not None and lower is not None:
        print(f"\n5. Validating...")
        confidence = detector.validate_fascia_curves(enhanced, upper, lower)
        print(f"   Confidence: {confidence:.3f}")
        
        # Debug validation
        spacing = lower[:, 1].astype(float) - upper[:, 1].astype(float)
        print(f"   Spacing: mean={spacing.mean():.1f}, std={spacing.std():.1f}, range=[{spacing.min():.1f}, {spacing.max():.1f}]")
    
    # Full detection
    print(f"\n6. Full detection result:")
    result = detector.detect(img)
    print(f"   Status: {result.get('status')}")
    print(f"   Confidence: {result.get('confidence'):.3f}")


if __name__ == '__main__':
    debug_detection()
