#!/usr/bin/env python3
"""
Debug fascia detection on the ACTUAL ultrasound image provided.
Will show exactly what the detector finds vs what should be found.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, './backend')

from vision.segmentation.edge_fascia_detector import EdgeFasciaDetector


def save_image_from_attachment():
    """
    The image was provided as attachment. 
    We need to recreate it or find where it is.
    For now, let's check if there's an image in common locations.
    """
    possible_paths = [
        '/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/ultrasound.png',
        '/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/test_ultrasound.png',
        '/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png',
        '/Users/HariKrishnaD/Downloads/ultrasound.png',
        '/Users/HariKrishnaD/Downloads/test.png',
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"✓ Found image at: {path}")
            return cv2.imread(path)
    
    print("⚠️  Image not found in common locations")
    return None


def analyze_image(image):
    """Analyze the actual ultrasound image with detailed debugging."""
    
    if image is None:
        print("❌ No image provided")
        return
    
    h, w = image.shape[:2]
    print(f"\n{'='*70}")
    print(f"ANALYZING REAL ULTRASOUND IMAGE")
    print(f"{'='*70}")
    print(f"\nImage dimensions: {w} x {h}")
    
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    print(f"Intensity range: [{gray.min()}, {gray.max()}]")
    print(f"Mean intensity: {gray.mean():.1f}")
    
    # Initialize detector
    detector = EdgeFasciaDetector()
    
    # Step 1: Enhance
    enhanced = detector.enhance_image(image)
    print(f"\n1. Enhanced image:")
    print(f"   Range: [{enhanced.min()}, {enhanced.max()}]")
    print(f"   Mean: {enhanced.mean():.1f}")
    
    # Step 2: Find band
    y_start, y_end = detector.find_fascia_band_region(enhanced)
    print(f"\n2. Band region found:")
    print(f"   y=[{y_start}, {y_end}] (height={y_end-y_start})")
    
    # Analyze brightness profile in band
    region = enhanced[y_start:y_end+1, :]
    brightness_per_row = np.mean(region, axis=1)
    
    print(f"\n3. Brightness profile in band region:")
    print(f"   Min: {brightness_per_row.min():.1f}")
    print(f"   Max: {brightness_per_row.max():.1f}")
    print(f"   Mean: {brightness_per_row.mean():.1f}")
    
    # Find peaks
    from scipy.ndimage import gaussian_filter1d
    brightness_smooth = gaussian_filter1d(brightness_per_row, sigma=1)
    gradient = np.diff(brightness_smooth)
    sign_change = np.diff(np.sign(gradient))
    peaks = np.where(sign_change < 0)[0]
    
    print(f"\n4. Peaks found in brightness profile:")
    print(f"   Total peaks: {len(peaks)}")
    
    if len(peaks) > 0:
        peak_brightness = brightness_per_row[peaks]
        for i, (peak_idx, brightness) in enumerate(zip(peaks[:5], peak_brightness[:5])):
            y_global = y_start + peak_idx
            print(f"   Peak {i+1}: y={y_global}, brightness={brightness:.1f}")
    
    # Step 3: Find boundaries using NEW pair-based logic
    pair = detector.find_fascia_pair(enhanced, y_start, y_end)
    
    if pair is not None:
        upper, lower = pair
    else:
        upper = None
        lower = None
    
    print(f"\n5. Boundary detection:")
    if upper is not None:
        print(f"   Upper: {len(upper)} points, y=[{upper[:, 1].min()}, {upper[:, 1].max()}], mean={upper[:, 1].mean():.1f}")
    else:
        print(f"   Upper: NOT FOUND")
    
    if lower is not None:
        print(f"   Lower: {len(lower)} points, y=[{lower[:, 1].min()}, {lower[:, 1].max()}], mean={lower[:, 1].mean():.1f}")
    else:
        print(f"   Lower: NOT FOUND")
    
    # Step 4: Spacing
    if upper is not None and lower is not None:
        spacing = lower[:, 1].astype(float) - upper[:, 1].astype(float)
        print(f"\n6. Spacing between boundaries:")
        print(f"   Mean: {spacing.mean():.1f}"  )
        print(f"   Std: {spacing.std():.1f}")
        print(f"   Range: [{spacing.min():.1f}, {spacing.max():.1f}]")
        
        # Validate
        confidence = detector.validate_boundaries(enhanced, upper, lower)
        print(f"\n7. Validation:")
        print(f"   Confidence: {confidence:.3f}")
    
    # Full detection
    print(f"\n8. Full detection result:")
    result = detector.detect(image)
    print(f"   Status: {result['status']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    # Create visualization
    print(f"\n9. Creating visualization...")
    
    # Draw detected boundaries on image
    vis = image.copy()
    
    if result['upper_edge'] is not None:
        upper = result['upper_edge']
        cv2.polylines(vis, [upper], False, (0, 255, 255), 2)  # Cyan upper
        print(f"   ✓ Drew upper boundary")
    
    if result['lower_edge'] is not None:
        lower = result['lower_edge']
        cv2.polylines(vis, [lower], False, (255, 0, 0), 2)  # Blue lower
        print(f"   ✓ Drew lower boundary")
    
    # Save visualization
    output_path = 'fascia_detection_result.png'
    cv2.imwrite(output_path, vis)
    print(f"   ✓ Saved to: {output_path}")
    
    return result


if __name__ == '__main__':
    # Try to load image
    print("Looking for ultrasound image...")
    image = save_image_from_attachment()
    
    if image is None:
        print("\n⚠️  Creating test image from scratch (since actual image not found)")
        print("   Please provide the actual ultrasound image file path")
        print("\n   Usage: python3 debug_real_ultrasound.py <image_path>")
        
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            print(f"\n   Attempting to load: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"   ❌ Failed to load: {image_path}")
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Analyze
    analyze_image(image)
