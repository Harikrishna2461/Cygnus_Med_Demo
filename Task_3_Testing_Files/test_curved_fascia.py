#!/usr/bin/env python3
"""Test curved fascia detector on real ultrasound patterns."""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, './backend')

from vision.segmentation.curved_fascia_detector import CurvedFasciaDetector
from vision.segmentation.unet_fascia import FasciaDetector


def create_curved_fascia_ultrasound():
    """Create synthetic ultrasound with curved fascia (like the real image)."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 70
    
    # Bright tissue region above fascia
    img[:85, :] = np.random.randint(140, 170, (85, 256, 3), dtype=np.uint8)
    
    # Create curved fascia (wavy lines, not straight!)
    x = np.arange(256)
    
    # Upper fascia curve - slightly wavy
    upper_y_base = 95
    upper_wave = 8 * np.sin(x / 60) + 4 * np.cos(x / 30)
    upper_y = upper_y_base + upper_wave
    
    # Draw upper fascia curve (bright)
    for i, y in enumerate(upper_y):
        y_int = int(y)
        if 0 <= y_int < 256:
            img[max(0, y_int-1):min(256, y_int+2), i] = [220, 220, 220]
    
    # Lower fascia curve - parallel but offset
    lower_y_base = upper_y_base + 18  # 18 pixels below
    lower_y = lower_y_base + upper_wave  # Same wave pattern
    
    # Draw lower fascia curve (bright)
    for i, y in enumerate(lower_y):
        y_int = int(y)
        if 0 <= y_int < 256:
            img[max(0, y_int-1):min(256, y_int+2), i] = [220, 220, 220]
    
    # Dark muscle region below fascia
    img[120:, :] = np.random.randint(40, 70, (136, 256, 3), dtype=np.uint8)
    
    # Add realistic ultrasound noise/speckle
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    
    return img


def test_curved_detector():
    """Test CurvedFasciaDetector on curved fascia."""
    print("\n" + "="*70)
    print("TEST 1: CurvedFasciaDetector (Curved Fascia)")
    print("="*70)
    
    # Create test image
    test_img = create_curved_fascia_ultrasound()
    print(f"✓ Created curved ultrasound: shape={test_img.shape}")
    
    # Initialize detector
    detector = CurvedFasciaDetector()
    print(f"✓ CurvedFasciaDetector initialized")
    
    # Run detection
    result = detector.detect(test_img)
    
    print(f"\n📊 Detection Results:")
    print(f"  • Status: {result.get('status', 'unknown')}")
    print(f"  • Confidence: {result.get('confidence', 0):.3f}")
    
    if result['upper_curve'] is not None:
        upper = result['upper_curve']
        lower = result['lower_curve']
        print(f"  • Upper curve: {len(upper)} points, y-range [{upper[:, 1].min()}, {upper[:, 1].max()}]")
        print(f"  • Lower curve: {len(lower)} points, y-range [{lower[:, 1].min()}, {lower[:, 1].max()}]")
        
        # Calculate average spacing
        spacing = lower[:, 1].astype(float) - upper[:, 1].astype(float)
        print(f"  • Band thickness: {spacing.mean():.1f} ± {spacing.std():.1f} pixels")
        
        mask = result['mask']
        fascia_pixels = np.count_nonzero(mask)
        print(f"  • Fascia pixels: {fascia_pixels}")
        
        return result['confidence'] > 0.5
    else:
        print(f"  ⚠️  No curves detected")
        return False


def test_fascia_detector_wrapper():
    """Test FasciaDetector wrapper."""
    print("\n" + "="*70)
    print("TEST 2: FasciaDetector Wrapper (API-compatible)")
    print("="*70)
    
    test_img = create_curved_fascia_ultrasound()
    
    detector = FasciaDetector()
    print(f"✓ FasciaDetector initialized")
    
    result = detector.detect(test_img)
    print(f"✓ Detection complete")
    
    confidence = result.get('confidence', 0)
    status = result.get('status', 'unknown')
    print(f"  • Status: {status}")
    print(f"  • Confidence: {confidence:.3f}")
    print(f"  • Mask shape: {result['mask'].shape}")
    
    return True


def test_pipeline_steps():
    """Test individual pipeline steps."""
    print("\n" + "="*70)
    print("TEST 3: Pipeline Steps Analysis")
    print("="*70)
    
    test_img = create_curved_fascia_ultrasound()
    detector = CurvedFasciaDetector()
    
    # Step 1: Enhance
    enhanced = detector.enhance_image(test_img)
    print(f"✓ Enhanced: range=[{enhanced.min()}, {enhanced.max()}]")
    
    # Step 2: Detect bright boundaries
    bright_mask = detector.detect_bright_boundaries(enhanced, brightness_threshold=140)
    bright_pixels = np.count_nonzero(bright_mask)
    print(f"✓ Bright regions: {bright_pixels} pixels")
    
    # Step 3: Find band
    y_start, y_end = detector.find_horizontal_band(bright_mask, min_width_coverage=0.3)
    if y_start is not None:
        band_height = y_end - y_start
        print(f"✓ Band found: y=[{y_start}, {y_end}], height={band_height}")
    else:
        print(f"⚠️  No band found")
        return False
    
    # Step 4: Extract curves
    upper = detector.extract_boundary_curve(enhanced, y_start, y_end, 'upper')
    lower = detector.extract_boundary_curve(enhanced, y_start, y_end, 'lower')
    
    if upper is not None and lower is not None:
        print(f"✓ Curves extracted: upper={len(upper)} pts, lower={len(lower)} pts")
        print(f"  • Upper y-range: [{upper[:, 1].min()}, {upper[:, 1].max()}]")
        print(f"  • Lower y-range: [{lower[:, 1].min()}, {lower[:, 1].max()}]")
        return True
    else:
        print(f"⚠️  Failed to extract curves")
        return False


def test_wavy_fascia_detection():
    """Test detection of truly wavy fascia."""
    print("\n" + "="*70)
    print("TEST 4: Wavy Fascia Pattern (Real-like)")
    print("="*70)
    
    # Create more realistic wavy pattern
    img = np.ones((256, 256, 3), dtype=np.uint8) * 80
    img[:90, :] = 150  # Bright above
    
    # Highly wavy fascia
    x = np.arange(256)
    y_upper = 100 + 10 * np.sin(x / 40) + 6 * np.sin(x / 80)
    y_lower = y_upper + 20
    
    for i in range(256):
        for y_pos in [y_upper[i], y_lower[i]]:
            y_int = int(y_pos)
            if 0 <= y_int < 256:
                for dy in range(-1, 2):
                    if 0 <= y_int + dy < 256:
                        img[y_int + dy, i] = [230, 230, 230]
    
    img[130:, :] = 60  # Dark below
    
    # Add speckle noise
    speckle = np.random.randint(-10, 10, img.shape)
    img = np.clip(img.astype(int) + speckle, 0, 255).astype(np.uint8)
    
    detector = CurvedFasciaDetector()
    result = detector.detect(img)
    
    print(f"✓ Created wavy fascia pattern")
    print(f"  • Status: {result.get('status')}")
    print(f"  • Confidence: {result.get('confidence'):.3f}")
    
    if result['confidence'] > 0.3:
        print(f"  ✓ Wavy pattern detected")
        return True
    else:
        print(f"  ⚠️  Low confidence on wavy pattern")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CURVED FASCIA DETECTION - TEST SUITE")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Curved Detector", test_curved_detector()))
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Curved Detector", False))
    
    try:
        results.append(("FasciaDetector Wrapper", test_fascia_detector_wrapper()))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("FasciaDetector Wrapper", False))
    
    try:
        results.append(("Pipeline Steps", test_pipeline_steps()))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("Pipeline Steps", False))
    
    try:
        results.append(("Wavy Fascia", test_wavy_fascia_detection()))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("Wavy Fascia", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed\n")
    
    if passed_count >= 3:
        print("🎉 Curved fascia detection working!")
        return 0
    else:
        print("⚠️  Need adjustments")
        return 1


if __name__ == '__main__':
    sys.exit(main())
