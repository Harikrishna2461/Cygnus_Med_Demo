#!/usr/bin/env python3
"""
Test script for Hough Transform-based fascia detection.
Tests the complete pipeline: denoise → edges → Hough → cluster → validate.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

from vision.segmentation.hough_fascia_detector import HoughFasciaDetector
from vision.segmentation.unet_fascia import FasciaDetector


def create_synthetic_ultrasound():
    """Create realistic synthetic ultrasound test image."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 90
    
    # Bright region (fat/tissue) - top 80 pixels
    img[:85, :] = np.random.randint(170, 210, (85, 256, 3), dtype=np.uint8)
    
    # Fascia lines - bright horizontal band (9 pixels apart)
    # Upper fascia line
    img[95:98, 30:220] = [220, 220, 220]  # Upper line (3px thick)
    # Lower fascia line  
    img[104:107, 30:220] = [220, 220, 220]  # Lower line (9px apart, 3px thick)
    
    # Add some intensity variation to make it realistic
    for i in range(30, 220):
        intensity = 220 + np.random.randint(-5, 5)
        img[96, i] = intensity
        img[105, i] = intensity
    
    # Darker region (muscle) - below fascia
    img[110:, :] = np.random.randint(40, 80, (146, 256, 3), dtype=np.uint8)
    
    # Add noise
    noise = np.random.normal(0, 3, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    
    return img


def test_hough_detector():
    """Test HoughFasciaDetector directly."""
    print("\n" + "="*70)
    print("TEST 1: HoughFasciaDetector (Hough Transform Pipeline)")
    print("="*70)
    
    # Create test image
    test_img = create_synthetic_ultrasound()
    print(f"✓ Created synthetic ultrasound: shape={test_img.shape}, dtype={test_img.dtype}")
    
    # Initialize detector
    detector = HoughFasciaDetector()
    print("✓ HoughFasciaDetector initialized")
    
    # Run detection
    result = detector.detect(test_img)
    
    # Display results
    print(f"\n📊 Detection Results:")
    print(f"  • Confidence: {result['confidence']:.3f}")
    print(f"  • Lines detected: {result.get('num_lines_detected', 'N/A')}")
    print(f"  • Clusters found: {result.get('num_clusters', 'N/A')}")
    print(f"  • Mask shape: {result['mask'].shape}")
    print(f"  • Mask range: [{result['mask'].min()}, {result['mask'].max()}]")
    
    if result['upper_line'] is not None:
        y_u, x_start_u, x_end_u = result['upper_line']
        y_l, x_start_l, x_end_l = result['lower_line']
        print(f"\n📍 Fascia Boundaries:")
        print(f"  • Upper line: y={y_u:.1f}, x=[{x_start_u:.0f}, {x_end_u:.0f}]")
        print(f"  • Lower line: y={y_l:.1f}, x=[{x_start_l:.0f}, {x_end_l:.0f}]")
        print(f"  • Band thickness: {y_l - y_u:.1f} pixels")
        
        # Check coverage
        x_union_len = min(x_end_u, x_end_l) - max(x_start_u, x_start_l)
        coverage = x_union_len / test_img.shape[1]
        print(f"  • Width coverage: {coverage:.1%}")
    else:
        print(f"\n⚠️  No fascia band detected")
    
    return result['confidence'] > 0.3


def test_fascia_detector_wrapper():
    """Test FasciaDetector wrapper class."""
    print("\n" + "="*70)
    print("TEST 2: FasciaDetector Wrapper (API-compatible)")
    print("="*70)
    
    # Create test image
    test_img = create_synthetic_ultrasound()
    
    # Initialize wrapper
    detector = FasciaDetector()
    print("✓ FasciaDetector wrapper initialized")
    
    # Run detection
    result = detector.detect(test_img)
    print(f"✓ Detection complete: type={type(result)}")
    
    if isinstance(result, dict):
        print(f"  • Result keys: {list(result.keys())}")
        print(f"  • Confidence: {result.get('confidence', 'N/A')}")
        mask = result.get('mask')
        if mask is not None:
            print(f"  • Mask shape: {mask.shape}, dtype: {mask.dtype}")
    else:
        print(f"  • Result type: {type(result)}")
    
    return True


def test_edge_detection_pipeline():
    """Test individual pipeline steps."""
    print("\n" + "="*70)
    print("TEST 3: Pipeline Steps Analysis")
    print("="*70)
    
    # Create test image
    test_img = create_synthetic_ultrasound()
    detector = HoughFasciaDetector()
    
    # Step 1: Denoise
    denoised = detector.denoise(test_img)
    print(f"✓ Denoise: shape={denoised.shape}, range=[{denoised.min()}, {denoised.max()}]")
    
    # Step 2: Detect edges
    edges = detector.detect_horizontal_edges(denoised)
    edge_pixels = np.count_nonzero(edges)
    print(f"✓ Edge detection: {edge_pixels} pixels detected (max={edges.max()})")
    
    # Step 3: Hough lines
    lines = detector.detect_lines_hough(edges, threshold=40, min_length=80, max_gap=15)
    print(f"✓ Hough Transform: {len(lines)} lines detected")
    
    if len(lines) > 0:
        print(f"  • First line: {lines[0]}")
        for i, line in enumerate(lines[:3]):
            x1, y1, x2, y2 = line
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            print(f"    Line {i+1}: length={length:.1f}px, angle={angle:.1f}°")
    
    # Step 4: Clustering
    clusters = detector.cluster_parallel_lines(lines, y_tolerance=8)
    print(f"✓ Clustering: {len(clusters)} clusters")
    for i, cluster in enumerate(clusters[:3]):
        print(f"  • Cluster {i+1}: {len(cluster)} lines")
    
    return len(clusters) >= 2


def test_intensity_validation():
    """Test intensity-based validation."""
    print("\n" + "="*70)
    print("TEST 4: Intensity Validation")
    print("="*70)
    
    # Create test image with known intensities
    test_img = np.ones((256, 256, 3), dtype=np.uint8) * 100
    
    # Bright region (above)
    test_img[:80, :] = 190
    
    # Fascia
    test_img[90:92, 30:200] = 230
    test_img[102:104, 30:200] = 230
    
    # Dark region (below)
    test_img[110:, :] = 60
    
    detector = HoughFasciaDetector()
    result = detector.detect(test_img)
    
    print(f"Intensity structure:")
    print(f"  • Above fascia: ~190 (bright)")
    print(f"  • Fascia: ~230 (brightest)")
    print(f"  • Below fascia: ~60 (dark)")
    print(f"\n✓ Validation result: confidence={result['confidence']:.3f}")
    
    if result['confidence'] > 0.5:
        print("✓ Correctly identified bright-to-dark transition")
        return True
    else:
        print("⚠️  Validation confidence low")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HOUGH TRANSFORM FASCIA DETECTION - TEST SUITE")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Hough Detector", test_hough_detector()))
    except Exception as e:
        print(f"❌ Hough Detector test failed: {e}")
        results.append(("Hough Detector", False))
    
    try:
        results.append(("FasciaDetector Wrapper", test_fascia_detector_wrapper()))
    except Exception as e:
        print(f"❌ FasciaDetector Wrapper test failed: {e}")
        results.append(("FasciaDetector Wrapper", False))
    
    try:
        results.append(("Pipeline Steps", test_edge_detection_pipeline()))
    except Exception as e:
        print(f"❌ Pipeline Steps test failed: {e}")
        results.append(("Pipeline Steps", False))
    
    try:
        results.append(("Intensity Validation", test_intensity_validation()))
    except Exception as e:
        print(f"❌ Intensity Validation test failed: {e}")
        results.append(("Intensity Validation", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
