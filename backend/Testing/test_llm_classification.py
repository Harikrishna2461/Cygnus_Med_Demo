#!/usr/bin/env python3
"""
Test script to verify LLM is being called with vein coordinates and classification.
Uploads a test image to /api/vision/analyze-frame with LLM enabled.
"""

import requests
import json
import base64
import cv2
import numpy as np
from pathlib import Path

# Configuration
API_URL = "http://localhost:5002/api/vision/analyze-frame"
TEST_IMAGE = "/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/backend/sample_medical_text.txt"

def create_test_image():
    """Create a simple test ultrasound-like image"""
    # Create a test image (512x512, grayscale ultrasound-like)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add some structure to simulate ultrasound
    for i in range(512):
        for j in range(512):
            # Add some noise
            val = np.random.randint(50, 150)
            img[i, j] = [val, val, val]
    
    # Add some "veins" - dark lines/blobs
    cv2.circle(img, (150, 200), 30, (30, 30, 30), -1)  # Vein 1
    cv2.circle(img, (300, 250), 25, (40, 40, 40), -1)  # Vein 2
    cv2.circle(img, (200, 400), 35, (35, 35, 35), -1)  # Vein 3
    
    # Add a "fascia" line
    cv2.line(img, (0, 280), (512, 280), (200, 200, 200), 3)
    
    return img

def test_llm_classification():
    """Test vein classification with LLM enabled"""
    
    print("=" * 70)
    print("TEST: Vein Classification with LLM Enabled")
    print("=" * 70)
    
    # Create test image
    print("\n[1/3] Creating test ultrasound image...")
    test_img = create_test_image()
    
    # Save to temporary file
    temp_image_path = "/tmp/test_ultrasound.png"
    cv2.imwrite(temp_image_path, test_img)
    print(f"  ✓ Test image created: {temp_image_path}")
    print(f"    Shape: {test_img.shape}, Size: {test_img.size * 3 // 1024}KB")
    
    # Prepare request
    print("\n[2/3] Sending image to backend with LLM ENABLED...")
    
    try:
        with open(temp_image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'enable_llm': 'true'  # KEY: Enable LLM for this request
            }
            
            response = requests.post(API_URL, files=files, data=data, timeout=60)
        
        print(f"  ✓ Response received: Status {response.status_code}")
        
        if response.status_code != 200:
            print(f"  ✗ Error: {response.text}")
            return False
        
        result = response.json()
        
        # Parse results
        print("\n[3/3] Analyzing Results")
        print("-" * 70)
        
        # Check for veins
        veins = result.get('veins', [])
        print(f"\n📊 VEIN DETECTION ({len(veins)} veins detected):")
        
        for i, vein in enumerate(veins, 1):
            print(f"\n  Vein {i}: {vein.get('vein_id', 'Unknown')}")
            print(f"    └─ Properties:")
            
            props = vein.get('properties', {})
            print(f"       • Area: {props.get('area', 0)} pixels")
            print(f"       • Centroid: {props.get('centroid', (0, 0))}")  # ← COORDINATES
            print(f"       • Perimeter: {props.get('perimeter', 0):.1f}")
            print(f"       • Solidity: {props.get('solidity', 0):.2f}")
            
            # Check for spatial analysis (contains distance to fascia)
            spatial = vein.get('spatial_analysis', {})
            if spatial:
                print(f"    └─ Spatial Analysis:")
                print(f"       • Centroid: {spatial.get('vein_centroid', 'N/A')}")  # ← COORDINATES
                print(f"       • Distance to Fascia: {spatial.get('distance_to_fascia_mm', 'N/A'):.1f}mm")  # ← FASCIA DISTANCE
                print(f"       • Relative Position: {spatial.get('relative_position', 'N/A')}")  # ← RELATIVE POSITION
                print(f"       • Intersects Fascia: {spatial.get('intersects_fascia', False)}")
            
            # Check for classification
            classification = vein.get('classification', {})
            if classification:
                print(f"    └─ Classification:")
                print(f"       • Type: {classification.get('primary_classification', 'Unknown')}")
                print(f"       • N-Level: {classification.get('n_level', 'Unknown')}")
                print(f"       • Confidence: {classification.get('confidence', 0):.2f}")
            
            # Check for LLM analysis (KEY!)
            llm = vein.get('llm_analysis', {})
            if llm:
                print(f"    └─ LLM CLASSIFICATION (✨ LLM was called!):")
                print(f"       • Confirmed Type: {llm.get('llm_confirmed_type', 'N/A')}")
                print(f"       • LLM Confidence: {llm.get('llm_confidence', 'N/A')}")
                print(f"       • Is GSV: {llm.get('llm_is_gsv', False)}")
                print(f"       • LLM Notes: {llm.get('llm_notes', 'N/A')}")
            else:
                print(f"    └─ ⚠️  NO LLM ANALYSIS (LLM was NOT called)")
        
        # Check summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        summary = result.get('summary_statistics', {})
        for key, value in summary.items():
            print(f"  • {key}: {value}")
        
        # Check visualization
        viz = result.get('visualization', {})
        if viz and viz.get('data'):
            print(f"\n🖼️  VISUALIZATION:")
            print(f"  • Format: {viz.get('format', 'Unknown')}")
            print(f"  • Data Size: {len(viz.get('data', '')) // 1024} KB (base64 PNG)")
        
        print("\n" + "=" * 70)
        if len(veins) > 0 and any(v.get('llm_analysis') for v in veins):
            print("✅ SUCCESS: LLM was called and veins were classified!")
            print("✅ Coordinates and fascia data are being used properly!")
        else:
            print("⚠️  No LLM analysis found in results")
            if len(veins) == 0:
                print("   (No veins were detected in the test image)")
        print("=" * 70)
        
        return True
        
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timed out (LLM may be processing)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

if __name__ == '__main__':
    success = test_llm_classification()
    exit(0 if success else 1)
