#!/usr/bin/env python3
"""
Test the vision API with the real ultrasound image.
Make sure the backend is running first:
  cd backend && python3 app.py
"""

import requests
import cv2
import json
from pathlib import Path

API_URL = "http://127.0.0.1:5002/api/vision"
IMAGE_PATH = "/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png"

def test_api_with_image():
    """Test the vision API with real ultrasound image."""
    
    # Verify image exists
    if not Path(IMAGE_PATH).exists():
        print(f"❌ Image not found: {IMAGE_PATH}")
        return False
    
    # Read image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"❌ Failed to load image")
        return False
    
    _, buffer = cv2.imencode('.png', image)
    image_bytes = buffer.tobytes()
    
    print(f"✓ Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Test 1: analyze-fascia endpoint
    print("\n[1] Testing /api/vision/analyze-fascia...")
    try:
        response = requests.post(
            f"{API_URL}/analyze-fascia",
            files={'file': ('real_ultrasound.png', image_bytes, 'image/png')},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Status: {response.status_code}")
            print(f"  ✓ Confidence: {result.get('confidence', 'N/A')}")
            print(f"  ✓ Has mask: {'mask' in result}")
            if result.get('confidence', 0) > 0:
                print(f"  ✓ Fascia detected!")
            else:
                print(f"  ⚠️  No fascia (confidence=0)")
        else:
            print(f"  ❌ Error: {response.status_code}")
            print(f"  {response.text[:200]}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")
    
    # Test 2: analyze-frame endpoint
    print("\n[2] Testing /api/vision/analyze-frame...")
    try:
        response = requests.post(
            f"{API_URL}/analyze-frame",
            files={'file': ('real_ultrasound.png', image_bytes, 'image/png')},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Status: {response.status_code}")
            print(f"  ✓ Frame analysis complete")
        else:
            print(f"  ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")
    
    # Test 3: analyze-integrated-veins endpoint
    print("\n[3] Testing /api/vision/analyze-integrated-veins...")
    try:
        response = requests.post(
            f"{API_URL}/analyze-integrated-veins",
            files={'file': ('real_ultrasound.png', image_bytes, 'image/png')},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Status: {response.status_code}")
            print(f"  ✓ Fascia confidence: {result.get('fascia_confidence', 'N/A')}")
            veins = result.get('veins', [])
            print(f"  ✓ Veins detected: {len(veins)}")
            if len(veins) > 0:
                for i, vein in enumerate(veins[:3]):
                    pos = vein.get('position', {})
                    print(f"    Vein {i+1}: class={vein.get('vein_class')}, x={pos.get('x')}, y={pos.get('y')}")
        else:
            print(f"  ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")
    
    # Test 4: health check
    print("\n[4] Testing /api/vision/health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Status: healthy")
            print(f"  ✓ Fascia detector: {result.get('fascia_detector', 'unknown')}")
        else:
            print(f"  ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")
    
    print("\n" + "="*60)
    print("API Test Complete")


if __name__ == '__main__':
    print("="*60)
    print("Testing Vision API with Real Ultrasound Image")
    print("="*60)
    print("\n⚠️  IMPORTANT: Backend must be running first!")
    print("   In another terminal: cd backend && python3 app.py")
    print()
    
    test_api_with_image()
