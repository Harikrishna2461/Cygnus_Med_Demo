#!/usr/bin/env python3
"""Test vision API with a sample image"""

import requests
import numpy as np
import cv2
import json
import base64
import time

# Wait for backend to be ready
print("Waiting for backend to be ready...")
for i in range(10):
    try:
        response = requests.get("http://localhost:5002/api/info", timeout=2)
        if response.status_code == 200:
            print("✅ Backend is running")
            break
    except:
        print(f"  Attempt {i+1}/10... waiting...")
        time.sleep(1)
else:
    print("❌ Backend not responding")
    exit(1)

# Create a synthetic ultrasound-like image
print("\nCreating synthetic ultrasound image...")
image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
# Add some patterns to look more like ultrasound
for i in range(10):
    y = np.random.randint(100, 400)
    x = np.random.randint(100, 600)
    cv2.circle(image, (x, y), np.random.randint(20, 80), (np.random.randint(100, 150), 50, 50), -1)

# Encode as JPEG
success, buffer = cv2.imencode('.jpg', image)
image_base64 = base64.b64encode(buffer).decode('utf-8')

print(f"  Image size: {image.shape}")
print(f"  Base64 length: {len(image_base64)}")

# Test API
print("\nTesting /api/vision/analyze-frame endpoint...")
try:
    response = requests.post(
        "http://localhost:5002/api/vision/analyze-frame",
        json={"image": image_base64, "filename": "test_ultrasound.jpg"},
        timeout=30
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ API response received")
        
        # Display results structure
        if "veins" in data:
            print(f"  ✓ Veins detected: {len(data['veins'])}")
            if data['veins']:
                v = data['veins'][0]
                print(f"    - Vein 0: type={v.get('vein_type')}, confidence={v.get('confidence', 'N/A')}")
        
        if "fascia" in data:
            fascia_data = data.get('fascia', {})
            if isinstance(fascia_data, dict) and 'detected' in fascia_data:
                print(f"  ✓ Fascia detected: {fascia_data['detected']}")
        
        # Print raw JSON response
        print("\nFull response (first 500 chars):")
        print(json.dumps(data, indent=2)[:500] + "...")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text[:500])
        
except requests.exceptions.Timeout:
    print("❌ Request timed out (30s)")
except requests.exceptions.ConnectionError:
    print("❌ Connection refused")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("Now refresh http://localhost:5002 and try uploading an image")
print("=" * 60)
