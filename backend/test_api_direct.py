#!/usr/bin/env python3
"""Direct API test to see what visualization data is returned"""

import requests
import cv2
import numpy as np
import base64
import json

# Create a synthetic ultrasound-like image
image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
# Add some patterns
for i in range(10):
    y = np.random.randint(100, 400)
    x = np.random.randint(100, 600)
    cv2.circle(image, (x, y), np.random.randint(20, 80), (np.random.randint(100, 150), 50, 50), -1)

# Encode as JPEG
success, buffer = cv2.imencode('.jpg', image)
image_bytes = buffer.tobytes()

print("Testing API endpoint: /api/vision/analyze-frame")
print(f"Image size: {len(image_bytes)} bytes")

try:
    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    data = {'enable_llm': 'true'}
    
    response = requests.post(
        'http://127.0.0.1:5002/api/vision/analyze-frame',
        files=files,
        data=data,
        timeout=120
    )
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ API Response received")
        print(f"Response keys: {result.keys()}")
        print(f"Status: {result.get('status')}")
        print(f"Veins detected: {len(result.get('veins', []))}")
        
        # Check visualization
        viz = result.get('visualization', {})
        print(f"\nVisualization section:")
        print(f"  - format: {viz.get('format')}")
        print(f"  - data type: {type(viz.get('data'))}")
        print(f"  - data length: {len(viz.get('data', '')) if viz.get('data') else 'NULL'}")
        
        if viz.get('data'):
            print(f"  - data (first 50 chars): {str(viz.get('data'))[:50]}...")
            print(f"  ✅ Visualization data IS being returned!")
        else:
            print(f"  ❌ Visualization data is NULL or empty!")
        
        # Check veins structure
        if result.get('veins'):
            vein = result['veins'][0]
            print(f"\nFirst vein structure:")
            for key in vein.keys():
                print(f"  - {key}: {type(vein[key])}")
    else:
        print(f"❌ Error {response.status_code}: {response.text[:200]}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
