#!/usr/bin/env python3
"""Test the API endpoint with improved fascia detector."""

import requests
import json

# Load test image as file
img_path = '/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png'

try:
    print("Testing /api/vision/analyze-fascia endpoint...")
    with open(img_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:5000/api/vision/analyze-fascia',
            files=files,
            timeout=30
        )
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ API Request successful!")
        print(f"  Status: {result.get('status')}")
        fascia = result.get('fascia', {})
        print(f"  Detected: {fascia.get('detected')}")
        print(f"  Confidence: {fascia.get('confidence')}")
        print(f"  Center: {fascia.get('center')}")
        print(f"  Boundary points: {len(fascia.get('boundary', []))}")
    elif response.status_code == 403:
        print(f"\n✗ Forbidden (403) - Check authentication")
    else:
        print(f"\n✗ API Error (status {response.status_code}):")
        print(response.text)
        
except ConnectionError:
    print("\n✗ Cannot connect to API. Is the backend running?")
    print("  Run: python3 backend/app.py")
except FileNotFoundError:
    print(f"\n✗ Image file not found: {img_path}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
