#!/usr/bin/env python3
"""Test the API endpoint with improved fascia detector."""

import requests
import cv2
import base64
import json

# Load test image
img = cv2.imread('/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/real_ultrasound.png')
if img is None:
    print("Image not found")
    exit(1)

# Encode image
_, img_encoded = cv2.imencode('.png', img)
img_base64 = base64.b64encode(img_encoded).decode('utf-8')

# Prepare request
headers = {'Content-Type': 'application/json'}
payload = {
    'image_base64': img_base64,
    'image_format': 'png'
}

# Try the API
try:
    print("Testing /api/vision/analyze-fascia endpoint...")
    response = requests.post(
        'http://localhost:5000/api/vision/analyze-fascia',
        json=payload,
        headers=headers,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ API Request successful!")
        print(f"  Status: {result.get('status')}")
        if result.get('fascia_detected'):
            print(f"  Upper y: {result.get('fascia_upper_y')}")
            print(f"  Lower y: {result.get('fascia_lower_y')}")
            print(f"  Confidence: {result.get('confidence')}")
    else:
        print(f"\n✗ API Error (status {response.status_code}):")
        print(response.text)
        
except ConnectionError:
    print("\n✗ Cannot connect to API. Is the backend running?")
    print("  Run: python3 backend/app.py")
except Exception as e:
    print(f"\n✗ Error: {e}")
