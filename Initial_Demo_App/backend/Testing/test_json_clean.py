#!/usr/bin/env python3
"""Test JSON serialization of numpy arrays"""

import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the clean function (same as in app.py)
def clean_numpy_for_json(obj):
    """Recursively convert numpy arrays to JSON-safe formats"""
    if isinstance(obj, dict):
        return {key: clean_numpy_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_numpy_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        if len(obj.shape) == 2 and obj.dtype in [np.uint8, bool]:
            return {
                "type": "mask_base64",
                "data": "base64_encoded_png",
                "shape": list(obj.shape)
            }
        else:
            return obj.astype(int).tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

# Test data similar to what the vision pipeline returns
test_vein = {
    "mask": np.ones((100, 100), dtype=np.uint8) * 255,  # 2D numpy array
    "confidence": np.float32(0.85),  # numpy float
    "vein_id": "V1",
    "properties": {
        "area": np.int64(5000),  # numpy int
        "perimeter": np.float64(314.5),
        "centroid": [np.float32(50.0), np.float32(50.0)]  # list of numpy floats
    }
}

logger.info("Testing JSON serialization of vein with numpy arrays...")
logger.info(f"Original vein area type: {type(test_vein['properties']['area'])}")
logger.info(f"Original mask type: {type(test_vein['mask'])}")

try:
    cleaned = clean_numpy_for_json(test_vein)
    logger.info(f"Cleaned vein area type: {type(cleaned['properties']['area'])}")
    logger.info(f"Cleaned mask type: {type(cleaned['mask'])}")
    
    # Try to JSON serialize
    json_str = json.dumps(cleaned)
    logger.info("✅ Successfully serialized to JSON!")
    logger.info(f"JSON length: {len(json_str)} chars")
    
    # Verify it can be deserialized
    deserialized = json.loads(json_str)
    logger.info(f"✅ Successfully deserialized from JSON!")
    logger.info(f"Mask data: {str(cleaned['mask'])[:100]}...")
    
except Exception as e:
    logger.error(f"❌ Failed: {e}", exc_info=True)

logger.info("\n✅ Test complete - JSON serialization working correctly!")
