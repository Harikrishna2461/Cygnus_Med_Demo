#!/usr/bin/env python3
"""Quick test of fallback detector flow"""

import numpy as np
import cv2
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Test SAMSegmenter with CPU
logger.info("=" * 60)
logger.info("TEST: SAMSegmenter with CPU (should use fallback)")
logger.info("=" * 60)

from vision.segmentation.sam_wrapper import SAMSegmenter

segmenter = SAMSegmenter(model_type="vit_b", device="cpu")
logger.info(f"Created segmenter: use_fallback={segmenter.use_fallback}, has_detector={segmenter.fallback_detector is not None}")

# Create a dummy image
dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
logger.info(f"Created dummy image: {dummy_image.shape}")

# Test segment_fascia first
logger.info("\nCalling segment_fascia...")
try:
    fascia_mask = segmenter.segment_fascia(dummy_image)
    logger.info(f"✓ segment_fascia returned mask: shape={fascia_mask.shape}, unique values={np.unique(fascia_mask)}")
except Exception as e:
    logger.error(f"✗ segment_fascia failed: {e}", exc_info=True)
    fascia_mask = None

# Test segment_veins - this should trigger the fallback and use fascia_mask
logger.info("\nCalling segment_veins...")
try:
    results = segmenter.segment_veins(dummy_image, fascia_mask=fascia_mask, num_masks=3)
    logger.info(f"✓ segment_veins returned {len(results)} masks")
    for i, result in enumerate(results):
        logger.info(f"  Mask {i}: id={result.get('vein_id')}, confidence={result.get('confidence'):.2f}")
except Exception as e:
    logger.error(f"✗ segment_veins failed: {e}", exc_info=True)

logger.info("\n" + "=" * 60)
logger.info("TEST COMPLETE")
logger.info("=" * 60)
