#!/usr/bin/env python3
"""
Integration Test: Vein Detection System with Echo VLM + Qdrant RAG
Tests the complete 4-stage pipeline with proper RAG context injection
"""

import sys
import os
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_synthetic_ultrasound_image(width=512, height=512):
    """Create a synthetic ultrasound-like image for testing"""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add grayscale ultrasound texture
    texture = np.random.randint(30, 100, (height, width), dtype=np.uint8)
    image[:, :] = texture[:, :, np.newaxis]

    # Draw fascia line (horizontal line in middle)
    fascia_y = height // 2
    cv2.line(image, (0, fascia_y), (width, fascia_y), (100, 150, 100), 3)

    # Draw some vein-like structures
    veins = [
        {'x': 150, 'y': 200, 'radius': 20},  # Above fascia (N3)
        {'x': 256, 'y': 256, 'radius': 25},  # At fascia (N2)
        {'x': 350, 'y': 320, 'radius': 18},  # Below fascia (N1)
    ]

    for vein in veins:
        cv2.circle(image, (vein['x'], vein['y']), vein['radius'], (50, 80, 200), 2)

    return image, fascia_y, veins


def test_retrieve_context_function():
    """Test that retrieve_context function works"""
    logger.info("=" * 60)
    logger.info("TEST 1: Verify Qdrant RAG retrieval function")
    logger.info("=" * 60)

    try:
        from app import retrieve_context, qdrant_client, load_qdrant_client

        # Ensure Qdrant is initialized
        if qdrant_client is None:
            logger.info("Initializing Qdrant client...")
            load_qdrant_client()

        # Test retrieval
        test_query = "vein depth classification at fascia N2 CHIVA perforator"
        logger.info(f"Querying: {test_query}")

        results = retrieve_context(test_query, k=2)

        if results:
            logger.info(f"✅ Retrieved {len(results)} context chunks from Qdrant")
            for i, chunk in enumerate(results, 1):
                preview = chunk[:100] if len(chunk) > 100 else chunk
                logger.info(f"   Chunk {i}: {preview}...")
            return True
        else:
            logger.warning("⚠ No results from Qdrant (may not be initialized)")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to test retrieve_context: {e}")
        return False


def test_echo_vlm_initialization():
    """Test that Echo VLM can be initialized"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Verify Echo VLM initialization")
    logger.info("=" * 60)

    try:
        from echo_vlm_integration import EchoVLMIntegration

        # Try initializing WITHOUT RAG first
        logger.info("Initializing Echo VLM (without RAG)...")
        vlm = EchoVLMIntegration(
            model_id="chaoyinshe/EchoVLM",
            device_map="auto",
            retrieve_context_fn=None
        )

        if vlm._initialized:
            logger.info("✅ Echo VLM initialized successfully")
            logger.info(f"   Model: {vlm.model_id}")
            logger.info(f"   Device: {vlm.model.device if vlm.model else 'unknown'}")
            return True
        else:
            logger.warning("⚠ Echo VLM not initialized (may not have model weights)")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to initialize Echo VLM: {e}")
        return False


def test_echo_vlm_with_rag():
    """Test Echo VLM with RAG context injection"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Verify Echo VLM + RAG Integration")
    logger.info("=" * 60)

    try:
        from echo_vlm_integration import EchoVLMIntegration
        from app import retrieve_context, load_qdrant_client, qdrant_client

        # Ensure Qdrant is ready
        if qdrant_client is None:
            load_qdrant_client()

        logger.info("Initializing Echo VLM WITH RAG context function...")
        vlm = EchoVLMIntegration(
            model_id="chaoyinshe/EchoVLM",
            device_map="auto",
            retrieve_context_fn=retrieve_context  # ← RAG injection
        )

        if vlm._initialized and vlm.retrieve_rag_context:
            logger.info("✅ Echo VLM initialized with RAG context function")
            logger.info(f"   VLM has retrieve_rag_context: {vlm.retrieve_rag_context is not None}")

            # Test RAG context retrieval within VLM
            test_query = "vein classification N2 fascia interface"
            context = vlm.retrieve_rag_context(test_query, k=2)
            if context:
                logger.info(f"✅ VLM successfully retrieved {len(context)} RAG chunks")
                return True
            else:
                logger.warning("⚠ RAG retrieval returned no results")
                return False
        else:
            logger.warning("⚠ Echo VLM not fully initialized")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to test Echo VLM + RAG: {e}")
        return False


def test_realtime_analyzer_integration():
    """Test RealtimeVeinAnalyzer with full 4-stage pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Verify RealtimeVeinAnalyzer 4-stage pipeline")
    logger.info("=" * 60)

    try:
        from realtime_vein_analyzer import RealtimeVeinAnalyzer
        from app import retrieve_context, load_qdrant_client, qdrant_client

        # Ensure Qdrant is ready
        if qdrant_client is None:
            load_qdrant_client()

        # Create synthetic test image
        test_image, fascia_y, veins = create_synthetic_ultrasound_image()

        logger.info("Creating RealtimeVeinAnalyzer with RAG-enabled VLM...")
        analyzer = RealtimeVeinAnalyzer(
            device='cpu',  # Use CPU for testing
            enable_vlm=True,
            vlm_config={
                'retrieve_context_fn': retrieve_context  # ← Pass RAG function
            }
        )

        logger.info("✅ RealtimeVeinAnalyzer created with VLM + RAG config")

        # Test analyze frame (will use CNN but may skip VLM if model not loaded)
        logger.info("Analyzing test frame...")
        result = analyzer.analyze_frame(test_image)

        logger.info(f"✅ Frame analysis completed")
        logger.info(f"   Fascia detected: {result.fascia_detected}")
        logger.info(f"   Fascia Y: {result.fascia_y}")
        logger.info(f"   Veins found: {len(result.veins)}")
        logger.info(f"   Processing time: {result.processing_time*1000:.1f}ms")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to test RealtimeVeinAnalyzer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vein_detection_service_with_rag():
    """Test VeinDetectionService with RAG injection"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Verify VeinDetectionService with RAG")
    logger.info("=" * 60)

    try:
        from vein_detection_service import get_vein_detection_service
        from app import retrieve_context, load_qdrant_client, qdrant_client

        # Ensure Qdrant is ready
        if qdrant_client is None:
            load_qdrant_client()

        logger.info("Getting VeinDetectionService with RAG function...")
        service = get_vein_detection_service(retrieve_context_fn=retrieve_context)

        if service.retrieve_context_fn:
            logger.info("✅ Service received retrieve_context_fn")
        else:
            logger.warning("⚠ Service does not have retrieve_context_fn")

        # Create synthetic test image
        test_image, fascia_y, veins = create_synthetic_ultrasound_image()

        logger.info("Analyzing image frame with RAG-enabled VLM...")
        result = service.analyze_image_frame(
            image_data=test_image,
            enable_vlm=True,
            return_visualizations=False
        )

        logger.info("✅ Image analysis completed")
        logger.info(f"   Fascia detected: {result['fascia_detected']}")
        logger.info(f"   Veins found: {result['num_veins']}")
        logger.info(f"   Processing time: {result['processing_time_ms']:.1f}ms")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to test VeinDetectionService: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    logger.info("\n" + "🧪 " * 30)
    logger.info("VEIN DETECTION + ECHO VLM + QDRANT RAG INTEGRATION TEST")
    logger.info("🧪 " * 30 + "\n")

    results = {
        "Qdrant RAG Retrieval": test_retrieve_context_function(),
        "Echo VLM Initialization": test_echo_vlm_initialization(),
        "Echo VLM + RAG Integration": test_echo_vlm_with_rag(),
        "RealtimeVeinAnalyzer Pipeline": test_realtime_analyzer_integration(),
        "VeinDetectionService with RAG": test_vein_detection_service_with_rag(),
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "⚠️ WARN/FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)

    if passed == total:
        logger.info("✅ All integration tests PASSED!")
        return 0
    else:
        logger.info(f"⚠️ {total - passed} test(s) need attention (may be due to missing model weights)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
