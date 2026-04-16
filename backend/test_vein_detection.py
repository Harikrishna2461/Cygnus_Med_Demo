#!/usr/bin/env python3
"""
End-to-End Testing for Task-3 Vein Detection Pipeline
Tests Vision Transformer + Echo VLM integration with Sample_Data videos
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List
import json
import time

from vein_detection_service import get_vein_detection_service
from vein_dataset import VeinDatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VeinDetectionTester:
    """Comprehensive testing for vein detection system"""

    def __init__(self):
        self.service = get_vein_detection_service()
        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': []
        }

    def test_model_initialization(self) -> bool:
        """Test that model initializes correctly"""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Model Initialization")
        logger.info("="*70)

        try:
            info = self.service.get_model_info()
            assert info is not None, "Model info is None"
            assert info['device'] in ['cuda', 'cpu'], f"Invalid device: {info['device']}"
            assert 'CustomUltrasoundViT' in info['model'], "Wrong model type"

            logger.info(f"✓ Model initialized on {info['device']}")
            logger.info(f"  Capabilities: {', '.join(info['capabilities'][:3])}...")

            self.results['tests_passed'] += 1
            self.results['test_details'].append({
                'test': 'Model Initialization',
                'status': 'PASSED',
                'device': info['device']
            })
            return True

        except Exception as e:
            logger.error(f"✗ Model initialization failed: {e}")
            self.results['tests_failed'] += 1
            self.results['test_details'].append({
                'test': 'Model Initialization',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_single_frame_analysis(self) -> bool:
        """Test analysis of a single ultrasound frame"""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Single Frame Analysis")
        logger.info("="*70)

        try:
            # Create dummy ultrasound frame
            frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

            # Add some structure to make it look like ultrasound
            cv2.ellipse(frame, (256, 256), (100, 150), 0, 0, 360, (150, 150, 150), -1)

            start_time = time.time()
            result = self.service.analyze_image_frame(frame, enable_vlm=False, return_visualizations=True)
            elapsed = time.time() - start_time

            assert result is not None, "Result is None"
            assert 'fascia_detected' in result, "Missing fascia_detected key"
            assert 'veins' in result, "Missing veins key"
            assert 'processing_time_ms' in result, "Missing processing_time_ms key"

            logger.info(f"✓ Frame analysis successful")
            logger.info(f"  Fascia detected: {result['fascia_detected']}")
            logger.info(f"  Veins found: {len(result['veins'])}")
            logger.info(f"  Processing time: {result['processing_time_ms']:.1f}ms")

            self.results['tests_passed'] += 1
            self.results['test_details'].append({
                'test': 'Single Frame Analysis',
                'status': 'PASSED',
                'veins_detected': len(result['veins']),
                'processing_time_ms': result['processing_time_ms']
            })
            return True

        except Exception as e:
            logger.error(f"✗ Frame analysis failed: {e}")
            self.results['tests_failed'] += 1
            self.results['test_details'].append({
                'test': 'Single Frame Analysis',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_vein_classification(self) -> bool:
        """Test vein classification output format"""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Vein Classification Output")
        logger.info("="*70)

        try:
            frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

            result = self.service.analyze_image_frame(frame, enable_vlm=False)

            # Check vein structure
            for vein in result.get('veins', []):
                assert 'id' in vein, "Missing vein id"
                assert 'x' in vein, "Missing vein x"
                assert 'y' in vein, "Missing vein y"
                assert 'n_level' in vein, "Missing vein n_level"
                assert vein['n_level'] in ['N1', 'N2', 'N3'], f"Invalid n_level: {vein['n_level']}"
                assert 'confidence' in vein, "Missing vein confidence"

            logger.info(f"✓ Vein classification format validated")
            logger.info(f"  Veins have correct N1/N2/N3 labels: {all(v.get('n_level') in ['N1', 'N2', 'N3'] for v in result.get('veins', []))}")

            self.results['tests_passed'] += 1
            self.results['test_details'].append({
                'test': 'Vein Classification Output',
                'status': 'PASSED',
                'sample_vein': result.get('veins', [{}])[0] if result.get('veins') else {}
            })
            return True

        except Exception as e:
            logger.error(f"✗ Classification validation failed: {e}")
            self.results['tests_failed'] += 1
            self.results['test_details'].append({
                'test': 'Vein Classification Output',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_sample_data_loading(self) -> bool:
        """Test loading videos from Sample_Data"""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Sample Data Loading")
        logger.info("="*70)

        try:
            sample_data_root = Path("/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/Sample_Data")

            if not sample_data_root.exists():
                logger.warning("⚠ Sample_Data folder not found, skipping")
                self.results['test_details'].append({
                    'test': 'Sample Data Loading',
                    'status': 'SKIPPED',
                    'reason': 'Sample_Data not found'
                })
                return True

            # Count videos
            video_dirs = [
                sample_data_root / "Set 1" / "0 - Raw videos",
                sample_data_root / "Set 1" / "1 - Videos",
                sample_data_root / "Set 1" / "2 - Annotated videos",
                sample_data_root / "Set 1" / "3 - Simple Annotated videos",
            ]

            total_videos = 0
            for vdir in video_dirs:
                if vdir.exists():
                    count = len(list(vdir.glob("*.mp4")))
                    total_videos += count
                    logger.info(f"  {vdir.name}: {count} videos")

            assert total_videos > 0, "No videos found in Sample_Data"

            logger.info(f"✓ Sample_Data loaded successfully")
            logger.info(f"  Total videos found: {total_videos}")

            self.results['tests_passed'] += 1
            self.results['test_details'].append({
                'test': 'Sample Data Loading',
                'status': 'PASSED',
                'total_videos': total_videos
            })
            return True

        except Exception as e:
            logger.error(f"✗ Sample data loading failed: {e}")
            self.results['tests_failed'] += 1
            self.results['test_details'].append({
                'test': 'Sample Data Loading',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_video_analysis_on_sample(self) -> bool:
        """Test video analysis on actual Sample_Data video"""
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Video Analysis on Sample Data")
        logger.info("="*70)

        try:
            # Find first available video
            sample_data_root = Path("/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/Sample_Data")
            video_path = None

            for vdir_name in ["1 - Videos", "0 - Raw videos"]:
                vdir = sample_data_root / "Set 1" / vdir_name
                if vdir.exists():
                    videos = list(vdir.glob("*.mp4"))
                    if videos:
                        video_path = str(videos[0])
                        break

            if not video_path:
                logger.warning("⚠ No sample videos found, skipping")
                self.results['test_details'].append({
                    'test': 'Video Analysis on Sample Data',
                    'status': 'SKIPPED',
                    'reason': 'No sample videos found'
                })
                return True

            logger.info(f"  Analyzing: {Path(video_path).name}")

            start_time = time.time()
            result = self.service.analyze_video_file(
                video_path,
                max_frames=10,  # Limit for testing
                skip_frames=2,
                save_output=False
            )
            elapsed = time.time() - start_time

            assert result is not None, "Video analysis result is None"
            assert 'total_frames_processed' in result, "Missing total_frames_processed"
            assert 'processing_stats' in result, "Missing processing_stats"

            logger.info(f"✓ Video analysis successful")
            logger.info(f"  Frames processed: {result['total_frames_processed']}")
            logger.info(f"  Total veins detected: {result['processing_stats']['total_veins']}")
            logger.info(f"  Total time: {elapsed:.1f}s")
            logger.info(f"  Avg time/frame: {result['processing_stats']['avg_processing_time_ms']:.1f}ms")

            self.results['tests_passed'] += 1
            self.results['test_details'].append({
                'test': 'Video Analysis on Sample Data',
                'status': 'PASSED',
                'frames_processed': result['total_frames_processed'],
                'total_veins': result['processing_stats']['total_veins'],
                'avg_time_ms': result['processing_stats']['avg_processing_time_ms']
            })
            return True

        except Exception as e:
            logger.error(f"✗ Video analysis failed: {e}")
            self.results['tests_failed'] += 1
            self.results['test_details'].append({
                'test': 'Video Analysis on Sample Data',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_gpu_availability(self) -> bool:
        """Test GPU availability"""
        logger.info("\n" + "="*70)
        logger.info("TEST 6: GPU Availability")
        logger.info("="*70)

        try:
            has_cuda = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if has_cuda else 0

            if has_cuda:
                logger.info(f"✓ CUDA available")
                logger.info(f"  Devices: {device_count}")
                for i in range(device_count):
                    logger.info(f"    {i}: {torch.cuda.get_device_name(i)}")
            else:
                logger.warning("⚠ CUDA not available, using CPU (slower)")

            self.results['tests_passed'] += 1
            self.results['test_details'].append({
                'test': 'GPU Availability',
                'status': 'PASSED',
                'cuda_available': has_cuda,
                'device_count': device_count
            })
            return True

        except Exception as e:
            logger.error(f"✗ GPU check failed: {e}")
            self.results['tests_failed'] += 1
            self.results['test_details'].append({
                'test': 'GPU Availability',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n" + "█" * 70)
        logger.info("VEIN DETECTION SYSTEM - END-TO-END TEST SUITE")
        logger.info("█" * 70)

        tests = [
            self.test_gpu_availability,
            self.test_model_initialization,
            self.test_single_frame_analysis,
            self.test_vein_classification,
            self.test_sample_data_loading,
            self.test_video_analysis_on_sample,
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test execution error: {e}")

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)

        total = self.results['tests_passed'] + self.results['tests_failed']
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {self.results['tests_passed']} ✓")
        logger.info(f"Failed: {self.results['tests_failed']} ✗")

        if self.results['tests_failed'] == 0:
            logger.info("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            logger.warning(f"\n⚠️ {self.results['tests_failed']} test(s) failed. See details above.")

        # Save results
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nDetailed results saved to: {results_file}")

        return self.results['tests_failed'] == 0


if __name__ == "__main__":
    tester = VeinDetectionTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)
