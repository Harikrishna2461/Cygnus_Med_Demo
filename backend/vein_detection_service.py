"""
Vein Detection Service - Unified API for Task-3
Combines Vision Transformer + Echo VLM for medical-grade ultrasound vein analysis
"""

import cv2
import numpy as np
import torch
import logging
import json
import base64
from pathlib import Path
from typing import Dict, Optional, Tuple
from io import BytesIO
from PIL import Image
import tempfile

from realtime_vein_analyzer import RealtimeVeinAnalyzer
from echo_vlm_integration import EchoVLMIntegration

logger = logging.getLogger(__name__)


class VeinDetectionService:
    """Service for vein detection and classification"""

    _instance = None
    _model = None
    _analyzer = None

    def __new__(cls, retrieve_context_fn=None):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, retrieve_context_fn=None):
        """Initialize service (singleton)"""
        if self._initialized:
            # Update retrieve_context_fn if provided
            if retrieve_context_fn is not None:
                self.retrieve_context_fn = retrieve_context_fn
            return

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._initialized = True
        self.retrieve_context_fn = retrieve_context_fn

        logger.info(f"Initializing VeinDetectionService on {self.device}")

        # Initialize analyzer (model will be lazy-loaded)
        self._analyzer = None
        self._vlm = None

    @property
    def analyzer(self) -> RealtimeVeinAnalyzer:
        """Lazy-load analyzer"""
        if self._analyzer is None:
            logger.info("Initializing RealtimeVeinAnalyzer...")
            vlm_config = {'use_local': True}
            if self.retrieve_context_fn:
                vlm_config['retrieve_context_fn'] = self.retrieve_context_fn
            self._analyzer = RealtimeVeinAnalyzer(
                device=self.device,
                enable_vlm=True,
                vlm_config=vlm_config
            )
        return self._analyzer

    def analyze_image_frame(
        self,
        image_data: np.ndarray,
        enable_vlm: bool = True,
        return_visualizations: bool = False,
    ) -> Dict:
        """
        Analyze a single ultrasound image frame

        Args:
            image_data: Image array (BGR or RGB)
            enable_vlm: Enable Echo VLM verification
            return_visualizations: Return annotated image

        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing single image frame...")

        try:
            # Analyze frame
            result = self.analyzer.analyze_frame(image_data)

            # Prepare output
            output = {
                'fascia_detected': result.fascia_detected,
                'fascia_y': result.fascia_y,
                'veins': [],
                'processing_time_ms': result.processing_time * 1000,
                'num_veins': len(result.veins),
            }

            # Add vein details
            for vein in result.veins:
                vein_info = {
                    'id': vein['id'],
                    'x': vein['x'],
                    'y': vein['y'],
                    'radius': vein['radius'],
                    'area': float(vein['area']),
                }

                # Add classification if available
                for clf in result.classifications:
                    if clf.get('vein_id') == vein['id']:
                        vein_info.update({
                            'n_level': clf.get('classification', 'N2'),
                            'primary_classification': cls._map_n_level_to_classification(
                                clf.get('classification', 'N2')
                            ),
                            'confidence': float(clf.get('confidence', 0.5)),
                            'reasoning': clf.get('reasoning', ''),
                            'distance_to_fascia_mm': clf.get('distance_to_fascia'),
                            'relative_position': clf.get('relative_position'),
                        })
                        break

                output['veins'].append(vein_info)

            # Return visualization
            if return_visualizations:
                annotated = self.analyzer.annotate_frame(image_data, result)
                vis_base64 = self._encode_image_to_base64(annotated)
                output['visualization'] = {
                    'data': vis_base64,
                    'format': 'png'
                }

            logger.info(f"✓ Frame analysis complete: {len(result.veins)} veins detected")
            return output

        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            raise

    def analyze_video_file(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
        save_output: bool = False,
        crop_mode: str = 'none',
    ) -> Dict:
        """
        Analyze ultrasound video file

        Args:
            video_path: Path to input video
            max_frames: Maximum frames to process
            skip_frames: Skip every nth frame
            save_output: Save annotated video
            crop_mode: 'none' (default), 'auto' (detect ROI), or 'square' (center square crop)

        Returns:
            Analysis summary with per-frame results
        """
        logger.info(f"Analyzing video: {video_path} (crop_mode={crop_mode})")

        try:
            # Determine output path
            output_path = None
            if save_output:
                input_path = Path(video_path)
                output_path = str(input_path.parent / f"{input_path.stem}_analyzed.mp4")

            # Process video with ROI cropping support
            summary = self.analyzer.process_video(
                video_path,
                output_video_path=output_path,
                max_frames=max_frames,
                skip_frames=skip_frames,
                crop_mode=crop_mode,
            )

            # Format output
            output = {
                'video_path': video_path,
                'total_frames_processed': summary.get('total_frames_processed'),
                'total_frames': summary.get('total_frames'),
                'fps': summary.get('fps'),
                'resolution': summary.get('resolution'),
                'output_video': output_path,
                'processing_stats': {
                    'avg_processing_time_ms': summary.get('avg_processing_time', 0) * 1000,
                    'total_veins': summary.get('total_veins_detected', 0),
                    'avg_veins_per_frame': summary.get('avg_veins_per_frame', 0),
                    'fascia_detection_rate': summary.get('fascia_detection_rate', 0),
                },
                'frame_results': self._serialize_frame_results(summary.get('frame_results', [])),
            }

            logger.info(f"✓ Video analysis complete: {output['processing_stats']['total_veins']} veins detected")
            return output

        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            raise

    def get_model_info(self) -> Dict:
        """Get model and service information"""
        return {
            'service': 'VeinDetectionService',
            'device': self.device,
            'model': 'CustomUltrasoundViT',
            'capabilities': [
                'fascia_detection',
                'vein_segmentation',
                'vein_classification_N1_N2_N3',
                'echo_vlm_verification',
                'real_time_video_processing',
            ],
            'input_specs': {
                'image_size': (512, 512),
                'format': 'BGR or RGB',
                'channels': 3,
            },
            'output_specs': {
                'fascia_detection': 'boolean',
                'veins': 'list of detections with N1/N2/N3 classification',
                'confidence_scores': 'per-vein confidence 0-1',
            }
        }

    @staticmethod
    def _map_n_level_to_classification(n_level: str) -> str:
        """Map N1/N2/N3 to classification names"""
        mapping = {
            'N1': 'deep_vein',
            'N2': 'superficial_vein_at_fascia',
            'N3': 'superficial_vein',
        }
        return mapping.get(n_level, 'unknown_vein')

    @staticmethod
    def _encode_image_to_base64(image: np.ndarray) -> str:
        """Encode image to base64 PNG"""
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).tobytes().decode('utf-8')

    @staticmethod
    def _serialize_frame_results(frame_results) -> list:
        """Serialize frame results for JSON output"""
        serialized = []

        for result in frame_results:
            frame_data = {
                'frame_id': result.frame_id,
                'timestamp': result.timestamp,
                'fascia_detected': result.fascia_detected,
                'fascia_y': result.fascia_y,
                'num_veins': len(result.veins),
                'processing_time_ms': result.processing_time * 1000,
                'veins': [
                    {
                        'id': v['id'],
                        'x': v['x'],
                        'y': v['y'],
                        'radius': v['radius'],
                    }
                    for v in result.veins
                ],
                'classifications': result.classifications,
            }
            serialized.append(frame_data)

        return serialized


def get_vein_detection_service(retrieve_context_fn=None) -> VeinDetectionService:
    """Get singleton instance of VeinDetectionService

    Args:
        retrieve_context_fn: Optional function to retrieve RAG context from Qdrant

    Returns:
        VeinDetectionService singleton instance
    """
    service = VeinDetectionService(retrieve_context_fn=retrieve_context_fn)
    return service


if __name__ == "__main__":
    # Test service
    logging.basicConfig(level=logging.INFO)

    service = get_vein_detection_service()

    # Get model info
    info = service.get_model_info()
    print("Service Info:")
    print(json.dumps(info, indent=2))

    # Test on image
    print("\nTesting on image...")
    test_image_path = "/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/Sample_Data/Set 1/1 - Videos/sample_data.mp4"

    if Path(test_image_path).exists():
        # Extract a frame from video for testing
        cap = cv2.VideoCapture(test_image_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            result = service.analyze_image_frame(frame, enable_vlm=False, return_visualizations=True)
            print(f"✓ Image analysis result: {result['num_veins']} veins detected")
