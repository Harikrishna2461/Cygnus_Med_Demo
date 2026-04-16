"""
Real-time Ultrasound Vein Analysis Pipeline
Combines Vision Transformer with Echo VLM for live video processing
"""

import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from queue import Queue
from threading import Thread
import time
from dataclasses import dataclass

from vein_detector_vit import CustomUltrasoundViT, VeinDetectionConfig, VeinDetectionPostProcessor
from echo_vlm_integration import EchoVLMIntegration

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysisResult:
    """Result of analyzing a single frame"""
    frame_id: int
    timestamp: float
    fascia_detected: bool
    fascia_y: Optional[int]
    veins: List[Dict]
    classifications: List[Dict]
    processing_time: float
    confidence_scores: Dict


class RealtimeVeinAnalyzer:
    """Real-time vein detection and classification for video streams"""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = 'cuda',
        enable_vlm: bool = True,
        vlm_config: Optional[Dict] = None,
    ):
        """
        Initialize real-time analyzer

        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            enable_vlm: Enable Echo VLM verification
            vlm_config: Echo VLM configuration
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.enable_vlm = enable_vlm

        # Load Vision Transformer model
        logger.info(f"Loading model on {self.device}...")
        config = VeinDetectionConfig()
        self.model = CustomUltrasoundViT(config).to(self.device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No model checkpoint found, using untrained model")

        # Initialize post-processor
        self.post_processor = VeinDetectionPostProcessor(image_size=512, patch_size=16)

        # Initialize Echo VLM
        if enable_vlm:
            vlm_config = vlm_config or {}
            # Extract retrieve_context_fn if provided
            retrieve_context_fn = vlm_config.pop('retrieve_context_fn', None)
            # Remove unknown parameters
            vlm_config.pop('use_local', None)  # Not used by EchoVLMIntegration
            self.vlm = EchoVLMIntegration(
                retrieve_context_fn=retrieve_context_fn,
                **vlm_config
            )
        else:
            self.vlm = None

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess frame for model input"""
        original_size = frame.shape[:2]

        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (512, 512))

        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)

        return frame_tensor, original_size

    @torch.no_grad()
    def detect_veins(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect fascia and veins in frame

        Returns:
            fascia_mask, vein_mask, classification_mask
        """
        frame_tensor, _ = self.preprocess_frame(frame)

        # Forward pass
        outputs = self.model(frame_tensor)

        # Post-process
        fascia_logits = outputs['fascia_logits'][0]  # Remove batch dim
        vein_logits = outputs['vein_logits'][0]
        classification_logits = outputs['classification_logits'][0]

        # Convert to numpy
        fascia_mask = self.post_processor.logits_to_segmentation(fascia_logits.unsqueeze(0))
        vein_mask = self.post_processor.logits_to_segmentation(vein_logits.unsqueeze(0))
        classification_mask = self.post_processor.logits_to_segmentation(classification_logits.unsqueeze(0))

        return fascia_mask[0], vein_mask[0], classification_mask[0]

    def extract_vein_detections(
        self,
        vein_mask: np.ndarray,
        min_area: int = 20,
    ) -> List[Dict]:
        """Extract individual vein detections from mask"""
        veins = []

        # Find contours
        contours, _ = cv2.findContours(vein_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour_id, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)

            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])

            vein = {
                'id': contour_id,
                'x': x,
                'y': y,
                'radius': radius,
                'area': area,
                'contour': contour,
            }

            veins.append(vein)

        return veins

    def extract_fascia_line(self, fascia_mask: np.ndarray) -> Optional[int]:
        """Extract fascia Y-coordinate from mask"""
        # Find the dominant horizontal structure
        horizontal_sum = np.sum(fascia_mask > 0, axis=1)

        if np.max(horizontal_sum) == 0:
            return None

        fascia_y = np.argmax(horizontal_sum)
        return fascia_y

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> FrameAnalysisResult:
        """
        Analyze single frame

        Args:
            frame: Input frame (BGR)
            frame_id: Frame number
            timestamp: Timestamp in video

        Returns:
            FrameAnalysisResult
        """
        start_time = time.time()

        # Detect fascia and veins
        fascia_mask, vein_mask, classification_mask = self.detect_veins(frame)

        # Extract fascia position
        fascia_y = self.extract_fascia_line(fascia_mask)
        fascia_detected = fascia_y is not None

        # Extract vein detections
        veins = self.extract_vein_detections(vein_mask)

        # Prepare results
        result = FrameAnalysisResult(
            frame_id=frame_id,
            timestamp=timestamp,
            fascia_detected=fascia_detected,
            fascia_y=fascia_y,
            veins=veins,
            classifications=[],
            processing_time=0.0,
            confidence_scores={}
        )

        # Echo VLM verification and classification
        if self.enable_vlm and self.vlm and veins:
            try:
                vlm_results = self.vlm.comprehensive_analysis(
                    frame, veins, fascia_y
                )

                # Store VLM results
                result.classifications = vlm_results.get('classifications', [])
                result.confidence_scores = {
                    'fascia': vlm_results['fascia']['confidence'],
                    'veins': [v.get('vlm_confidence', 0.5) for v in vlm_results.get('validated_veins', [])],
                }

                # Merge VLM results with vein detections
                for vein, validated in zip(veins, vlm_results.get('validated_veins', [])):
                    vein['vlm_valid'] = validated.get('vlm_valid', True)
                    vein['vlm_confidence'] = validated.get('vlm_confidence', 0.5)

            except Exception as e:
                logger.warning(f"VLM analysis failed: {e}")

        # Calculate processing time
        result.processing_time = time.time() - start_time

        return result

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: FrameAnalysisResult,
        draw_segmentation: bool = False,
    ) -> np.ndarray:
        """
        Annotate frame with detection results

        Args:
            frame: Original frame (BGR)
            result: Analysis result
            draw_segmentation: Draw segmentation overlays

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw fascia
        if result.fascia_detected and result.fascia_y is not None:
            y = result.fascia_y
            cv2.line(annotated, (0, y), (annotated.shape[1], y), (0, 255, 0), 3)
            cv2.putText(
                annotated, "Fascia",
                (10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

        # Draw veins with classification
        color_map = {
            'N1': (255, 0, 0),  # Blue
            'N2': (0, 255, 0),  # Green
            'N3': (0, 0, 255),  # Red
        }

        for vein in result.veins:
            vein_id = vein['id']

            # Find classification for this vein
            classification = None
            for clf in result.classifications:
                if clf.get('vein_id') == vein_id:
                    classification = clf.get('classification', 'N2')
                    break

            if classification is None:
                classification = 'N2'  # Default

            color = color_map.get(classification, (255, 255, 255))

            # Draw circle
            cv2.circle(annotated, (vein['x'], vein['y']), vein['radius'], color, 3)

            # Draw label
            label = f"V{vein_id}-{classification}"
            cv2.putText(
                annotated, label,
                (vein['x'] - 20, vein['y'] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )

            # Draw confidence
            if result.confidence_scores.get('veins') and vein_id < len(result.confidence_scores['veins']):
                conf = result.confidence_scores['veins'][vein_id]
                cv2.putText(
                    annotated, f"{conf:.1%}",
                    (vein['x'] - 20, vein['y'] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1
                )

        # Draw stats
        stats_text = [
            f"Frame: {result.frame_id}",
            f"Veins: {len(result.veins)}",
            f"Processing: {result.processing_time*1000:.1f}ms",
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(
                annotated, text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

        return annotated

    def process_video(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ) -> Dict:
        """
        Process video file and optionally save annotated output

        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output (optional)
            max_frames: Maximum frames to process
            skip_frames: Process every nth frame

        Returns:
            Dictionary with analysis summary
        """
        logger.info(f"Opening video: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_video_path}")
            return {}

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

        # Initialize output video writer
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Analysis results
        frame_results = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue

                # Analyze frame
                timestamp = frame_count / fps
                result = self.analyze_frame(frame, frame_id=frame_count, timestamp=timestamp)
                frame_results.append(result)

                # Annotate and save
                annotated = self.annotate_frame(frame, result)

                if out:
                    out.write(annotated)

                logger.info(
                    f"Frame {frame_count}: {len(result.veins)} veins, "
                    f"fascia={result.fascia_detected}, "
                    f"time={result.processing_time*1000:.1f}ms"
                )

                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames...")

        finally:
            cap.release()
            if out:
                out.release()

        logger.info(f"Video processing complete. Processed {frame_count} frames")

        # Compile summary
        summary = {
            'total_frames_processed': frame_count,
            'total_frames': total_frames,
            'fps': fps,
            'resolution': (width, height),
            'output_video': output_video_path,
            'avg_processing_time': np.mean([r.processing_time for r in frame_results]) if frame_results else 0,
            'total_veins_detected': sum(len(r.veins) for r in frame_results),
            'avg_veins_per_frame': sum(len(r.veins) for r in frame_results) / max(frame_count, 1),
            'fascia_detection_rate': sum(r.fascia_detected for r in frame_results) / max(frame_count, 1),
            'frame_results': frame_results,
        }

        return summary


if __name__ == "__main__":
    # Test real-time analyzer
    logging.basicConfig(level=logging.INFO)

    analyzer = RealtimeVeinAnalyzer(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_vlm=False,  # Disable VLM for testing
    )

    # Test on sample video
    test_video = "/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/Sample_Data/Set 1/1 - Videos/sample_data.mp4"

    if Path(test_video).exists():
        results = analyzer.process_video(
            test_video,
            output_video_path="/tmp/annotated_output.mp4",
            max_frames=50,
            skip_frames=2,
        )

        print(f"Analysis Summary:")
        print(f"  Total frames: {results.get('total_frames_processed')}")
        print(f"  Avg processing time: {results.get('avg_processing_time')*1000:.1f}ms")
        print(f"  Total veins: {results.get('total_veins_detected')}")
        print(f"  Avg veins/frame: {results.get('avg_veins_per_frame'):.1f}")
