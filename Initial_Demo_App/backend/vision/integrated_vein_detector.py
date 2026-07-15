"""
Integrated vein detection: fascia + blobs + classification.
Detects fascia, blobs (veins), and classifies each vein relative to fascia.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from vision.blob_detector import BlobDetector
from vision.segmentation.unet_fascia import FasciaDetector
from vision.classification.vein_classifier import VeinClassifier


class IntegratedVeinDetector:
    """
    Complete vein detection pipeline:
    1. Detect fascia using UNet
    2. Detect veins (blobs) using SimpleBlobDetector + KLT
    3. Classify veins relative to fascia (N1/N2/N3)
    """
    
    def __init__(
        self,
        fascia_model_path: Optional[str] = None,
        device: str = 'cpu',
        detect_fascia: bool = True
    ):
        """
        Initialize integrated detector.
        
        Args:
            fascia_model_path: path to trained UNet fascia model
            device: 'cpu' or 'cuda'
            detect_fascia: whether to detect fascia (requires trained model)
        """
        self.blob_detector = BlobDetector()
        self.device = device
        self.detect_fascia_flag = detect_fascia
        self.vein_classifier = VeinClassifier()
        
        # Fascia detector (optional)
        if detect_fascia:
            try:
                self.fascia_detector = FasciaDetector(
                    model_path=fascia_model_path,
                    device=device
                )
                print("✓ Fascia detector initialized")
            except Exception as e:
                print(f"⚠ Fascia detector failed to initialize: {e}")
                print("  Proceeding without fascia detection")
                self.fascia_detector = None
        else:
            self.fascia_detector = None
    
    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Process frame: detect fascia, blobs, and classify veins.
        
        Args:
            frame_bgr: input frame (H, W, 3) in BGR format
        
        Returns:
            dict with keys:
                'image': annotated output image
                'targets': list of detected targets with vein classifications
                'fascia': fascia detection result
                'vein_summary': summary of detected veins by type
                'success': whether detection was successful
        """
        
        # Step 1: Detect blobs
        blob_image, blob_result = self.blob_detector.process_frame(frame_bgr)
        
        # Step 2: Detect fascia
        fascia_result = None
        if self.fascia_detector:
            try:
                fascia_result = self.fascia_detector.detect(
                    frame_bgr,
                    threshold=0.5,
                    return_boundary=True
                )
            except Exception as e:
                print(f"⚠ Fascia detection failed: {e}")
                fascia_result = None
        
        # Step 3: Get blob detections and classify
        targets_with_classification = []
        
        if blob_result.get('targets'):
            # Get blob states from detector
            blobs = self.blob_detector.s.blobs if hasattr(self.blob_detector, 's') else {}
            
            # Classify blobs
            classifications = self.vein_classifier.classify_blobs(
                blobs,
                fascia_result if fascia_result else {}
            )
            
            # Merge blob data with classifications
            for target in blob_result['targets']:
                blob_id = target['id']
                classification = classifications.get(blob_id)
                
                if classification:
                    target['vein_type'] = classification.vein_type
                    target['vein_label'] = classification.vein_label
                    target['position'] = classification.position
                    target['distance_to_fascia'] = classification.distance_to_fascia
                    target['vein_confidence'] = classification.confidence
                
                targets_with_classification.append(target)
        
        # Step 4: Visualize
        output_image = self._visualize(
            blob_image,
            targets_with_classification,
            fascia_result
        )
        
        # Step 5: Summary
        vein_summary = self.vein_classifier.get_summary()
        
        return {
            'image': output_image,
            'targets': targets_with_classification,
            'fascia': fascia_result,
            'vein_summary': vein_summary,
            'success': len(targets_with_classification) > 0,
            'blob_result': blob_result,
        }
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Process entire video.
        
        Args:
            video_path: path to input video
            output_path: path to save output video (optional)
            skip_frames: process every Nth frame (0 = all frames)
            max_frames: maximum frames to process (optional)
            verbose: print progress
        
        Returns:
            dict with overall results and per-frame detections
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detections = []
        frame_count = 0
        processed = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                result = self.process_frame(frame)
                
                # Store detection
                detection_record = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'targets': result['targets'],
                    'vein_summary': result['vein_summary'],
                    'fascia_detected': result['fascia'] is not None,
                }
                detections.append(detection_record)
                
                # Write output video
                if writer:
                    writer.write(result['image'])
                
                frame_count += 1
                processed += 1
                
                if max_frames and processed >= max_frames:
                    break
                
                if verbose and processed % 10 == 0:
                    print(f"  Processed {processed}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # Summary statistics
        total_veins_detected = sum(
            d['vein_summary']['total_veins']
            for d in detections
            if d['vein_summary']
        )
        
        vein_type_counts = {'N1_deep': 0, 'N2_gsv': 0, 'N3_superficial': 0}
        for detection in detections:
            for target in detection['targets']:
                vein_type = target.get('vein_type')
                if vein_type in vein_type_counts:
                    vein_type_counts[vein_type] += 1
        
        result = {
            'video_path': video_path,
            'output_path': output_path,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': processed,
            'detections': detections,
            'summary': {
                'total_veins': total_veins_detected,
                'deep_veins': vein_type_counts['N1_deep'],
                'gsv': vein_type_counts['N2_gsv'],
                'superficial_veins': vein_type_counts['N3_superficial'],
            }
        }
        
        return result
    
    def _visualize(
        self,
        image: np.ndarray,
        targets: list,
        fascia_result: Optional[Dict]
    ) -> np.ndarray:
        """
        Draw vein classifications on image.
        
        Args:
            image: base image with blob detections
            targets: list of targets with vein classifications
            fascia_result: fascia detection result
        
        Returns:
            annotated image
        """
        output = image.copy()
        
        # Draw fascia if available
        if fascia_result:
            # Draw fascia line
            if 'center' in fascia_result and fascia_result['center']:
                cy = int(fascia_result['center'][1])
                cv2.line(output, (0, cy), (output.shape[1], cy), (100, 100, 100), 2)
                cv2.putText(
                    output, 'Fascia',
                    (10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2
                )
            
            # Draw fascia boundary if available
            if 'boundary' in fascia_result and fascia_result['boundary']:
                boundary = fascia_result['boundary']
                if boundary:
                    pts = np.array(boundary, dtype=np.int32)
                    cv2.polylines(output, [pts], False, (100, 100, 100), 1)
        
        # Draw vein classifications
        vein_colors = {
            'N1_deep': (0, 255, 0),          # Green
            'N2_gsv': (255, 0, 255),         # Magenta
            'N3_superficial': (0, 165, 255)  # Orange
        }
        
        vein_labels = {
            'N1_deep': 'Deep Vein (N1)',
            'N2_gsv': 'GSV (N2)',
            'N3_superficial': 'Superficial (N3)'
        }
        
        for target in targets:
            cx, cy = target['center']
            vein_type = target.get('vein_type', 'unknown')
            vein_label = vein_labels.get(vein_type, 'Unknown')
            vein_conf = target.get('vein_confidence', 0)
            
            color = vein_colors.get(vein_type, (200, 200, 200))
            
            # Draw enhanced annotations
            cv2.putText(
                output,
                f"{vein_label} #{target['id']}",
                (int(cx) - 50, int(cy) - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            cv2.putText(
                output,
                f"Conf: {vein_conf:.0f}%",
                (int(cx) - 50, int(cy) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
            
            position = target.get('position', 'unknown')
            if position != 'unknown':
                cv2.putText(
                    output,
                    f"Pos: {position}",
                    (int(cx) - 50, int(cy) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
                )
        
        return output
    
    def get_vein_statistics(self) -> Dict:
        """Get overall vein classification statistics."""
        return self.vein_classifier.get_summary()
