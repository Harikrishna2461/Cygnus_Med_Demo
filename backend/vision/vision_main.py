"""
Main Orchestrator for Ultrasound Vein Detection Pipeline

Coordinates all modules: frame extraction, segmentation, geometry, classification, and visualization.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

from .video.frame_extractor import FrameExtractor
from .segmentation.unet_detector import PretrainedSegmentationDetector
from .geometry.spatial_analysis import SpatialAnalyzer
from .classification.rules import VeinClassifier
from .classification.llm_interface import VisionLLMInterface
from .utils.visualization import UltrasoundVisualizer

logger = logging.getLogger(__name__)


class VeinDetectionPipeline:
    """
    Complete pipeline for ultrasound vein detection and classification
    
    Workflow:
    1. Extract frames from video
    2. Segment fascia and veins
    3. Compute spatial relationships
    4. Classify veins
    5. Use LLM for ambiguous cases
    6. Visualize results
    """
    
    def __init__(
        self,
        enable_llm: bool = False,
        llm_provider: str = "openai",
        llm_api_key: Optional[str] = None,
        llm_model: str = "neural-chat",
        pixels_per_mm: float = 1.0,
        target_fps: int = 5,
        resize_shape: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize pipeline
        
        Args:
            enable_llm: Whether to use Vision LLM for classification
            llm_provider: LLM provider ("openai", "anthropic", or "ollama")
            llm_api_key: API key for LLM service (not needed for ollama)
            llm_model: Model name for ollama (default: neural-chat)
            pixels_per_mm: Calibration factor
            target_fps: Target FPS for frame extraction
            resize_shape: Optional (height, width) for frame resizing
        """
        self.enable_llm = enable_llm
        self.llm_api_key = llm_api_key
        
        # Initialize modules
        self.frame_extractor = FrameExtractor(target_fps=target_fps, resize_shape=resize_shape)
        self.segmenter = PretrainedSegmentationDetector()
        self.spatial_analyzer = SpatialAnalyzer(pixels_per_mm=pixels_per_mm)
        self.classifier = VeinClassifier()
        self.visualizer = UltrasoundVisualizer()
        
        # Initialize LLM if enabled
        self.llm = None
        if enable_llm:
            try:
                self.llm = VisionLLMInterface(
                    model_provider=llm_provider,
                    api_key=llm_api_key,
                    model_name=llm_model
                )
                logger.info(f"✓ Vision LLM enabled ({llm_provider}:{llm_model})")
            except Exception as e:
                logger.warning(f"⚠ Could not initialize LLM: {e}")
                self.llm = None
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        save_visualizations: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Process entire video and analyze all frames
        
        Args:
            video_path: Path to ultrasound video
            max_frames: Maximum frames to process
            save_visualizations: Whether to save annotated frames
            output_dir: Directory for saving outputs (optional)
        
        Returns:
            Dictionary with analysis results for all frames
        """
        logger.info(f"Starting video processing: {video_path}")
        
        # Extract frames
        frames = self.frame_extractor.extract_frames(video_path, max_frames=max_frames)
        logger.info(f"✓ Extracted {len(frames)} frames")
        
        # Process each frame
        frame_results = []
        visualization_outputs = []
        
        for frame_idx, frame in enumerate(frames):
            logger.info(f"Processing frame {frame_idx + 1}/{len(frames)}")
            
            result = self.process_frame(frame, frame_idx)
            frame_results.append(result)
            
            # Save visualization if requested
            if save_visualizations and result.get('visualizations'):
                viz = result['visualizations']
                visualization_outputs.append(viz)
        
        # Create summary report
        summary = self._create_summary_report(frame_results)
        
        # Optionally save outputs
        if save_visualizations and output_dir:
            self._save_outputs(frame_results, visualization_outputs, output_dir)
        
        return {
            "video_path": video_path,
            "total_frames_processed": len(frames),
            "frame_results": frame_results,
            "summary": summary
        }
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict:
        """
        Process single ultrasound frame
        
        Args:
            frame: Input frame (BGR)
            frame_idx: Frame index
        
        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"Processing frame {frame_idx}")
        
        try:
            # Step 1: Segment fascia and veins
            logger.info("  [1/5] Segmenting fascia and veins...")
            fascia_mask = self.segmenter.segment_fascia(frame)
            # Pass fascia_mask to vein detection so veins are classified by their position relative to fascia
            vein_masks = self.segmenter.segment_veins(frame, fascia_mask=fascia_mask)
            
            logger.info(f"  Found {len(vein_masks)} vein candidates")
            
            # Step 2: Analyze spatial relationships
            logger.info("  [2/5] Analyzing spatial relationships...")
            veins_with_spatial = self.spatial_analyzer.batch_analyze_veins(vein_masks, fascia_mask)
            
            # Step 3: Classify veins
            logger.info("  [3/5] Classifying veins...")
            veins_classified = self.classifier.classify_batch(veins_with_spatial)
            
            # Step 4: LLM analysis (if enabled)
            veins_final = veins_classified
            if self.llm and self.enable_llm:
                logger.info("  [4/5] Running LLM analysis...")
                
                # Create overlay visualization
                overlay = self.visualizer.visualize_classification(frame, fascia_mask, veins_classified)
                
                # Classify with LLM
                veins_final = self.llm.classify_veins_with_llm(frame, veins_classified, overlay)
                logger.info("  ✓ LLM analysis complete")
            else:
                logger.info("  [4/5] Skipping LLM analysis (disabled)")
            
            # Step 5: Visualization
            logger.info("  [5/5] Creating visualizations...")
            viz_output = self._create_visualizations(frame, fascia_mask, veins_final)
            
            # Compile results
            return {
                "frame_index": frame_idx,
                "timestamp": datetime.now().isoformat(),
                "segmentation": {
                    "fascia_mask_shape": fascia_mask.shape,
                    "num_veins_detected": len(veins_final)
                },
                "veins": veins_final,
                "summary_statistics": self._compute_frame_statistics(veins_final),
                "visualizations": viz_output
            }
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            return {
                "frame_index": frame_idx,
                "error": str(e),
                "veins": [],
                "summary_statistics": {}
            }
    
    def _create_visualizations(
        self,
        frame: np.ndarray,
        fascia_mask: np.ndarray,
        veins: List[Dict]
    ) -> Dict:
        """Create various visualization outputs"""
        
        vein_masks_only = [v.get('mask') for v in veins if v.get('mask') is not None]
        
        viz = {
            "segmentation": self.visualizer.visualize_segmentation(frame, fascia_mask, vein_masks_only),
            "classification": self.visualizer.visualize_classification(frame, fascia_mask, veins),
            "detailed": self.visualizer.visualize_detailed_analysis(frame, fascia_mask, veins)
        }
        
        # Create comparison grid if all visualizations exist
        try:
            viz["grid"] = self.visualizer.create_comparison_grid(
                frame,
                viz["segmentation"],
                viz["classification"],
                viz["detailed"]
            )
        except Exception as e:
            logger.debug(f"Could not create comparison grid: {e}")
        
        return viz
    
    def _compute_frame_statistics(self, veins: List[Dict]) -> Dict:
        """Compute summary statistics for frame"""
        
        stats = {
            "total_veins": len(veins),
            "by_type": {},
            "by_nlevel": {}
        }
        
        for vein in veins:
            classification = vein.get('classification', {})
            vtype = classification.get('primary_classification', 'unknown')
            nlevel = classification.get('n_level', 'unknown')
            
            stats["by_type"][vtype] = stats["by_type"].get(vtype, 0) + 1
            stats["by_nlevel"][nlevel] = stats["by_nlevel"].get(nlevel, 0) + 1
        
        # Check for GSV
        gsv_found = any(
            vein.get('llm_analysis', {}).get('llm_is_gsv', False)
            for vein in veins
        ) or any(
            vein.get('classification', {}).get('primary_classification') == 'gsv'
            for vein in veins
        )
        
        stats["gsv_present"] = gsv_found
        
        return stats
    
    def _create_summary_report(self, frame_results: List[Dict]) -> Dict:
        """Create overall summary report"""
        
        report = {
            "total_frames": len(frame_results),
            "successful_frames": sum(1 for f in frame_results if 'error' not in f),
            "failed_frames": sum(1 for f in frame_results if 'error' in f),
            "total_veins_detected": sum(
                f.get('summary_statistics', {}).get('total_veins', 0)
                for f in frame_results
            ),
            "vein_type_summary": {},
            "nlevel_summary": {},
            "gsv_frames": []
        }
        
        # Aggregate statistics
        for frame_idx, frame_result in enumerate(frame_results):
            stats = frame_result.get('summary_statistics', {})
            
            # Type distribution
            for vtype, count in stats.get('by_type', {}).items():
                report["vein_type_summary"][vtype] = report["vein_type_summary"].get(vtype, 0) + count
            
            # N-level distribution
            for nlevel, count in stats.get('by_nlevel', {}).items():
                report["nlevel_summary"][nlevel] = report["nlevel_summary"].get(nlevel, 0) + count
            
            # GSV tracking
            if stats.get('gsv_present', False):
                report["gsv_frames"].append(frame_idx)
        
        return report
    
    def _save_outputs(
        self,
        frame_results: List[Dict],
        visualization_outputs: List[Dict],
        output_dir: str
    ):
        """Save analysis results and visualizations to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_path / "analysis_results.json", "w") as f:
            # Convert to serializable format
            serializable_results = []
            for result in frame_results:
                r = result.copy()
                # Remove numpy arrays and visualization data for JSON
                r.pop('visualizations', None)
                serializable_results.append(r)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✓ Saved analysis results to {output_path / 'analysis_results.json'}")
        
        # Save visualizations
        for idx, viz in enumerate(visualization_outputs):
            if 'classification' in viz:
                viz_path = output_path / f"frame_{idx:04d}_classification.png"
                cv2.imwrite(str(viz_path), viz['classification'])
            
            if 'detailed' in viz:
                viz_path = output_path / f"frame_{idx:04d}_detailed.png"
                cv2.imwrite(str(viz_path), viz['detailed'])
        
        logger.info(f"✓ Saved visualizations to {output_path}")


def process_ultrasound_video(
    video_path: str,
    enable_llm: bool = True,
    llm_provider: str = "openai",
    llm_api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_frames: int = 30
) -> Dict:
    """
    High-level function to process ultrasound video
    
    Args:
        video_path: Path to video file
        enable_llm: Use Vision LLM
        llm_provider: LLM provider
        llm_api_key: API key
        output_dir: Where to save results
        max_frames: Max frames to process
    
    Returns:
        Complete analysis result dictionary
    """
    
    pipeline = VeinDetectionPipeline(
        enable_llm=enable_llm,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        pixels_per_mm=1.0,
        target_fps=5
    )
    
    result = pipeline.process_video(
        video_path,
        max_frames=max_frames,
        save_visualizations=True,
        output_dir=output_dir
    )
    
    return result
