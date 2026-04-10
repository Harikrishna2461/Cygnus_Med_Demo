"""
Frame Extraction Module for Ultrasound Video Processing

Extracts frames from ultrasound video files at configurable sampling rates.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extracts frames from ultrasound video files"""
    
    def __init__(self, target_fps: int = 5, resize_shape: Optional[Tuple[int, int]] = None):
        """
        Initialize frame extractor
        
        Args:
            target_fps: Target frames per second to extract (default: 5)
            resize_shape: Optional tuple (height, width) to resize frames
        """
        self.target_fps = target_fps
        self.resize_shape = resize_shape
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None for all)
        
        Returns:
            List of frame numpy arrays (BGR format)
        
        Raises:
            ValueError: If video cannot be opened or no frames extracted
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate sampling interval
            sample_interval = max(1, int(fps / self.target_fps))
            
            logger.info(f"Video info: {total_frames} frames @ {fps:.2f} fps")
            logger.info(f"Sampling interval: {sample_interval} (target: {self.target_fps} fps)")
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and (max_frames is None or extracted_count < max_frames):
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Sample at target FPS
                if frame_count % sample_interval == 0:
                    if self.resize_shape:
                        frame = cv2.resize(frame, self.resize_shape[::-1])  # OpenCV uses (width, height)
                    
                    frames.append(frame)
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:
                        logger.debug(f"Extracted {extracted_count} frames")
                
                frame_count += 1
            
            logger.info(f"✓ Extracted {len(frames)} frames from video")
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            return frames
        
        finally:
            cap.release()
    
    def extract_frame_at_time(self, video_path: str, timestamp_ms: float) -> np.ndarray:
        """
        Extract single frame at specific timestamp
        
        Args:
            video_path: Path to video file
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            Frame as numpy array (BGR format)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Cannot read frame at {timestamp_ms}ms")
            
            if self.resize_shape:
                frame = cv2.resize(frame, self.resize_shape[::-1])
            
            return frame
        
        finally:
            cap.release()
