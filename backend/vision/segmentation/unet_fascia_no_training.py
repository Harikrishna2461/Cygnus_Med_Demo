"""
Fascia detector that works without training.
Uses rule-based image processing for ultrasound analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class FasciaDetector:
    """
    Fascia detection wrapper that uses rule-based approach.
    No machine learning model needed - works on image processing principles.
    """
    
    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize fascia detector.
        
        Args:
            model_path: ignored (not needed for rule-based)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_path = model_path
        self.ready = True
        
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect fascia boundaries in ultrasound image.
        
        Args:
            image: ultrasound image (H, W, 3) with values 0-255 or 0-1
        
        Returns:
            binary fascia mask (H, W) with values 0-255
        """
        # Ensure proper format
        if len(image.shape) == 3:
            if image.dtype == np.float32 or image.dtype == np.float64:
                img = (image * 255).astype(np.uint8)
            else:
                img = image.astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8) if image.max() > 1 else (image * 255).astype(np.uint8)
        
        H, W = gray.shape
        
        # Step 1: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Detect edges (fascia appears as bright edges)
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Step 3: Detect linear structures (fascia = lines)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20))
        
        lines_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        lines_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        lines = cv2.bitwise_or(lines_h, lines_v)
        
        # Step 4: Filter for fascia (bright boundaries)
        _, intensity_mask = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_and(intensity_mask, lines)
        
        # Step 5: Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to make fascia boundary visible
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def detect_batch(self, images: list) -> list:
        """
        Detect fascia in multiple images.
        
        Args:
            images: list of images
        
        Returns:
            list of fascia masks
        """
        masks = []
        for img in images:
            mask = self.detect(img)
            masks.append(mask)
        return masks
