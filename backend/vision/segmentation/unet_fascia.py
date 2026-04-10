"""
Fascia segmentation using Edge-based Boundary Detection.
Finds THIN curved fascia boundaries (1-2 pixels), not thick bands.
Works without training data.
"""

import numpy as np
import cv2
from pathlib import Path
import warnings
from typing import Dict, Any


class FasciaDetector:
    """
    Fascia detection using edge-based boundary finding.
    Optimized for real ultrasound with thin delicate fascia.
    """
    
    def __init__(self, model_path=None, device='cpu'):
        """Initialize fascia detector."""
        self.device = device
        self.model_path = model_path
        self.ready = True
        # Import here to avoid circular imports
        from .edge_fascia_detector import EdgeFasciaDetector
        self.detector = EdgeFasciaDetector()
        
    def load_model(self, model_path):
        """Fake load for compatibility."""
        pass
    
    def detect(self, image: np.ndarray, threshold: float = 0.5, 
               return_boundary: bool = False) -> Dict[str, Any]:
        """
        Detect fascia as thin edge boundaries.
        
        Args:
            image: input image (H, W, 3) or (H, W), uint8 or float
            threshold: ignored (for API compatibility)
            return_boundary: if True, include boundary coordinates
        
        Returns:
            dict with 'mask', 'upper_edge', 'lower_edge', 'confidence', 'center', 'boundary'
        """
        # Normalize image to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                img = (image * 255).astype(np.uint8)
            else:
                img = image.astype(np.uint8)
        else:
            img = image.astype(np.uint8)
        
        # Use edge-based detector for thin boundaries
        result = self.detector.detect(img)
        
        # Add compatibility fields for API
        if result['upper_edge'] is not None and result['lower_edge'] is not None:
            # Calculate center point
            upper_y = result['upper_edge'][:, 1].mean()
            lower_y = result['lower_edge'][:, 1].mean()
            center_y = (upper_y + lower_y) / 2
            center_x = result['upper_edge'].shape[0] // 2
            result['center'] = [center_x, center_y]
            
            # Include boundary if requested
            if return_boundary:
                boundary = result['lower_edge'].tolist() if result['lower_edge'] is not None else []
                result['boundary'] = boundary
        else:
            result['center'] = None
            result['boundary'] = []
        
        return result
    
    def get_mask(self, x=None, threshold=0.5):
        """
        Get binary mask (for compatibility with UNet interface).
        
        Args:
            x: input (optional, not used for fascia detection)
            threshold: ignored
        
        Returns:
            binary mask (H, W)
        """
        if self.detector.fascia_mask is not None:
            return self.detector.fascia_mask  # Return (H, W) uint8 mask
        return np.zeros((256, 256), dtype=np.uint8)
