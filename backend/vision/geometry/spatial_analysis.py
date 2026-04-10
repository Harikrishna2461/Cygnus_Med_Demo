"""
Spatial Analysis Module for Vein-Fascia Relationships

Computes geometric relationships between detected veins and fascia.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class SpatialAnalyzer:
    """Analyzes spatial relationships between veins and fascia"""
    
    def __init__(self, depth_threshold_mm: float = 5.0, pixels_per_mm: float = 1.0):
        """
        Initialize spatial analyzer
        
        Args:
            depth_threshold_mm: Threshold for depth classification (mm)
            pixels_per_mm: Conversion factor from pixels to millimeters
        """
        self.depth_threshold_mm = depth_threshold_mm
        self.pixels_per_mm = pixels_per_mm
    
    def analyze_vein_position(
        self,
        vein_mask: np.ndarray,
        fascia_mask: np.ndarray
    ) -> Dict:
        """
        Analyze position of vein relative to fascia
        
        Args:
            vein_mask: Binary mask of vein
            fascia_mask: Binary mask of fascia
        
        Returns:
            Dictionary with spatial analysis results
        """
        vein_centroid = self._compute_centroid(vein_mask)
        
        # Distance to fascia
        distance_pixels = self._compute_distance_to_fascia(vein_mask, fascia_mask)
        distance_mm = distance_pixels / self.pixels_per_mm
        
        # Check intersection
        intersects = self._check_intersection(vein_mask, fascia_mask)
        
        # Relative position
        position = self._determine_relative_position(vein_mask, fascia_mask)
        
        # Compute depth approximation
        depth_info = self._compute_depth_info(vein_centroid, fascia_mask)
        
        return {
            "vein_centroid": vein_centroid,
            "distance_to_fascia_mm": float(distance_mm),
            "distance_to_fascia_px": float(distance_pixels),
            "intersects_fascia": bool(intersects),
            "intersection_length_px": float(self._compute_intersection_length(vein_mask, fascia_mask)),
            "relative_position": position,
            "depth_info": depth_info
        }
    
    def _compute_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of binary mask"""
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return (0.0, 0.0)
        
        cx = float(np.mean(x_coords))
        cy = float(np.mean(y_coords))
        
        return (cx, cy)
    
    def _compute_distance_to_fascia(
        self,
        vein_mask: np.ndarray,
        fascia_mask: np.ndarray
    ) -> float:
        """
        Compute minimum distance from vein to fascia
        
        Uses distance transform for efficiency
        """
        # Create distance map from fascia
        fascia_inv = 1 - fascia_mask.astype(np.uint8)
        dist_transform = cv2.distanceTransform(fascia_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # Get minimum distance where vein exists
        vein_pixels = vein_mask > 0
        
        if not np.any(vein_pixels):
            return 0.0
        
        distances = dist_transform[vein_pixels]
        min_distance = float(np.min(distances))
        
        return min_distance
    
    def _check_intersection(self, vein_mask: np.ndarray, fascia_mask: np.ndarray) -> bool:
        """Check if vein intersects with fascia"""
        intersection = np.logical_and(vein_mask > 0, fascia_mask > 0)
        return bool(np.any(intersection))
    
    def _compute_intersection_length(self, vein_mask: np.ndarray, fascia_mask: np.ndarray) -> float:
        """Compute length of intersection between vein and fascia"""
        intersection = np.logical_and(vein_mask > 0, fascia_mask > 0).astype(np.uint8)
        
        # Count connected pixels as proxy for length
        return float(np.sum(intersection))
    
    def _determine_relative_position(self, vein_mask: np.ndarray, fascia_mask: np.ndarray) -> str:
        """
        Determine if vein is above, below, or crossing fascia
        
        Strategy:
        - If intersects: crossing
        - If centroid above fascia: above
        - If centroid below fascia: below
        """
        if self._check_intersection(vein_mask, fascia_mask):
            return "crossing"
        
        vein_centroid = self._compute_centroid(vein_mask)
        cx, cy = int(vein_centroid[0]), int(vein_centroid[1])
        
        # Check if centroid is within bounds
        if cy >= fascia_mask.shape[0] or cx >= fascia_mask.shape[1]:
            return "unknown"
        
        # Determine position based on Y coordinate relative to fascia
        # In ultrasound, top = surface, bottom = deep
        fascia_y_coords = np.where(fascia_mask > 0)[0]
        
        if len(fascia_y_coords) == 0:
            return "unknown"
        
        fascia_center_y = float(np.mean(fascia_y_coords))
        
        if cy < fascia_center_y:
            return "above"
        else:
            return "below"
    
    def _compute_depth_info(self, vein_centroid: Tuple[float, float], fascia_mask: np.ndarray) -> Dict:
        """
        Compute depth-related information
        
        In ultrasound: Y=0 is skin surface, Y=max is deepest
        """
        cx, cy = vein_centroid
        
        # Distance from skin (top of image)
        distance_from_skin_px = float(cy)
        distance_from_skin_mm = distance_from_skin_px / self.pixels_per_mm
        
        # Distance from fascia
        fascia_y_coords = np.where(fascia_mask > 0)[0]
        if len(fascia_y_coords) > 0:
            fascia_center_y = float(np.mean(fascia_y_coords))
            distance_from_fascia_px = float(abs(cy - fascia_center_y))
            distance_from_fascia_mm = distance_from_fascia_px / self.pixels_per_mm
        else:
            distance_from_fascia_px = 0.0
            distance_from_fascia_mm = 0.0
        
        return {
            "distance_from_skin_mm": distance_from_skin_mm,
            "distance_from_skin_px": distance_from_skin_px,
            "distance_from_fascia_mm": distance_from_fascia_mm,
            "distance_from_fascia_px": distance_from_fascia_px
        }
    
    def batch_analyze_veins(
        self,
        vein_list: List[Dict],
        fascia_mask: np.ndarray
    ) -> List[Dict]:
        """
        Analyze multiple veins at once
        
        Args:
            vein_list: List of vein dictionaries with 'mask' key
            fascia_mask: Binary mask of fascia
        
        Returns:
            List of analyzed vein dictionaries (original + spatial analysis)
        """
        analyzed_veins = []
        
        for vein in vein_list:
            vein_mask = vein.get('mask')
            if vein_mask is None:
                continue
            
            analysis = self.analyze_vein_position(vein_mask, fascia_mask)
            
            # Merge with original vein data
            vein_analyzed = {
                **vein,
                "spatial_analysis": analysis
            }
            analyzed_veins.append(vein_analyzed)
        
        return analyzed_veins
