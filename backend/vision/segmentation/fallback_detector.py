"""
Fallback Vein Detector - Classical CV approach when SAM unavailable

Uses morphological operations + edge detection for vein segmentation.
No neural networks = no memory issues, instant processing.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class FallbackVeinDetector:
    """Classical CV-based vein detection as fallback when SAM unavailable"""
    
    def __init__(self):
        """Initialize fallback detector"""
        logger.info("✓ Fallback vein detector initialized (classical CV)")
    
    def segment_fascia(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Find fascia as the BRIGHTEST continuous HORIZONTAL line across the image.
        
        Fascia appears as a bright white/yellow horizontal band. We find it by:
        1. Looking for VERY bright pixels (top 5% brightness)
        2. Finding horizontal continuity (morphological close)
        3. Locating the row with maximum horizontal pixel density
        
        Args:
            image: Input ultrasound image (BGR or RGB)
        
        Returns:
            Binary mask with fascia boundary marked
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        logger.info(f"[FASCIA] Detecting fascia as brightest horizontal line")
        
        # Step 1: Find VERY BRIGHT pixels only (top 5% brightness)
        bright_th = np.percentile(gray, 95)  # Top 5% brightness only
        _, bright = cv2.threshold(gray, int(bright_th), 255, cv2.THRESH_BINARY)
        
        # Step 2: Enhance horizontal continuity with wide kernel
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))  # Very wide, thin
        h_lines = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, h_kernel)
        
        # Step 3: Find the row with maximum horizontal pixel density
        row_brightness = np.sum(h_lines > 0, axis=1)
        
        if np.max(row_brightness) < 20:
            logger.warning("[FASCIA] Weak signal, using default midline")
            fascia_y = h // 2
        else:
            # Smooth to find the peak of the horizontal band
            smooth_kernel = np.ones(7) / 7  # Slightly wider smoothing
            smoothed = np.convolve(row_brightness.astype(float), smooth_kernel, mode='same')
            fascia_y = np.argmax(smoothed)
            logger.info(f"[FASCIA] Brightest horizontal line at row {fascia_y}/{h} (density: {row_brightness[fascia_y]:.0f} pixels)")
        
        # Step 4: Create mask - everything above fascia
        fascia_mask = np.zeros((h, w), dtype=np.uint8)
        fascia_mask[0:fascia_y, :] = 255
        
        logger.info(f"[FASCIA] Fascia separator at row {fascia_y}/{h}")
        return fascia_mask.astype(np.uint8)
    
    def segment_veins(self, image: np.ndarray, fascia_mask: np.ndarray = None, num_masks: int = 5) -> List[Dict]:
        """
        FIXED: Find ONLY the darkest vein blobs - NOT large regions.
        
        Key: Use STRICT darkness threshold to isolate actual vein circles only.
        
        Args:
            image: Input ultrasound image
            fascia_mask: Optional fascia mask
            num_masks: Max veins to detect
        
        Returns:
            List of detected vein masks
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        logger.info(f"[VEINS] Detecting veins from {gray.shape}")
        
        # Normalize
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Step 1: Invert and enhance
        inverted = 255 - gray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inverted)
        
        # Step 2: Find only SIGNIFICANTLY dark pixels - balanced threshold
        dark_th = np.percentile(enhanced, 89)  # Top 11% darkest pixels (balanced)
        _, dark_only = cv2.threshold(enhanced, int(dark_th), 255, cv2.THRESH_BINARY)
        
        logger.info(f"[VEINS] Dark threshold: {int(dark_th)}, pixels: {np.count_nonzero(dark_only)}")
        
        # Step 3: Remove small noise and aggressively clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(dark_only, cv2.MORPH_OPEN, kernel)  # Remove noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)   # Fill holes
        
        # Step 4: Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned.astype(np.uint8), connectivity=8
        )
        
        logger.info(f"[VEINS] Found {num_labels - 1} candidate blobs")
        
        vein_masks = []
        candidates = []  # Track all candidates with quality scores
        
        # Step 5: Size filtering - accept blobs in realistic vein size range
        blob_areas = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            blob_areas.append((label_id, area))
        
        # Sort by area (largest first)
        blob_areas.sort(key=lambda x: x[1], reverse=True)
        
        min_area = 480    # Balanced minimum
        max_area = 5000   # Balanced maximum
        
        logger.info(f"[VEINS] Size filter: {min_area}-{max_area} pixels")
        
        for label_id, area in blob_areas:
            # Size check
            if area < min_area or area > max_area:
                logger.debug(f"[VEINS] Reject {label_id}: area {area} out of [{min_area}, {max_area}]")
                continue
            
            # Get mask
            mask = (labels == label_id).astype(np.uint8) * 255
            
            # Get contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = contours[0]
            perimeter = cv2.arcLength(contour, True)
            
            # Solidity: must be reasonably compact
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6) if hull_area > 0 else 0
            
            # Balanced solidity check
            if solidity < 0.55:
                logger.debug(f"[VEINS] Reject {label_id}: solidity {solidity:.2f} < 0.55 (not compact enough)")
                continue
            
            # Circularity: moderate constraint
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-6) if perimeter > 0 else 0
            if circularity < 0.35:
                logger.debug(f"[VEINS] Reject {label_id}: circularity {circularity:.2f} < 0.35 (too elongated/linear)")
                continue
            
            # Aspect ratio check - reject line-like structures
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > 7.0:
                logger.debug(f"[VEINS] Reject {label_id}: aspect ratio {aspect:.1f} too line-like")
                continue
            
            # Confidence and quality score
            confidence = min(0.95, max(0.6, solidity * 0.9 + 0.1))
            quality_score = solidity + circularity  # Higher is better
            
            logger.debug(f"[VEINS] Candidate {label_id}: area={area}, solidity={solidity:.2f}, circ={circularity:.2f}, quality={quality_score:.2f}")
            
            candidates.append({
                "label_id": label_id,
                "mask": mask,
                "confidence": float(confidence),
                "quality_score": quality_score,
                "properties": {
                    "area": int(area),
                    "solidity": float(solidity),
                    "circularity": float(circularity),
                    "aspect_ratio": float(aspect),
                    "centroid": tuple(centroids[label_id])
                }
            })
        
        # Step 6: Select top 2 best candidates by quality score
        candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        for idx, candidate in enumerate(candidates[:2]):  # Take top 2
            candidate["vein_id"] = f"V{idx + 1}"
            logger.info(f"[VEINS] ✓ Accept {candidate['vein_id']}: area={candidate['properties']['area']}, solidity={candidate['properties']['solidity']:.2f}, circ={candidate['properties']['circularity']:.2f}")
            vein_masks.append({
                "mask": mask,
                "confidence": float(confidence),
                "vein_id": f"V{len(vein_masks) + 1}",
                "properties": {
                    "area": int(area),
                    "centroid": tuple(centroids[label_id]),
                    "perimeter": float(perimeter) if perimeter > 0 else 0,
                    "num_points": len(contour),
                    "solidity": float(solidity),
                    "circularity": float(circularity)
                }
            })
            
            if len(vein_masks) >= num_masks:
                logger.info(f"[VEINS] Reached max: {num_masks}")
                break
        
        logger.info(f"[VEINS] Final: {len(vein_masks)} veins detected")
        return vein_masks
    
    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of binary mask"""
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return (0, 0)
        
        cx = float(np.mean(x_coords))
        cy = float(np.mean(y_coords))
        
        return (cx, cy)
