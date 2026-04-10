"""
Hybrid Fascia/Vein Detector: Smart Preprocessing + Blob Detection
Uses adaptive thresholding and morphological analysis for robust detection
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class PretrainedSegmentationDetector:
    """Fast and reliable detector using advanced OpenCV techniques"""
    
    def __init__(self):
        """Initialize detector (no model downloads)"""
        logger.info("[Detector] Initializing adaptive segmentation detector...")
        # Import EdgeFasciaDetector for fascia detection
        from .edge_fascia_detector import EdgeFasciaDetector
        self.fascia_detector = EdgeFasciaDetector()
        logger.info("[Detector] ✓ Detector ready (using EdgeFasciaDetector for fascia)")
    
    def segment_fascia(self, image: np.ndarray) -> np.ndarray:
        """
        Detect fascia using EdgeFasciaDetector.
        Returns the fascia mask showing the region between upper and lower boundaries.
        
        Args:
            image: Input ultrasound image (H, W) or (H, W, 3)
        
        Returns:
            Binary fascia mask showing the fascia band region
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        logger.info(f"[Fascia] Detecting fascia band in {h}x{w} image using EdgeFasciaDetector...")
        
        # Use edge-based detector
        result = self.fascia_detector.detect(gray)
        
        # Create mask from boundaries
        fascia_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result['status'] == 'success' and result['upper_edge'] is not None and result['lower_edge'] is not None:
            upper = result['upper_edge'].astype(np.int32)
            lower = result['lower_edge'].astype(np.int32)
            
            # Draw the fascia region as filled area between upper and lower boundaries
            # Create a mask by filling the region
            pts_upper = upper.reshape((-1, 1, 2))
            pts_lower = lower.reshape((-1, 1, 2))
            
            # Combine points: upper boundary reversed + lower boundary
            points = np.vstack([pts_upper, pts_lower[::-1]])
            cv2.fillPoly(fascia_mask, [points], 255)
            
            upper_y = upper[:, 1].mean()
            lower_y = lower[:, 1].mean()
            logger.info(f"[Fascia] ✓ Found fascia band from y={upper_y:.0f} to y={lower_y:.0f}, span={lower_y-upper_y:.0f}px, confidence={result['confidence']:.2f}")
        else:
            logger.info(f"[Fascia] Could not detect fascia boundaries: {result['status']}")
        
        return fascia_mask
    
    def segment_veins(self, image: np.ndarray, fascia_mask: np.ndarray = None, num_masks: int = 5) -> List[Dict]:
        """
        Detect veins using adaptive thresholding + morphological analysis
        
        Veins are dark circular blobs in ultrasound.
        Strategy:
        1. Enhance contrast with CLAHE
        2. Use adaptive thresholding to find dark regions
        3. Filter by size, shape, and position
        4. Return top 2 by quality score
        
        Args:
            image: Input ultrasound image
            fascia_mask: Optional fascia mask for spatial filtering
            num_masks: Max veins to detect
        
        Returns:
            List of detected vein masks
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            logger.info(f"[Veins] Detecting veins from {h}x{w}...")
            
            # Normalize
            gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Invert (veins are dark)
            inverted = 255 - gray_norm
            
            # Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
            enhanced = clahe.apply(inverted)
            
            # Adaptive thresholding - better for images with varying illumination
            # Block size: odd number, medium for balance between detail and noise
            adaptive_th = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=19,  # Medium window for balance
                C=7  # Moderate constant
            )
            
            logger.info(f"[Veins] Adaptive threshold applied, dark pixels: {np.count_nonzero(adaptive_th)}")
            
            # Morphological cleanup
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            
            # Open: remove small noise
            cleaned = cv2.morphologyEx(adaptive_th, cv2.MORPH_OPEN, kernel_small)
            # Close: fill small holes in blobs
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
            
            logger.info(f"[Veins] After morphology, pixels: {np.count_nonzero(cleaned)}")
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cleaned, connectivity=8
            )
            
            logger.info(f"[Veins] Found {num_labels - 1} candidate components")
            
            candidates = []
            
            # Evaluate each component
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                x = stats[label_id, cv2.CC_STAT_LEFT]
                y = stats[label_id, cv2.CC_STAT_TOP]
                w_bbox = stats[label_id, cv2.CC_STAT_WIDTH]
                h_bbox = stats[label_id, cv2.CC_STAT_HEIGHT]
                
                # Size filter: realistic vein size
                if area < 380 or area > 5200:
                    logger.debug(f"[Veins] Skip {label_id}: area {area} out of range [380, 5200]")
                    continue
                
                # Get contour for shape analysis
                mask = (labels == label_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter < 10:  # Degenerate contour
                    continue
                
                # Shape metrics
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area == 0:
                    continue
                
                solidity = area / hull_area
                circularity = 4 * np.pi * area / (perimeter**2)
                
                # Aspect ratio (0 = symmetric circle, large = elongated line)
                aspect = max(w_bbox, h_bbox) / (min(w_bbox, h_bbox) + 1e-6)
                
                # Apply balanced shape filters
                if solidity < 0.54:
                    logger.debug(f"[Veins] Skip {label_id}: low solidity {solidity:.2f}")
                    continue
                
                if circularity < 0.32:
                    logger.debug(f"[Veins] Skip {label_id}: low circularity {circularity:.2f}")
                    continue
                
                if aspect > 7.0:
                    logger.debug(f"[Veins] Skip {label_id}: too elongated (aspect {aspect:.1f})")
                    continue
                
                # Spatial filter: prefer regions below fascia (if provided)
                centroid_y = centroids[label_id][1]
                
                quality_score = solidity * circularity  # Combined shape quality
                confidence = min(0.95, max(0.6, solidity * 0.85 + 0.15))
                
                logger.debug(f"[Veins] Candidate {label_id}: area={area}, sol={solidity:.2f}, circ={circularity:.2f}, aspect={aspect:.1f}, quality={quality_score:.4f}")
                
                candidates.append({
                    "label_id": label_id,
                    "mask": mask,
                    "confidence": float(confidence),
                    "quality": quality_score,
                    "area": int(area),
                    "solidity": float(solidity),
                    "circularity": float(circularity),
                    "aspect": float(aspect),
                    "centroid": tuple(centroids[label_id]),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w_bbox),
                    "h": int(h_bbox)
                })
            
            logger.info(f"[Veins] Evaluated {len(candidates)} candidates")
            
            # Select top 2 by quality
            candidates.sort(key=lambda x: x["quality"], reverse=True)
            vein_masks = []
            
            for idx, candidate in enumerate(candidates[:2]):
                candidate["vein_id"] = f"V{idx + 1}"
                logger.info(f"[Veins] ✓ {candidate['vein_id']}: area={candidate['area']}, solidity={candidate['solidity']:.2f}, circ={candidate['circularity']:.2f}, quality={candidate['quality']:.4f}")
                
                vein_masks.append({
                    "mask": candidate["mask"],
                    "confidence": candidate["confidence"],
                    "vein_id": candidate["vein_id"],
                    "properties": {
                        "area": candidate["area"],
                        "solidity": candidate["solidity"],
                        "circularity": candidate["circularity"],
                        "centroid": candidate["centroid"],
                        "aspect_ratio": candidate["aspect"]
                    }
                })
            
            logger.info(f"[Veins] ✓ Detected {len(vein_masks)} veins")
            return vein_masks
        
        except Exception as e:
            logger.error(f"[Veins] Detection error: {e}", exc_info=True)
            raise

# Alias for backward compatibility
UNetVeinDetector = PretrainedSegmentationDetector
