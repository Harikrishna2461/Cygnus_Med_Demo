"""
Vein classifier: integrates fascia detection with blob tracking.
Classifies each vein/blob as GSV (N2), Superficial (N3), or Deep (N1).
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class VeinClassification:
    """Classification result for a single vein."""
    blob_id: int
    vein_type: str  # 'N1_deep', 'N2_gsv', 'N3_superficial'
    vein_label: str  # 'Deep Vein', 'GSV', 'Superficial Vein'
    confidence: float  # 0-100
    distance_to_fascia: float  # pixels (negative = above, positive = below)
    position: str  # 'above', 'within', 'below'
    center: Tuple[float, float]  # (x, y)
    radius: float


class VeinClassifier:
    """
    Classify veins relative to fascia position.
    """
    
    # Distance thresholds (in pixels)
    FASCIA_MARGIN = 10  # pixels above/below fascia considered "within"
    
    # Vein types with codes
    VEIN_TYPES = {
        'N1_deep': ('Deep Vein', (0, 255, 0)),  # Green
        'N2_gsv': ('GSV', (255, 0, 255)),        # Magenta
        'N3_superficial': ('Superficial Vein', (0, 165, 255))  # Orange
    }
    
    def __init__(self):
        self.classifications: Dict[int, VeinClassification] = {}
    
    def classify_blobs(self, blobs: Dict, fascia_data: Dict) -> Dict[int, VeinClassification]:
        """
        Classify all detected blobs relative to fascia.
        
        Args:
            blobs: dict of {blob_id: blob_state} from blob_detector
            fascia_data: dict with keys 'mask', 'boundary', 'center', 'confidence' from fascia_detector
        
        Returns:
            dict of {blob_id: VeinClassification}
        """
        classifications = {}
        
        # Extract fascia position
        fascia_y = self._get_fascia_y(fascia_data)
        
        if fascia_y is None:
            # No fascia detected, classify by heuristics
            for blob_id, blob_state in blobs.items():
                classifications[blob_id] = self._classify_without_fascia(blob_id, blob_state)
        else:
            # Classify relative to fascia
            for blob_id, blob_state in blobs.items():
                classifications[blob_id] = self._classify_with_fascia(
                    blob_id, blob_state, fascia_y, fascia_data
                )
        
        self.classifications = classifications
        return classifications
    
    def _get_fascia_y(self, fascia_data: Dict) -> Optional[float]:
        """Extract y-coordinate of fascia boundary."""
        if not fascia_data:
            return None
        
        # Try to get center y-coordinate
        if 'center' in fascia_data and fascia_data['center']:
            return fascia_data['center'][1]
        
        # Try to estimate from boundary
        if 'boundary' in fascia_data and fascia_data['boundary']:
            boundary = fascia_data['boundary']
            ys = [p[1] for p in boundary]
            return np.median(ys)
        
        # Try from mask
        if 'mask' in fascia_data:
            mask = fascia_data['mask']
            rows_with_mask = np.where(mask.sum(axis=1) > 0)[0]
            if len(rows_with_mask) > 0:
                return float(np.mean(rows_with_mask))
        
        return None
    
    def _classify_with_fascia(
        self,
        blob_id: int,
        blob_state,
        fascia_y: float,
        fascia_data: Dict
    ) -> VeinClassification:
        """Classify blob relative to fascia position."""
        
        # Get blob position
        blob_center = blob_state.center
        if blob_center is None:
            blob_center = (0, 0)
        
        blob_y = blob_center[1]
        distance = blob_y - fascia_y  # negative = above fascia, positive = below
        
        # Determine position relative to fascia
        if abs(distance) <= self.FASCIA_MARGIN:
            # Within or very near fascia
            vein_type = 'N2_gsv'
            position = 'within'
            confidence = 75.0
        elif distance < 0:
            # Above fascia
            vein_type = 'N3_superficial'
            position = 'above'
            confidence = 80.0
        else:
            # Below fascia
            vein_type = 'N1_deep'
            position = 'below'
            confidence = 80.0
        
        # Boost confidence based on fascia detection quality
        if 'confidence' in fascia_data:
            fascia_confidence = fascia_data['confidence']
            confidence *= fascia_confidence
        
        vein_label = self.VEIN_TYPES[vein_type][0]
        
        return VeinClassification(
            blob_id=blob_id,
            vein_type=vein_type,
            vein_label=vein_label,
            confidence=min(100.0, confidence),
            distance_to_fascia=distance,
            position=position,
            center=blob_center,
            radius=blob_state.radius
        )
    
    def _classify_without_fascia(self, blob_id: int, blob_state) -> VeinClassification:
        """
        Classify blob without fascia info using heuristics.
        Based on size, position, etc.
        """
        
        # Heuristic: larger blobs are more likely deep veins
        radius = blob_state.radius if blob_state.radius else 0
        
        if radius > 30:
            vein_type = 'N1_deep'
            confidence = 50.0
        elif radius > 15:
            vein_type = 'N2_gsv'
            confidence = 50.0
        else:
            vein_type = 'N3_superficial'
            confidence = 50.0
        
        vein_label = self.VEIN_TYPES[vein_type][0]
        blob_center = blob_state.center if blob_state.center else (0, 0)
        
        return VeinClassification(
            blob_id=blob_id,
            vein_type=vein_type,
            vein_label=vein_label,
            confidence=confidence,
            distance_to_fascia=None,
            position='unknown',
            center=blob_center,
            radius=radius
        )
    
    def get_classification(self, blob_id: int) -> Optional[VeinClassification]:
        """Get classification for a specific blob."""
        return self.classifications.get(blob_id)
    
    def get_all_classifications(self) -> Dict[int, VeinClassification]:
        """Get all classifications."""
        return self.classifications
    
    def get_veins_by_type(self, vein_type: str) -> List[VeinClassification]:
        """Get all veins of a specific type."""
        return [c for c in self.classifications.values() if c.vein_type == vein_type]
    
    def visualize_classifications(
        self,
        image: np.ndarray,
        fascia_data: Optional[Dict] = None,
        show_labels: bool = True,
        text_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Draw classifications on image.
        
        Args:
            image: input image (H, W, 3)
            fascia_data: fascia detection result
            show_labels: whether to show vein type labels
            text_color: color of text labels
        
        Returns:
            image with visualizations
        """
        output = image.copy()
        
        # Draw fascia boundary if available
        if fascia_data and 'boundary' in fascia_data:
            boundary = fascia_data['boundary']
            if boundary:
                pts = np.array(boundary, dtype=np.int32)
                cv2.polylines(output, [pts], False, (100, 100, 100), 2)
        
        # Draw fascia line
        if fascia_data and 'center' in fascia_data:
            center = fascia_data['center']
            if center:
                cy = int(center[1])
                cv2.line(output, (0, cy), (output.shape[1], cy), (100, 100, 100), 1)
                cv2.putText(output, 'Fascia', (10, cy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Draw each vein
        for blob_id, classification in self.classifications.items():
            cx, cy = classification.center
            cx, cy = int(cx), int(cy)
            radius = int(classification.radius)
            
            # Get vein color
            vein_color = self.VEIN_TYPES[classification.vein_type][1]
            
            # Draw circle
            cv2.circle(output, (cx, cy), radius, vein_color, 2)
            cv2.circle(output, (cx, cy), 3, vein_color, -1)
            
            # Draw label
            if show_labels:
                label = f"{classification.vein_label} (ID:{blob_id})"
                cv2.putText(output, label, (cx + radius + 5, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                
                # Confidence
                conf_text = f"Conf: {classification.confidence:.1f}%"
                cv2.putText(output, conf_text, (cx + radius + 5, cy + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        
        return output
    
    def to_json(self) -> List[Dict]:
        """
        Convert classifications to JSON-serializable format.
        
        Returns:
            list of dicts with vein classification data
        """
        result = []
        for blob_id, classification in self.classifications.items():
            result.append({
                'blob_id': blob_id,
                'vein_type': classification.vein_type,
                'vein_label': classification.vein_label,
                'confidence': round(classification.confidence, 2),
                'position': classification.position,
                'distance_to_fascia': round(classification.distance_to_fascia, 2) if classification.distance_to_fascia is not None else None,
                'center': [round(classification.center[0], 1), round(classification.center[1], 1)],
                'radius': round(classification.radius, 1)
            })
        return result
    
    def get_summary(self) -> Dict:
        """Get summary of all veins."""
        summary = {
            'total_veins': len(self.classifications),
            'deep_veins': len(self.get_veins_by_type('N1_deep')),
            'gsv': len(self.get_veins_by_type('N2_gsv')),
            'superficial_veins': len(self.get_veins_by_type('N3_superficial')),
            'veins': self.to_json()
        }
        
        # Add position statistics
        positions = [c.position for c in self.classifications.values()]
        summary['positions'] = {
            'above': positions.count('above'),
            'within': positions.count('within'),
            'below': positions.count('below'),
            'unknown': positions.count('unknown')
        }
        
        return summary
