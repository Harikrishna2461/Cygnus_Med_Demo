"""
Rule-Based Classification Module for Vein Types

Classifies veins into categories using heuristic rules based on spatial relationships.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VeinClassifier:
    """Rule-based classification of vein types"""
    
    # Classification thresholds
    PERFORATOR_INTERSECTION_THRESHOLD = 10  # pixels
    DEEP_VEIN_DEPTH_THRESHOLD_MM = 10.0     # below fascia
    SUPERFICIAL_DEPTH_THRESHOLD_MM = 5.0    # above fascia
    N1_THRESHOLD_MM = 10.0                   # Very deep (deep veins)
    N2_THRESHOLD_MM = 5.0                    # Mid depth (GSVs)
    N3_THRESHOLD_MM = 2.0                    # Superficial (near skin)
    
    def __init__(self):
        """Initialize vein classifier"""
        self.classification_rules = self._build_rules()
    
    def classify_vein(self, vein_data: Dict) -> Dict:
        """
        Classify a single vein based on spatial analysis
        
        Args:
            vein_data: Dictionary with 'spatial_analysis' key
        
        Returns:
            Dictionary with classification results
        """
        spatial = vein_data.get('spatial_analysis', {})
        
        # Extract key features
        intersects = spatial.get('intersects_fascia', False)
        position = spatial.get('relative_position', 'unknown')
        distance_to_fascia = spatial.get('distance_to_fascia_mm', 0)
        distance_from_skin = spatial.get('depth_info', {}).get('distance_from_skin_mm', 0)
        
        # Rule-based classification
        vein_type = self._apply_classification_rules(
            intersects=intersects,
            position=position,
            distance_to_fascia=distance_to_fascia
        )
        
        # N-level (depth) classification
        n_level = self._compute_n_level(distance_from_skin)
        
        # Confidence (0-1)
        confidence = self._compute_confidence(vein_data, vein_type)
        
        return {
            "primary_classification": vein_type,
            "n_level": n_level,
            "confidence": float(confidence),
            "reasoning": self._generate_reasoning(spatial, vein_type, n_level),
            "requires_llm_confirmation": self._requires_llm_confirmation(vein_type, confidence)
        }
    
    def _apply_classification_rules(
        self,
        intersects: bool,
        position: str,
        distance_to_fascia: float
    ) -> str:
        """
        Apply rule-based classification
        
        Rules:
        1. If intersects fascia → Perforator vein
        2. If crossing fascia → Perforator vein
        3. If below fascia (deep) → Deep vein
        4. If above fascia (shallow) → Superficial vein
        5. Unknown/ambiguous → Requires LLM
        """
        
        # Rule 1: Intersection indicates perforator
        if intersects or position == "crossing":
            return "perforator_vein"
        
        # Rule 2: Below fascia → Deep vein
        if position == "below":
            return "deep_vein"
        
        # Rule 3: Above fascia → Superficial vein
        if position == "above":
            return "superficial_vein"
        
        # Default: Unknown (will need LLM)
        return "unknown"
    
    def _compute_n_level(self, distance_from_skin_mm: float) -> str:
        """
        Compute N-level classification based on depth from skin
        
        N1: Deep veins (> 10mm from skin)
        N2: Mid-depth (5-10mm from skin) - typically GSVs
        N3: Superficial (< 5mm from skin, above fascia)
        """
        if distance_from_skin_mm >= self.N1_THRESHOLD_MM:
            return "N1"
        elif distance_from_skin_mm >= self.N2_THRESHOLD_MM:
            return "N2"
        else:
            return "N3"
    
    def _compute_confidence(self, vein_data: Dict, vein_type: str) -> float:
        """
        Compute classification confidence
        
        Based on:
        - Segmentation confidence from SAM
        - Clarity of spatial relationships
        - Consistency with heuristics
        """
        base_confidence = vein_data.get('confidence', 0.5)
        
        spatial = vein_data.get('spatial_analysis', {})
        
        # Boost confidence for clear cases
        position = spatial.get('relative_position', 'unknown')
        
        if position != "unknown":
            base_confidence += 0.2
        
        if vein_type != "unknown":
            base_confidence += 0.1
        
        # Cap at 1.0
        return min(1.0, base_confidence)
    
    def _generate_reasoning(self, spatial: Dict, vein_type: str, n_level: str) -> str:
        """Generate human-readable reasoning for classification"""
        
        position = spatial.get('relative_position', 'unknown')
        distance_to_fascia = spatial.get('distance_to_fascia_mm', 0)
        intersects = spatial.get('intersects_fascia', False)
        
        reasoning_parts = []
        
        # Explain vein type
        if vein_type == "perforator_vein":
            reasoning_parts.append("Classified as PERFORATOR VEIN: Intersects or crosses fascia")
        elif vein_type == "deep_vein":
            reasoning_parts.append(f"Classified as DEEP VEIN: Located below fascia ({distance_to_fascia:.1f}mm)")
        elif vein_type == "superficial_vein":
            reasoning_parts.append(f"Classified as SUPERFICIAL VEIN: Located above fascia ({distance_to_fascia:.1f}mm)")
        else:
            reasoning_parts.append("Classification requires LLM analysis for disambiguation")
        
        # Add N-level info
        depth_info = spatial.get('depth_info', {})
        distance_from_skin = depth_info.get('distance_from_skin_mm', 0)
        reasoning_parts.append(f"Depth level {n_level}: {distance_from_skin:.1f}mm from skin surface")
        
        return " | ".join(reasoning_parts)
    
    def _requires_llm_confirmation(self, vein_type: str, confidence: float) -> bool:
        """
        Determine if LLM confirmation is needed
        
        Needed for:
        - Unknown classifications
        - Low confidence results
        - Potential GSV candidates
        """
        return vein_type == "unknown" or confidence < 0.6
    
    def classify_batch(self, vein_list: List[Dict]) -> List[Dict]:
        """
        Classify multiple veins
        
        Args:
            vein_list: List of vein dictionaries with spatial analysis
        
        Returns:
            List of veins with classification results
        """
        classified_veins = []
        
        for vein in vein_list:
            classification = self.classify_vein(vein)
            vein_classified = {
                **vein,
                "classification": classification
            }
            classified_veins.append(vein_classified)
        
        # Log summary
        vein_types = {}
        n_levels = {}
        for v in classified_veins:
            vtype = v.get('classification', {}).get('primary_classification', 'unknown')
            nlevel = v.get('classification', {}).get('n_level', 'unknown')
            vein_types[vtype] = vein_types.get(vtype, 0) + 1
            n_levels[nlevel] = n_levels.get(nlevel, 0) + 1
        
        logger.info(f"Classification summary: {vein_types}")
        logger.info(f"N-level summary: {n_levels}")
        
        return classified_veins
    
    def _build_rules(self) -> Dict:
        """Build rule dictionary for reference"""
        return {
            "perforator": "intersects_fascia OR crosses_fascia",
            "deep_vein": "relative_position == 'below'",
            "superficial_vein": "relative_position == 'above'",
            "gsv_candidate": "superficial_vein AND large AND compatible_depth"
        }
