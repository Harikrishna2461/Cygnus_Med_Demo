"""
REFERENCE: Shunt Type Classification from Table 3.2
Medical definitions - Ultrasound Probe Guidance

This is a REFERENCE FILE. The LLM performs actual classification.

TABLE 3.2 - VENOUS SHUNT CLASSIFICATION:

SHUNT TYPE 1 (N1-N2-N1):
- Reflux: N1-N2, Location: GSV
- Pattern: N1→N2→N1 reflux through great saphenous vein
- Figure: 3.19a, 3.22

SHUNT TYPE 2 (N2-N3):
- Reflux: N2-N3, Location: Tributary
- Pattern: Simple saphenous to tributary reflux
- Figure: 3.23, 3.24

SHUNT TYPE 3 (N1-N2-N3-N1, Two Sources):
- Reflux: N1-N2 or Two Sources
- Pattern: N1-N2-N3-N1 or dual entry points
- Figure: 3.25a, 3.17a

SHUNT TYPE 4 PELVIC (P-N2-N1):
- Reflux: Pelvic Reflux, Location: Pelvic
- Pattern: P→N2→N1 from pelvic origins
- Figure: 3.27a

SHUNT TYPE 4 PERFORATOR (N1-N3-N2-N1, Bone Perforator):
- Reflux: N1-N3 or Bone Perforator
- Pattern: Perforating veins to tributaries
- Figure: 3.27s, 3.13

SHUNT TYPE 5 PELVIC (P-N3-N2-N1):
- Reflux: Pelvic Reflux
- Pattern: Complex pelvic to saphenous pathway
- Figure: 3.29a

SHUNT TYPE 5 PERFORATOR (N1-N3-N2-N3-N1, Bone Perforator):
- Reflux: N1-N3 or Bone Perforator  
- Pattern: Complex perforator pathways
- Figure: 3.29b, 3.13

SHUNT TYPE 6 (N1-N3-N2):
- Reflux: N1-N3
- Pattern: Deep to tributary pathway
- Figure: 3.30

VEIN CATEGORIES:
- N1: Deep veins (Femoral, Popliteal, Perforating)
- N2: Saphenous axis (GSV, SSV)
- N3: Tributaries (Superficial tributaries, reticularveins)
- B: Bone perforator

The LLM uses this table to classify shunt types based on:
reflux_type + location + reflux_duration + description

NOTES FOR LLM CLASSIFICATION:
- Reflux Type values: "N1-N2", "Pelvic Reflux", "N1-N3", "N2-N3", "Bone Perforator", "Two Sources"
- Location values: "GSV", "SSV", "Pelvic", "Perforator", "Tributary", "Bone Perforator", "Mixed"
- Description values: Flow patterns like "N1-N2-N1", "N2-N3", "P-N2-N1", etc.
- Match the combination of (reflux_type, description) to Table 3.2 for classification
"""

# Dummy reference - the actual classification is done by the LLM in app.py
SHUNT_REFERENCE_TABLE = {
    "Type 1": {
        "reflux_type": "N1-N2",
        "location": "GSV",
        "patterns": ["N1-N2-N1"],
        "figure": "3.19a, 3.22"
    },
    "Type 2": {
        "reflux_type": "N2-N3",
        "location": "Tributary",
        "patterns": ["N2-N3"],
        "figure": "3.23, 3.24"
    },
    "Type 3": {
        "reflux_type": ["N1-N2", "Two Sources"],
        "location": ["GSV", "Mixed"],
        "patterns": ["N1-N2-N3-N1", "N1-N2-N3"],
        "figure": "3.25a, 3.17a"
    },
    "Type 4 Pelvic": {
        "reflux_type": "Pelvic Reflux",
        "location": "Pelvic",
        "patterns": ["P-N2-N1"],
        "figure": "3.27a"
    },
    "Type 4 Perforator": {
        "reflux_type": ["N1-N3", "Bone Perforator"],
        "location": ["Perforator", "Bone Perforator"],
        "patterns": ["N1-N3-N2-N1", "N1-B-N3-N2-N1"],
        "figure": "3.27s, 3.13"
    },
    "Type 5 Pelvic": {
        "reflux_type": "Pelvic Reflux",
        "location": "Pelvic",
        "patterns": ["P-N3-N2-N1", "P-N3-N2-N3-N1"],
        "figure": "3.29a"
    },
    "Type 5 Perforator": {
        "reflux_type": ["N1-N3", "Bone Perforator"],
        "location": ["Perforator", "Bone Perforator"],
        "patterns": ["N1-N3-N2-N3-N1", "N1-B-N3-N2-N3-N1"],
        "figure": "3.29b, 3.13"
    },
    "Type 6": {
        "reflux_type": "N1-N3",
        "location": "Perforator",
        "patterns": ["N1-N3-N2"],
        "figure": "3.30"
    }
}
"""
# END OF REFERENCE FILE - LLM classification happens in app.py
    "greater saphenous": "N2",
    "ssv": "N2",
    "small saphenous": "N2",
    "saphenous": "N2",
    
    # N3 - Tributaries
    "tributary": "N3",
    "reticularveins": "N3",
    "superficial tributary": "N3",
    "lateral": "N3",
    "medial": "N3",
    
    # N4 - Communicating
    "communicating": "N4",
    "perforate": "N4",
    
    # B - Bone
    "bone": "B",
"""

# Reflux path patterns and corresponding shunt types
REFLUX_PATTERNS = {
    # SHUNT I - Simple reflux (N1-N2)
    ("N1", "N2"): "Shunt I",
    
    # SHUNT II - Reflux with secondary path (Pelvic/Perforator origin)
    ("pelvic", "N2"): "Shunt II (Pelvic)",
    ("perforator", "N2"): "Shunt II (Perforator)",
    
    # SHUNT III - Reflux through tributaries (N1-N3)
    ("N1", "N3"): "Shunt III",
    
    # SHUNT IV - Bone perforator involvement
    ("bone", "N2"): "Shunt IV",
    ("B", "N2"): "Shunt IV",
    
    # SHUNT V - Multiple pathways with communicating veins
    ("N1", "communicating"): "Shunt V",
}


class ShuntClassifier:
    """Classify venous shunt types based on reflux parameters"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def classify_shunt(
        self,
        reflux_type: str,
        location: str,
        reflux_duration: float,
        description: str = ""
    ) -> Dict:
        """
        Classify shunt type based on input parameters
        
        Args:
            reflux_type: Type of reflux ("pelvic", "perforator", "tributary", etc.)
            location: Vein location (GSV, SSV, tributary, etc.)
            reflux_duration: Duration of reflux in seconds
            description: Additional clinical description
            
        Returns:
            Dict with:
            - shunt_type: Classified shunt type (I-V)
            - confidence: Confidence score (0-1)
            - vein_path: Vein pathway (N1-N2 format)
            - reasoning: Clinical reasoning
        """
        try:
            # Normalize inputs
            reflux_type_normalized = reflux_type.lower().strip()
            location_normalized = location.lower().strip()
            description_normalized = description.lower().strip()
            
            # Get vein classifications
            source_vein = self._get_vein_class(reflux_type_normalized)
            target_vein = self._get_vein_class(location_normalized)
            
            # Analyze reflux characteristics
            reflux_characteristics = self._analyze_reflux(
                reflux_type_normalized,
                location_normalized,
                reflux_duration,
                description_normalized
            )
            
            # Determine shunt type
            shunt_type, confidence = self._determine_shunt_type(
                source_vein,
                target_vein,
                reflux_characteristics,
                description_normalized
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                shunt_type,
                reflux_characteristics,
                source_vein,
                target_vein
            )
            
            return {
                "shunt_type": shunt_type,
                "confidence": confidence,
                "vein_path": f"{source_vein}-{target_vein}",
                "reflux_characteristics": reflux_characteristics,
                "reasoning": reasoning,
                "classification_method": "Medical diagram reference (Table 3.2)"
            }
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return {
                "shunt_type": "Unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _get_vein_class(self, vein_name: str) -> str:
        """Get vein classification (N1, N2, N3, N4, B)"""
        vein_lower = vein_name.lower().strip()
        
        # Direct lookup
        for key, classification in VEIN_CLASSIFICATIONS.items():
            if key in vein_lower:
                return classification
        
        # Default to N3 (tributary) for unknown
        return "N3"
    
    def _analyze_reflux(
        self,
        reflux_type: str,
        location: str,
        duration: float,
        description: str
    ) -> Dict:
        """Analyze reflux characteristics"""
        characteristics = {
            "is_pelvic": "pelvic" in reflux_type,
            "is_perforator": "perforator" in reflux_type,
            "is_tributary": "tributary" in reflux_type,
            "is_bone": "bone" in reflux_type,
            "duration_seconds": duration,
            "is_prolonged": duration > 1.0,  # > 1 second
            "is_severe": duration > 2.0,      # > 2 seconds
        }
        
        # Check description for additional patterns
        if description:
            characteristics["has_multiple_sources"] = "multiple" in description
            characteristics["has_communicating_veins"] = "communicating" in description
            characteristics["segmental_reflux"] = "segment" in description
        
        return characteristics
    
    def _determine_shunt_type(
        self,
        source_vein: str,
        target_vein: str,
        reflux_characteristics: Dict,
        description: str
    ) -> Tuple[str, float]:
        """
        Determine shunt type based on vein pathway and characteristics
        
        Returns:
            (shunt_type, confidence_score)
        """
        confidence = 0.7  # Base confidence
        
        # Check for specific patterns
        
        # SHUNT V - Multiple vein involvement with communicating veins
        if reflux_characteristics.get("has_communicating_veins"):
            return "Shunt V - Communicating Veins", 0.9
        
        # SHUNT IV - Bone perforator
        if reflux_characteristics.get("is_bone") or "bone" in description:
            return "Shunt IV - Bone Perforator", 0.85
        
        # SHUNT III - N1 to N3 (tributaries)
        if source_vein == "N1" and target_vein == "N3":
            if reflux_characteristics.get("is_prolonged"):
                confidence = 0.85
            return "Shunt III - Tributary Secondary", confidence
        
        # SHUNT II - Pelvic or Perforator sources
        if reflux_characteristics.get("is_pelvic"):
            if target_vein in ["N2"]:
                if reflux_characteristics.get("is_severe"):
                    return "Shunt II - Pelvic (Severe)", 0.9
                return "Shunt II - Pelvic", 0.85
        
        if reflux_characteristics.get("is_perforator"):
            if target_vein in ["N2", "N3"]:
                if reflux_characteristics.get("is_severe"):
                    return "Shunt II - Perforator (Severe)", 0.9
                return "Shunt II - Perforator", 0.85
        
        # SHUNT I - Simple reflux (N1-N2 or N2-N1)
        if (source_vein == "N1" and target_vein == "N2") or \
           (source_vein == "N2" and target_vein == "N1"):
            if reflux_characteristics.get("is_prolonged"):
                return "Shunt I - Deep to Saphenous", 0.85
            else:
                return "Shunt I - Simple Reflux", 0.8
        
        # Default classification based on confidence
        return "Shunt Type Unclassified", 0.5
    
    def _generate_reasoning(
        self,
        shunt_type: str,
        reflux_characteristics: Dict,
        source_vein: str,
        target_vein: str
    ) -> str:
        """Generate clinical reasoning for classification"""
        
        reasoning_parts = []
        
        # Vein pathway reasoning
        if source_vein and target_vein:
            reasoning_parts.append(
                f"Vein pathway identified: {source_vein} → {target_vein}"
            )
        
        # Reflux characteristics
        if reflux_characteristics.get("is_severe"):
            reasoning_parts.append("Severe reflux pattern detected (>2 seconds)")
        elif reflux_characteristics.get("is_prolonged"):
            reasoning_parts.append("Prolonged reflux detected (>1 second)")
        
        # Special characteristics
        if reflux_characteristics.get("has_communicating_veins"):
            reasoning_parts.append("Multiple venous pathways with communicating veins identified")
        
        if reflux_characteristics.get("is_pelvic"):
            reasoning_parts.append("Pelvic venous involvement detected")
        
        if reflux_characteristics.get("is_perforator"):
            reasoning_parts.append("Perforator origin identified")
        
        # Classification basis
        reasoning_parts.append(f"Classified as: {shunt_type}")
        
        return ". ".join(reasoning_parts) + "."


# TABLE 3.2 CLASSIFICATION - EXACT MAPPING FROM MEDICAL TABLE
TABLE_3_2_CLASSIFICATION = {
    # Reflux type + description flow pattern → (shunt type, figure)
    ("N1-N2", "N1-N2-N1"): ("Type 1", "3.19a, 3.22"),
    ("N1-N2", "N1-N2-N1-N3"): ("Type 1+2", "3.25"),
    ("N1-N2", "N1-N2-N3-N1"): ("Type 3", "3.25a"),
    ("Pelvic Reflux", "P-N2-N1"): ("Type 4 Pelvic", "3.27a"),
    ("Pelvic Reflux", "P-N3-N2-N1"): ("Type 5 Pelvic", "3.29a"),
    ("Pelvic Reflux", "P-N3-N2-N3-N1"): ("Type 5 Pelvic", "3.29a"),
    ("N1-N3", "N1-N3-N2-N1"): ("Type 4 Perforator", "3.27s"),
    ("N1-N3", "N1-N3-N2-N3-N1"): ("Type 5 Perforator", "3.29b"),
    ("N1-N3", "N1-N3-N2"): ("Type 6", "3.30"),
    ("N2-N3", "N2-N3"): ("Type 2", "3.23, 3.24"),
    ("Bone Perforator", "N1-B-N3-N1"): ("Type 4 Perforator", "3.13"),
    ("Bone Perforator", "N1-B-N3-N2-N1"): ("Type 4 Perforator", "3.13"),
    ("Bone Perforator", "N1-B-N3-N2-N3-N1"): ("Type 5 Perforator", "3.13"),
    ("Two Sources", "N1-N2-N3"): ("Type 3", "3.17a"),
    ("Two Sources", "N1-N3-N2-N3-N1"): ("Type 5 Perforator", "3.17b"),
}

def identify_shunt_from_stream(ultrasound_data: Dict) -> Dict:
    """
    Identify shunt type from continuous stream data using TABLE 3.2
    
    Args:
        ultrasound_data: Dict with 'reflux_type', 'description', 'location', etc.
        
    Returns:
        Classification result with shunt type per medical table
    """
    reflux_type = ultrasound_data.get("reflux_type", "").strip()
    description = ultrasound_data.get("description", "").strip()
    location = ultrasound_data.get("location", "").strip()
    reflux_duration = ultrasound_data.get("reflux_duration", 0.0)
    
    # Direct table lookup using exact reflux_type + description
    key = (reflux_type, description)
    
    if key in TABLE_3_2_CLASSIFICATION:
        shunt_type, figure = TABLE_3_2_CLASSIFICATION[key]
        return {
            "shunt_type": f"Shunt type {shunt_type} detected",
            "confidence": 0.95,
            "vein_path": description,
            "reflux_type": reflux_type,
            "location": location,
            "reflux_duration": reflux_duration,
            "figure_reference": f"Figure {figure}",
            "source": "Table 3.2 - Medical Classification Reference",
            "reasoning": f"Flow pattern '{description}' with reflux type '{reflux_type}' at {location} matches {shunt_type}"
        }
    
    # Fallback: Try to match just on description for close patterns
    for (rt, desc), (shunt_type, figure) in TABLE_3_2_CLASSIFICATION.items():
        if desc in description or description in desc:
            return {
                "shunt_type": f"Shunt type {shunt_type} detected",
                "confidence": 0.85,
                "vein_path": description,
                "reflux_type": reflux_type,
                "location": location,
                "reflux_duration": reflux_duration,
                "figure_reference": f"Figure {figure}",
                "source": "Table 3.2 - Medical Classification Reference (Pattern Match)",
                "reasoning": f"Flow pattern similar to {shunt_type}"
            }
    
    # Unknown classification
    return {
        "shunt_type": "Shunt type UNCLASSIFIED - review input data",
        "confidence": 0.0,
        "vein_path": description,
        "reflux_type": reflux_type,
        "location": location,
        "reflux_duration": reflux_duration,
        "source": "Table 3.2 - Medical Classification Reference",
        "reasoning": f"No match found for reflux_type='{reflux_type}' with description='{description}' in Table 3.2. Ensure inputs match reference table exactly."
    }
