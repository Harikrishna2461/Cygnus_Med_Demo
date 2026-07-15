"""
TASK-2: Probe-Guided Navigation
Guides surgeon towards varicose veins based on real-time probe position.
Uses probe coordinates relative to groin and clinical parameters to provide guidance.
Integrates knowledge from ultrasound anatomy and common pathology patterns.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ProbeOrientation(Enum):
    """Probe scanning orientation."""
    TRANSVERSE = "transverse"  # Axial view (cross-section)
    SAGITTAL = "sagittal"      # Longitudinal view (along vein)
    ANGLED = "angled"          # Angled for valve visualization


class EntryPointMarker(Enum):
    """Entry point classification (EP) vs Re-entry point (RP)."""
    ENTRY_POINT = "EP"      # Where reflux enters incompetent vein
    RE_ENTRY_POINT = "RP"   # Where vein re-enters deep system
    BOTH = "EP/RP"          # Complex pathway with multiple points
    UNKNOWN = "unknown"


class AnatomicalLandmark(Enum):
    """Common ultrasound landmarks for navigation."""
    SAPHENOFEMORAL_JUNCTION = "SFJ"
    SAPHENOPOPLITEAL_JUNCTION = "SPJ"
    HUNTERIAN_CANAL = "hunter"
    MEDIAL_CALF = "med_calf"
    LATERAL_CALF = "lat_calf"
    GROIN = "groin"
    INGUINAL_LIGAMENT = "inguinal"
    ADDUCTOR_CANAL = "adductor"
    POPLITEAL_FOSSA = "popliteal"


@dataclass
class ProbeLocation:
    """Current probe position and orientation."""
    x_ratio: float  # 0-1, relative to image width, 0=medial, 1=lateral
    y_ratio: float  # 0-1, relative to image height, 0=proximal, 1=distal
    depth_mm: float  # Estimated depth in mm (5-100mm typical)
    orientation: ProbeOrientation
    anatomical_region: str  # Current region (groin, thigh, knee, calf)
    confidence: float  # Position confidence (0-1)


@dataclass
class GuidanceInstruction:
    """Guidance instruction for probe movement."""
    action: str  # "move", "rotate", "apply_compression", "perform_valsalva"
    direction: str  # "medial", "lateral", "proximal", "distal", "deep", "superficial"
    magnitude: float  # How much to move (0-10 scale, or mm)
    target_landmark: Optional[str]  # Target anatomical landmark
    reason: str  # Why this move is recommended
    urgency: str  # "routine", "important", "critical"


class ProbeNavigator:
    """
    Real-time probe guidance system.
    Guides surgeon towards pathological veins based on ultrasound data.
    """
    
    # Anatomical regions (defined by Y-ratio relative to groin)
    REGIONS = {
        "groin": (0.0, 0.1),      # At groin level (0-10% from proximal)
        "upper_thigh": (0.1, 0.35),
        "mid_thigh": (0.35, 0.6),
        "lower_thigh": (0.6, 0.85),
        "knee": (0.85, 1.0),
        "upper_calf": (1.0, 1.4),
        "mid_calf": (1.4, 1.7),
        "lower_calf": (1.7, 2.0)
    }
    
    # Vein position expectations (x_ratio: medial=0, lateral=1)
    VEIN_POSITIONS = {
        "N1_femoral": {
            "x_range": (0.35, 0.65),  # More central
            "depth_range": (30, 60),
            "description": "Deep femoral vein (deeper, more central)"
        },
        "N2_gsv": {
            "x_range": (0.1, 0.4),    # More medial
            "depth_range": (5, 25),
            "description": "Great saphenous vein (superficial, medial)"
        },
        "N2_ssv": {
            "x_range": (0.6, 0.9),    # More lateral/posterior
            "depth_range": (10, 30),
            "description": "Small saphenous vein (superficial, posterior)"
        },
        "N3_tributary": {
            "x_range": (0.0, 1.0),    # Highly variable
            "depth_range": (2, 20),
            "description": "Tributary veins (very superficial)"
        },
        "perforator": {
            "x_range": (0.2, 0.8),
            "depth_range": (15, 40),
            "description": "Perforating vein (connecting superficial to deep)"
        }
    }
    
    # Navigation pathways for different pathologies
    PATHOLOGY_NAVIGATION = {
        "gsv_incompetence": {
            "first_step": "Locate saphenofemoral junction",
            "steps": [
                "Position at groin level, SFJ region",
                "Identify GSV in short axis with junctional valve",
                "Perform Valsalva to assess valve cusp separation",
                "Measure GSV diameter at SFJ",
                "Track GSV distally along medial thigh",
                "Identify tributary junction points",
                "Assess reflux duration with compression release"
            ]
        },
        "ssv_incompetence": {
            "first_step": "Locate saphenopopliteal junction",
            "steps": [
                "Position at popliteal fossa, medial aspect",
                "Identify SSV in short axis",
                "Locate popliteal vein junction (posterior to femur)",
                "Assess valve competence at SPJ",
                "Track SSV proximally along calf",
                "Identify medial calf perforators",
                "Assess reflux with Valsalva"
            ]
        },
        "perforator_incompetence": {
            "first_step": "Locate incompetent perforator",
            "steps": [
                "Identify entry point on deep vein",
                "Track perforator course through fascia",
                "Locate exit point on superficial vein",
                "Assess bidirectional flow on Doppler",
                "Measure perforator diameter",
                "Note exact anatomical location (medial, lateral, calf)",
                "Identify level relative to anatomical landmarks"
            ]
        },
        "pelvic_insufficiency": {
            "first_step": "Assess pelvic venous sources",
            "steps": [
                "Move proximal to groin, assess iliac veins",
                "Identify reflux entering at SFJ level",
                "Track flow pattern through proximal GSV",
                "Assess for gonadal vein involvement",
                "Note extent of pelvic contribution",
                "Evaluate for May-Thurner syndrome if present"
            ]
        }
    }
    
    def __init__(self):
        self.current_location: Optional[ProbeLocation] = None
        self.scan_history: List[ProbeLocation] = []
        self.guidance_history: List[GuidanceInstruction] = []
        self.detected_pathology: Optional[Dict] = None
        
    def update_probe_position(self, data_point: Dict) -> Dict:
        """
        Update probe position from incoming data point.
        
        Args:
            data_point: Data point with structure:
                {
                    "posXRatio": float (0-1),
                    "posYRatio": float (0-1),
                    "flow": "EP" | "RP",
                    "step": str,
                    "legSide": "left" | "right",
                    "fromType": str (N1/N2/N3/P/B),
                    "toType": str,
                    ...
                }
        
        Returns:
            Guidance information and current navigation status
        """
        try:
            # Extract position data
            x_ratio = data_point.get('posXRatio', 0.5)
            y_ratio = data_point.get('posYRatio', 0.5)
            flow_type = data_point.get('flow', 'unknown')
            step_name = data_point.get('step', 'scanning')
            vein_from = data_point.get('fromType', 'unknown')
            vein_to = data_point.get('toType', 'unknown')
            
            # Create probe location
            self.current_location = ProbeLocation(
                x_ratio=x_ratio,
                y_ratio=y_ratio,
                depth_mm=self._estimate_depth(vein_from, x_ratio, y_ratio),
                orientation=self._determine_orientation(data_point),
                anatomical_region=self._get_anatomical_region(y_ratio),
                confidence=data_point.get('confidence', 0.85)
            )
            
            self.scan_history.append(self.current_location)
            
            # Determine what vein should be in view
            expected_veins = self._get_veins_at_location(
                self.current_location.anatomical_region,
                x_ratio,
                y_ratio,
                vein_from
            )
            
            # Check if we're looking at correct structure
            is_correct_location = vein_from in expected_veins
            
            # Generate guidance
            guidance = self._generate_guidance(
                self.current_location,
                expected_veins,
                is_correct_location,
                step_name,
                flow_type
            )
            
            logger.debug(f"Probe at {self.current_location.anatomical_region}, "
                        f"X={x_ratio:.2f}, Y={y_ratio:.2f}, "
                        f"Expected: {expected_veins}, Found: {vein_from}")
            
            return {
                "current_location": {
                    "region": self.current_location.anatomical_region,
                    "x_ratio": x_ratio,
                    "y_ratio": y_ratio,
                    "depth_mm": self.current_location.depth_mm,
                    "orientation": self.current_location.orientation.value
                },
                "expected_veins": expected_veins,
                "is_correct_location": is_correct_location,
                "current_vein": vein_from,
                "guidance": guidance,
                "anatomical_context": self._get_anatomical_context(
                    self.current_location.anatomical_region
                ),
                "next_steps": self._get_next_steps_for_pathology(vein_from, vein_to)
            }
            
        except Exception as e:
            logger.error(f"Error updating probe position: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def _estimate_depth(self, vein_type: str, x_ratio: float, y_ratio: float) -> float:
        """Estimate scanning depth based on vein type."""
        vein_depth_range = self.VEIN_POSITIONS.get(
            f"N{vein_type.lstrip('N')}",
            {"depth_range": (10, 30)}
        ).get("depth_range", (10, 30))
        
        # Return average depth
        return sum(vein_depth_range) / 2
    
    def _determine_orientation(self, data_point: Dict) -> ProbeOrientation:
        """Determine likely probe orientation from context."""
        # In real implementation, would use more sophisticated detection
        # For now, infer from flow visibility
        if "flow" in data_point:
            return ProbeOrientation.SAGITTAL  # Flow best seen in sagittal
        return ProbeOrientation.TRANSVERSE
    
    def _get_anatomical_region(self, y_ratio: float) -> str:
        """Determine anatomical region from Y coordinate."""
        for region, (y_min, y_max) in self.REGIONS.items():
            if y_min <= y_ratio <= y_max:
                return region
        return "unknown"
    
    def _get_veins_at_location(
        self,
        region: str,
        x_ratio: float,
        y_ratio: float,
        current_vein: str
    ) -> List[str]:
        """
        Determine which veins should be visible at this location.
        
        Returns:
            List of vein codes that should be in view (N1, N2, N3, etc.)
        """
        veins_at_location = []
        
        # Groin level - SFJ area
        if region == "groin":
            veins_at_location = ["N1", "N2"]  # Femoral and GSV meet here
            if 0.3 < x_ratio < 0.7:
                veins_at_location.append("N1")  # Deep femoral more central
            if x_ratio < 0.4:
                veins_at_location.insert(1, "N2")  # GSV more medial
        
        # Thigh regions - main GSV course
        elif "thigh" in region:
            veins_at_location = ["N2", "N3"]  # GSV and tributaries
            veins_at_location.append("N1")  # Femoral vein deeper
            if x_ratio < 0.25:
                veins_at_location.insert(0, "N2")  # GSV path
        
        # Knee region - transition to calf
        elif region == "knee":
            veins_at_location = ["N2", "N3", "N1"]
        
        # Calf regions - SSV, deep veins, perforators
        elif "calf" in region:
            if x_ratio > 0.5:  # Lateral/posterior
                veins_at_location = ["N2", "N3"]  # SSV region
            else:  # Medial
                veins_at_location = ["N3", "N1"]  # Medial tributaries and perforators
        
        return veins_at_location
    
    def _generate_guidance(
        self,
        location: ProbeLocation,
        expected_veins: List[str],
        is_correct_location: bool,
        step_name: str,
        flow_type: str
    ) -> Dict:
        """Generate real-time guidance instructions."""
        instructions = []
        
        if not is_correct_location:
            # Guide to correct location
            if "groin" not in location.anatomical_region:
                instructions.append({
                    "action": "move",
                    "direction": "proximal",
                    "magnitude": 5.0,
                    "target_landmark": "Saphenofemoral Junction",
                    "reason": "Start anatomical survey from proximal reference point (SFJ)",
                    "urgency": "important"
                })
            
            # Adjust medial-lateral position
            if location.x_ratio > 0.5:
                instructions.append({
                    "action": "move",
                    "direction": "medial",
                    "magnitude": abs(location.x_ratio - 0.35),
                    "target_landmark": "GSV axis",
                    "reason": f"Move toward GSV position (medial). Found at x={location.x_ratio:.2f}, should be ~0.3",
                    "urgency": "routine"
                })
        else:
            # At correct location, optimize view
            instructions.append({
                "action": "optimize_view",
                "direction": "fine_tune",
                "magnitude": 0.5,
                "reason": f"Optimize visualization of {', '.join(expected_veins)}",
                "urgency": "routine"
            })
        
        # Add reflection-specific instructions
        if flow_type == "EP":
            instructions.append({
                "action": "mark_location",
                "direction": "mark",
                "magnitude": 0.0,
                "target_landmark": "Entry Point",
                "reason": "Mark entry point where reflux begins (proximal source)",
                "urgency": "critical"
            })
        elif flow_type == "RP":
            instructions.append({
                "action": "mark_location",
                "direction": "mark",
                "magnitude": 0.0,
                "target_landmark": "Re-entry Point",
                "reason": "Mark re-entry point where vein returns to deep system",
                "urgency": "critical"
            })
        
        return {
            "instructions": instructions,
            "primary_target": f"Visualize {', '.join(expected_veins)}",
            "next_action": instructions[0]["action"] if instructions else "continue_scanning"
        }
    
    def _get_anatomical_context(self, region: str) -> str:
        """Provide anatomical context for current region."""
        context_map = {
            "groin": "Saphenofemoral Junction (SFJ): GSV joins femoral vein. Key site for valve assessment and treatment.",
            "upper_thigh": "GSV course along medial thigh. Look for tributaries and perforator junctions.",
            "mid_thigh": "Mid-thigh GSV region. Common site for tributary reflux and incompetent perforators.",
            "lower_thigh": "Lower thigh GSV course. Approaching knee region. Watch for Hunterian canal perforators.",
            "knee": "Knee region transition. GSV may continue or join popliteal system.",
            "upper_calf": "Upper calf - bifurcation region. Medial calf perforators common here.",
            "mid_calf": "Mid-calf region. Key perforator locations. Assess for medial/lateral incompetence.",
            "lower_calf": "Lower calf near ankle. Deep venous system and distal tributaries."
        }
        return context_map.get(region, "Location context not available")
    
    def _get_next_steps_for_pathology(self, vein_from: str, vein_to: str) -> List[str]:
        """Get recommended next scanning steps based on detected pathology."""
        steps = []
        
        # Build flow pattern
        flow_pattern = f"{vein_from}-{vein_to}"
        
        # GSV incompetence (N1-N2)
        if "N1" in vein_from and "N2" in vein_to:
            steps = [
                "✓ Confirm SFJ valve incompetence with Valsalva",
                "→ Track GSV distally to identify tributaries",
                "→ Assess tributary reflux duration",
                "→ Look for medial thigh perforators",
                "→ Check for calf perforator involvement"
            ]
        
        # Tributary involvement (N2-N3)
        elif "N2" in vein_from and "N3" in vein_to:
            steps = [
                "✓ Visualize tributary junction with GSV",
                "→ Measure tributary diameter",
                "→ Assess flow direction and reflux duration",
                "→ Trace tributary course",
                "→ Consider for sclerotherapy or ablation"
            ]
        
        # Perforator involvement (N1-N3)
        elif "N1" in vein_from and "N3" in vein_to:
            steps = [
                "✓ Identify perforator entry and exit points",
                "→ Measure perforator diameter",
                "→ Confirm bidirectional flow",
                "→ Note fascial crossing",
                "→ Mark for targeted ligation or ablation"
            ]
        
        # Pelvic sources (P-N2 or P-N1)
        elif "P" in vein_from:
            steps = [
                "✓ Assess pelvic as reflux source",
                "→ Evaluate iliac and gonadal contributions",
                "→ Trace pathway through SFJ",
                "→ Document pelvic origin",
                "→ Plan proximal intervention"
            ]
        
        return steps
    
    def provide_real_time_guidance(self, current_data: Dict) -> str:
        """
        Provide a single, actionable guidance message in real time.
        
        Args:
            current_data: Current probe position and flow data
        
        Returns:
            Single most important guidance message
        """
        result = self.update_probe_position(current_data)
        
        if "error" in result:
            return "Error processing position data"
        
        guidance = result.get("guidance", {})
        instructions = guidance.get("instructions", [])
        
        if not instructions:
            return "Continue scanning systematically"
        
        # Return most urgent instruction
        instructions.sort(key=lambda x: {
            "critical": 0,
            "important": 1,
            "routine": 2
        }.get(x.get("urgency", "routine"), 3))
        
        primary = instructions[0]
        
        # Format as clear command
        if primary["action"] == "move":
            return (f"Move {primary['direction'].upper()} to {primary['target_landmark']}: "
                   f"{primary['reason']}")
        elif primary["action"] == "mark_location":
            return (f"MARK: {primary['target_landmark']} - {primary['reason']}")
        elif primary["action"] == "optimize_view":
            return f"Fine-tune probe to optimize {primary['reason']}"
        
        return primary.get("reason", "Continue scanning")
