"""
TASK-1: Temporal Flow Analysis
Tracks vein flow sequences across consecutive ultrasound frames.
Detects abnormal circular flow patterns (e.g., N1→N2→N3→N1).
Classifies shunt type when abnormal flow is detected.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class VeinType(Enum):
    """Vein classification categories."""
    N1 = "N1"  # Deep veins (Femoral, Popliteal, Perforating)
    N2 = "N2"  # Saphenous axis (GSV, SSV)
    N3 = "N3"  # Tributaries (Superficial tributaries, reticularveins)
    P = "P"    # Pelvic veins
    B = "B"    # Bone perforators


@dataclass
class FlowSequencePoint:
    """Represents a single point in the flow sequence."""
    sequence_number: int
    from_type: str  # Source vein type (N1, N2, N3, P, B)
    to_type: str    # Target vein type
    step_name: str  # Clinical step name (e.g., "SFJ-Knee")
    flow_direction: str  # Direction of flow
    confidence: float  # Detection confidence (0-1)
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class AbnormalFlowPattern:
    """Detected abnormal flow pattern."""
    pattern_sequence: List[str]  # e.g., ["N1", "N2", "N3", "N1"]
    is_circular: bool  # True if forms a loop
    entry_point: str  # Where abnormal flow enters
    exit_point: str  # Where it re-enters deep system
    reflux_points: List[str]  # Intermediate reflux locations
    total_duration_frames: int
    severity: str  # "mild", "moderate", "severe"


class TemporalFlowAnalyzer:
    """
    Analyzes temporal flow sequences in ultrasound data.
    Detects abnormal patterns and prepares for shunt classification.
    """
    
    # Known shunt patterns from medical literature
    SHUNT_PATTERNS = {
        "Type 1": {
            "pattern": ["N1", "N2", "N1"],
            "description": "Simple reflux GSV",
            "reflux_type": "N1-N2"
        },
        "Type 2": {
            "pattern": ["N2", "N3"],
            "description": "Tributary reflux",
            "reflux_type": "N2-N3"
        },
        "Type 3": {
            "pattern": ["N1", "N2", "N3", "N1"],
            "description": "Complex GSV with tributary involvement",
            "reflux_type": "N1-N2-N3"
        },
        "Type 4 Pelvic": {
            "pattern": ["P", "N2", "N1"],
            "description": "Pelvic to saphenous to deep",
            "reflux_type": "Pelvic Reflux"
        },
        "Type 4 Perforator": {
            "pattern": ["N1", "N3", "N2", "N1"],
            "description": "Bone perforator involvement",
            "reflux_type": "N1-N3-N2"
        },
        "Type 5 Pelvic": {
            "pattern": ["P", "N3", "N2", "N1"],
            "description": "Complex pelvic pathway",
            "reflux_type": "Pelvic Reflux"
        },
        "Type 5 Perforator": {
            "pattern": ["N1", "N3", "N2", "N3", "N1"],
            "description": "Complex perforator pathway",
            "reflux_type": "N1-N3-N2-N3"
        },
        "Type 6": {
            "pattern": ["N1", "N3", "N2"],
            "description": "Deep to tributary to saphenous",
            "reflux_type": "N1-N3"
        }
    }
    
    def __init__(self, max_history: int = 100):
        """
        Initialize temporal flow analyzer.
        
        Args:
            max_history: Maximum number of flow points to track
        """
        self.flow_history: deque = deque(maxlen=max_history)
        self.detected_patterns: List[AbnormalFlowPattern] = []
        self.current_sequence: List[str] = []
        self.session_id = None
        self.start_time = datetime.now()
        
    def add_flow_point(self, data_point: Dict) -> Optional[AbnormalFlowPattern]:
        """
        Add a new flow observation point and check for abnormal patterns.
        
        Args:
            data_point: Data point from video sequence with structure:
                {
                    "sequenceNumber": int,
                    "fromType": str (N1/N2/N3/P/B),
                    "toType": str,
                    "step": str,
                    "flow": str (RP/EP - retroperitoneal/entry point),
                    "clipPath": str,
                    "posXRatio": float,
                    "posYRatio": float,
                    "processingProgress": int,
                    ...
                }
        
        Returns:
            AbnormalFlowPattern if abnormal flow detected, None otherwise
        """
        try:
            # Extract flow information
            from_type = data_point.get('fromType', 'N1')
            to_type = data_point.get('toType', 'N2')
            step_name = data_point.get('step', 'Unknown')
            seq_num = data_point.get('sequenceNumber', 0)
            
            # Create flow point
            flow_point = FlowSequencePoint(
                sequence_number=seq_num,
                from_type=from_type,
                to_type=to_type,
                step_name=step_name,
                flow_direction=self._determine_flow_direction(from_type, to_type),
                confidence=data_point.get('confidence', 0.8),
                timestamp=datetime.now(),
                metadata=data_point
            )
            
            # Add to history
            self.flow_history.append(flow_point)
            self.current_sequence.append(from_type)
            
            # If we have at least 2 points, predict next and append to_type if it completes
            if len(self.current_sequence) > 1:
                self.current_sequence.append(to_type)
            
            logger.debug(f"Flow point added: {from_type}→{to_type} at step {step_name}")
            
            # Check for abnormal patterns
            abnormal = self._detect_abnormal_flow()
            if abnormal:
                self.detected_patterns.append(abnormal)
                logger.warning(f"Abnormal flow pattern detected: {abnormal.pattern_sequence}")
                return abnormal
                
            return None
            
        except Exception as e:
            logger.error(f"Error adding flow point: {e}")
            return None
    
    def _determine_flow_direction(self, from_type: str, to_type: str) -> str:
        """Determine if flow is normal (centripetal) or abnormal (reflux)."""
        # Centripetal flow is from superficial/perfo to deep (normal)
        # N1 = deep, N2 = saphenous, N3 = tributary, P = pelvic, B = bone
        
        normal_paths = [
            ("N3", "N2"),  # Tributary to saphenous
            ("N2", "N1"),  # Saphenous to deep
            ("B", "N2"),   # Bone perforator to saphenous
            ("P", "N2"),   # Pelvic to saphenous (in some cases)
            ("N3", "N1"),  # Tributary to deep via perforator
        ]
        
        if (from_type, to_type) in normal_paths:
            return "normal"
        elif (to_type, from_type) in normal_paths:
            return "reflux"
        else:
            return "unknown"
    
    def _detect_abnormal_flow(self) -> Optional[AbnormalFlowPattern]:
        """
        Detect abnormal flow patterns by checking for:
        1. Circular flows (return to starting point)
        2. Persistent reflux
        3. Multiple entry/exit points
        
        Returns:
            AbnormalFlowPattern if detected, None otherwise
        """
        if len(self.current_sequence) < 3:
            return None  # Need at least 3 points to detect pattern
        
        # Get recent sequence (last 5-10 points)
        recent_seq = list(self.flow_history)[-10:]
        sequence_types = [f.from_type for f in recent_seq]
        
        # Add last to_type if available
        if recent_seq:
            sequence_types.append(recent_seq[-1].to_type)
        
        # Check for known abnormal patterns
        for shunt_type, pattern_info in self.SHUNT_PATTERNS.items():
            if self._matches_pattern(sequence_types, pattern_info["pattern"]):
                # Determine severity
                severity = "moderate"
                if len(sequence_types) >= 5:  # Longer sequences are more severe
                    severity = "severe"
                elif len(sequence_types) == 3:
                    severity = "mild"
                
                abnormal = AbnormalFlowPattern(
                    pattern_sequence=sequence_types,
                    is_circular=sequence_types[0] == sequence_types[-1],
                    entry_point=sequence_types[0],
                    exit_point=sequence_types[-1],
                    reflux_points=sequence_types[1:-1],
                    total_duration_frames=len(sequence_types),
                    severity=severity
                )
                
                logger.info(f"Pattern match: {shunt_type} - {abnormal.pattern_sequence}")
                return abnormal
        
        return None
    
    def _matches_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence matches expected pattern (allowing some flexibility)."""
        if len(sequence) < len(pattern):
            return False
        
        # Direct match
        if sequence == pattern:
            return True
        
        # Partial match (sequence contains pattern as subset)
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        
        return False
    
    def get_flow_summary(self) -> Dict:
        """Get summary of current flow state."""
        if not self.flow_history:
            return {
                "total_points": 0,
                "current_sequence": [],
                "abnormal_patterns_detected": 0,
                "last_update": None
            }
        
        return {
            "total_points": len(self.flow_history),
            "current_sequence": list(self.current_sequence),
            "abnormal_patterns_detected": len(self.detected_patterns),
            "last_update": self.flow_history[-1].timestamp.isoformat(),
            "flow_direction_mix": self._analyze_flow_directions(),
            "entry_points": self._extract_entry_points(),
            "exit_points": self._extract_exit_points()
        }
    
    def _analyze_flow_directions(self) -> Dict[str, int]:
        """Analyze mix of normal vs reflux flow."""
        normal_count = 0
        reflux_count = 0
        
        for point in self.flow_history:
            if point.flow_direction == "normal":
                normal_count += 1
            elif point.flow_direction == "reflux":
                reflux_count += 1
        
        return {
            "normal": normal_count,
            "reflux": reflux_count,
            "abnormal_ratio": reflux_count / max(reflux_count + normal_count, 1)
        }
    
    def _extract_entry_points(self) -> List[str]:
        """Extract unique entry points (starting veins) from flow history."""
        entry_points = []
        for point in self.flow_history:
            if point.from_type not in entry_points:
                entry_points.append(point.from_type)
        return entry_points
    
    def _extract_exit_points(self) -> List[str]:
        """Extract unique exit points (ending veins) from flow history."""
        exit_points = []
        for point in self.flow_history:
            if point.to_type not in exit_points:
                exit_points.append(point.to_type)
        return exit_points
    
    def get_classified_shunt(self) -> Optional[Dict]:
        """
        Get the detected shunt type based on flow patterns.
        
        Returns:
            Dict with shunt type and pattern info, or None if no shunt detected
        """
        if not self.detected_patterns:
            return None
        
        # Get most recent pattern
        most_recent = self.detected_patterns[-1]
        
        # Match against known patterns
        for shunt_type, pattern_info in self.SHUNT_PATTERNS.items():
            if self._matches_pattern(
                most_recent.pattern_sequence, 
                pattern_info["pattern"]
            ):
                return {
                    "shunt_type": shunt_type,
                    "pattern_sequence": most_recent.pattern_sequence,
                    "is_circular": most_recent.is_circular,
                    "severity": most_recent.severity,
                    "reflux_type": pattern_info["reflux_type"],
                    "description": pattern_info["description"],
                    "entry_point": most_recent.entry_point,
                    "exit_point": most_recent.exit_point
                }
        
        return None
    
    def reset(self) -> None:
        """Reset analyzer state for new session."""
        self.flow_history.clear()
        self.detected_patterns.clear()
        self.current_sequence.clear()
        self.start_time = datetime.now()
        logger.info("Temporal flow analyzer reset")


# Stream processing helper
class FlowSequenceStreamProcessor:
    """Process streaming data points and maintain temporal state."""
    
    def __init__(self):
        self.analyzer = TemporalFlowAnalyzer()
        self.current_task_session = None
    
    def process_stream(self, data_point: Dict) -> Dict:
        """
        Process a single streaming data point.
        
        Returns:
            {
                "status": "processing" | "abnormal_flow_detected" | "complete",
                "flow_summary": {...},
                "abnormal_pattern": {...} (if detected),
                "shunt_classification": {...} (if available)
            }
        """
        result = {
            "status": "processing",
            "flow_summary": None,
            "abnormal_pattern": None,
            "shunt_classification": None
        }
        
        # Add flow point
        abnormal = self.analyzer.add_flow_point(data_point)
        
        # Get current summary
        result["flow_summary"] = self.analyzer.get_flow_summary()
        
        if abnormal:
            # Abnormal pattern detected
            result["status"] = "abnormal_flow_detected"
            result["abnormal_pattern"] = {
                "pattern_sequence": abnormal.pattern_sequence,
                "is_circular": abnormal.is_circular,
                "severity": abnormal.severity,
                "entry_point": abnormal.entry_point,
                "exit_point": abnormal.exit_point,
                "reflux_points": abnormal.reflux_points
            }
            
            # Try to classify shunt
            shunt = self.analyzer.get_classified_shunt()
            if shunt:
                result["shunt_classification"] = shunt
        
        return result
