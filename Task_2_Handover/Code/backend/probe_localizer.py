"""
Maps scanner localisation output to named anatomical regions.

Segment IDs (from scanner algorithm):
    0 - left thigh
    1 - left calf
    2 - right thigh
    3 - right calf

Segment distance: 0 = top of segment, 1 = bottom of segment
    Thigh: 0 = groin, 1 = knee
    Calf:  0 = knee,  1 = ankle

is_front:
    True  = scanner facing anterior / medial surface
    False = scanner facing posterior surface

Region mapping:
    SFJ     – Saphenofemoral Junction   (thigh, front, dist 0.00–0.12)
    GSV-THI – GSV Thigh                 (thigh, front, dist 0.12–1.00)
    GSV-CAL – GSV Calf                  (calf,  front, any dist)
    SPJ     – Saphenopopliteal Junction (thigh, back, dist ≥0.78)
                                         (calf,  back, dist ≤0.10)
    SSV     – Small Saphenous Vein      (calf,  back, dist >0.12)
    UNKNOWN – posterior thigh / transition zone
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from config import (
    SLIDING_WINDOW_SIZE,
    STABILITY_THRESHOLD,
    SFJ_THIGH_MAX_DIST,
    SPJ_CALF_MAX_DIST,
    SPJ_THIGH_MIN_DIST,
)

# ── constants ──────────────────────────────────────────────────────────────────

SEGMENT_META: dict[int, tuple[str, str]] = {
    0: ("left", "thigh"),
    1: ("left", "calf"),
    2: ("right", "thigh"),
    3: ("right", "calf"),
}

REGION_FULL_NAMES: dict[str, str] = {
    "SFJ":     "Saphenofemoral Junction",
    "GSV-THI": "Great Saphenous Vein - Thigh",
    "GSV-CAL": "Great Saphenous Vein - Calf",
    "SPJ":     "Saphenopopliteal Junction",
    "SSV":     "Small Saphenous Vein",
    "UNKNOWN": "Unknown / Transition Zone",
}

def get_leg(segment_id: int) -> str:
    """Return 'left' or 'right' for a segment_id."""
    leg, _ = SEGMENT_META.get(segment_id, ("unknown", "unknown"))
    return leg


# Which regions require the patient to be facing away from the examiner
POSTERIOR_REGIONS = {"SPJ", "SSV"}
ANTERIOR_REGIONS  = {"SFJ", "GSV-THI", "GSV-CAL"}

# Standard examination order (used for wrong-region reasoning)
EXAM_ORDER = ["SFJ", "GSV-THI", "GSV-CAL", "SPJ", "SSV"]


# ── data classes ───────────────────────────────────────────────────────────────

@dataclass
class ProbeLocation:
    region: str           # SFJ | GSV-THI | GSV-CAL | SPJ | SSV | UNKNOWN
    leg: str              # "left" | "right"
    segment_type: str     # "thigh" | "calf"
    is_front: bool
    segment_id: int
    segment_dist: float
    x: Optional[int] = None
    y: Optional[int] = None
    confidence: float = 1.0
    stable: bool = True
    description: str = ""

    @property
    def region_full_name(self) -> str:
        return REGION_FULL_NAMES.get(self.region, self.region)

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "region_full_name": self.region_full_name,
            "leg": self.leg,
            "segment_type": self.segment_type,
            "is_front": self.is_front,
            "segment_id": self.segment_id,
            "segment_dist": round(self.segment_dist, 4),
            "x": self.x,
            "y": self.y,
            "confidence": round(self.confidence, 3),
            "stable": self.stable,
            "description": self.description,
        }


@dataclass
class WrongRegionResult:
    is_wrong: bool
    reason: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_wrong": self.is_wrong,
            "reason": self.reason,
            "suggestion": self.suggestion,
        }


# ── pure classification ────────────────────────────────────────────────────────

def classify_single(segment_id: int, segment_dist: float, is_front: bool) -> str:
    """Return region string for a single scanner reading (no smoothing)."""
    if segment_id not in SEGMENT_META:
        return "UNKNOWN"

    _, seg_type = SEGMENT_META[segment_id]

    if seg_type == "thigh":
        if is_front:
            return "SFJ" if segment_dist <= SFJ_THIGH_MAX_DIST else "GSV-THI"
        else:
            return "SPJ" if segment_dist >= SPJ_THIGH_MIN_DIST else "UNKNOWN"

    if seg_type == "calf":
        if is_front:
            return "GSV-CAL"
        else:
            return "SPJ" if segment_dist <= SPJ_CALF_MAX_DIST else "SSV"

    return "UNKNOWN"


def check_wrong_region(
    current_region: str,
    expected_region: Optional[str],
) -> WrongRegionResult:
    """
    Determine whether the probe is in the wrong region for the expected step.
    Returns a WrongRegionResult (is_wrong=False when expected_region is None or matches).
    """
    if not expected_region or expected_region == "ANY":
        return WrongRegionResult(is_wrong=False)

    if current_region == expected_region:
        return WrongRegionResult(is_wrong=False)

    if current_region == "UNKNOWN":
        return WrongRegionResult(
            is_wrong=True,
            reason="Probe is in a non-standard scanning zone.",
            suggestion=f"Reposition probe to {expected_region} ({REGION_FULL_NAMES.get(expected_region, '')}).",
        )

    # Orientation mismatch (patient needs to turn)
    if expected_region in POSTERIOR_REGIONS and current_region in ANTERIOR_REGIONS:
        return WrongRegionResult(
            is_wrong=True,
            reason=(
                f"Probe is scanning anterior anatomy ({current_region}) but the protocol "
                f"requires posterior examination ({expected_region}). "
                "The patient needs to turn away from the examiner."
            ),
            suggestion="Ask the patient to turn around, then reposition probe to popliteal area.",
        )

    if expected_region in ANTERIOR_REGIONS and current_region in POSTERIOR_REGIONS:
        return WrongRegionResult(
            is_wrong=True,
            reason=(
                f"Probe is scanning posterior anatomy ({current_region}) but the protocol "
                f"requires anterior examination ({expected_region}). "
                "Patient should face the examiner."
            ),
            suggestion="Ask the patient to face you, then reposition probe.",
        )

    # Same orientation, wrong region
    return WrongRegionResult(
        is_wrong=True,
        reason=f"Probe detected at {current_region} ({REGION_FULL_NAMES.get(current_region, '')}) "
               f"but expected at {expected_region} ({REGION_FULL_NAMES.get(expected_region, '')}).",
        suggestion=f"Move probe to {expected_region} ({REGION_FULL_NAMES.get(expected_region, '')}).",
    )


# ── stateful localiser ─────────────────────────────────────────────────────────

class ProbeLocalizer:
    """
    Stateful localiser for a single examination session.
    Applies a sliding window majority vote for region stability.
    """

    def __init__(self, window_size: int = SLIDING_WINDOW_SIZE) -> None:
        self._window: deque[str] = deque(maxlen=window_size)
        self._window_size = window_size

    # ── public ────────────────────────────────────────────────────────────────

    def update(
        self,
        segment_id: int,
        segment_dist: float,
        is_front: bool,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> ProbeLocation:
        """
        Ingest one scanner reading and return the smoothed probe location.
        """
        raw_region = classify_single(segment_id, segment_dist, is_front)
        self._window.append(raw_region)

        stable_region, confidence, is_stable = self._smooth()

        leg, seg_type = SEGMENT_META.get(segment_id, ("unknown", "unknown"))
        desc = _build_description(stable_region, leg, seg_type, is_front, segment_dist)

        return ProbeLocation(
            region=stable_region,
            leg=leg,
            segment_type=seg_type,
            is_front=is_front,
            segment_id=segment_id,
            segment_dist=segment_dist,
            x=x,
            y=y,
            confidence=confidence,
            stable=is_stable,
            description=desc,
        )

    def reset(self) -> None:
        self._window.clear()

    @property
    def window_snapshot(self) -> list[str]:
        return list(self._window)

    # ── internal ──────────────────────────────────────────────────────────────

    def _smooth(self) -> tuple[str, float, bool]:
        """Return (region, confidence, is_stable) from the current window."""
        if not self._window:
            return "UNKNOWN", 0.0, False

        counts = Counter(self._window)
        top_region, top_count = counts.most_common(1)[0]
        confidence = top_count / len(self._window)
        is_stable = confidence >= STABILITY_THRESHOLD
        return top_region, confidence, is_stable


# ── formatting helpers ─────────────────────────────────────────────────────────

def _build_description(
    region: str,
    leg: str,
    seg_type: str,
    is_front: bool,
    segment_dist: float,
) -> str:
    side = "front" if is_front else "back"
    pct = f"{segment_dist * 100:.0f}%"
    full = REGION_FULL_NAMES.get(region, region)
    return f"{leg.capitalize()} {seg_type}, {side}, {pct} along segment - {full}"
