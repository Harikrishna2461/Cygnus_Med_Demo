"""
Passive guidance protocol steps for each anatomical region.
"""

from __future__ import annotations
from typing import Optional

GUIDANCE: dict[str, dict] = {
    "SFJ": {
        "full_name": "Saphenofemoral Junction",
        "patient_orientation": "facing",
        "steps": [
            "Patient standing upright, both legs naturally apart.",
            "Observe the anatomical structure of the SFJ.",
            "Place probe steadily at medial groin to locate SFJ.",
            "Switch to color Doppler.",
            "Compress and release calf.",
            "If sustained reverse flow appears, capture a clip.",
        ],
    },
    "GSV-THI": {
        "full_name": "Great Saphenous Vein - Thigh",
        "patient_orientation": "facing",
        "steps": [
            "Scan distally along GSV (centred) until above the knee.",
            "Pause at suspicious branches, observe longitudinally.",
            "Switch to color Doppler.",
            "Compress and observe flow.",
            "If sustained reverse flow appears, capture a clip.",
        ],
    },
    "GSV-CAL": {
        "full_name": "Great Saphenous Vein - Calf",
        "patient_orientation": "facing",
        "steps": [
            "Scan slowly across knee and calf veins towards ankle.",
            "Pause at suspicious locations, observe longitudinally.",
            "Switch to color Doppler.",
            "Compress and release calf.",
            "If sustained reverse flow appears, capture a clip.",
        ],
    },
    "SPJ": {
        "full_name": "Saphenopopliteal Junction",
        "patient_orientation": "turned",
        "steps": [
            "Patient turns around and adjusts standing position.",
            "Scan popliteal veins.",
            "Pause at suspicious locations, observe longitudinally.",
            "Switch to color Doppler.",
            "Compress and release calf.",
            "If sustained reverse flow appears, capture a clip.",
        ],
    },
    "SSV": {
        "full_name": "Small Saphenous Vein",
        "patient_orientation": "turned",
        "steps": [
            "Scan along the SSV.",
            "Pause at suspicious locations.",
            "Switch to color Doppler.",
            "Compress and release calf.",
            "If sustained reverse flow appears, capture a clip.",
            "Scan along the Giacomini vein.",
            "Scan complete. Press Assess for analysis.",
        ],
    },
}

REGION_ORDER = ["SFJ", "GSV-THI", "GSV-CAL", "SPJ", "SSV"]


def get_region(name: str) -> Optional[dict]:
    return GUIDANCE.get(name)


def get_steps(name: str) -> list[str]:
    r = GUIDANCE.get(name)
    return r["steps"] if r else []


def get_clips(name: str) -> list[dict]:
    """Return steps as clip-like dicts for API compatibility."""
    return [{"index": i, "subtitle_en": s} for i, s in enumerate(get_steps(name))]


def passive_guidance_summary(name: str) -> str:
    steps = get_steps(name)
    if not steps:
        return "No guidance available for this region."
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))


def patient_orientation(name: str) -> str:
    r = GUIDANCE.get(name)
    return r["patient_orientation"] if r else "unknown"
