"""
/api/guidance           — generate active guidance for current probe + context
/api/guidance-regions   — list all guidance regions
/api/guidance-clips/<region> — get passive clips for a region
"""

from __future__ import annotations

import base64
import logging

from flask import Blueprint, jsonify, request

from active_guidance_engine import generate_guidance
from guidance_config import GUIDANCE, get_clips, get_region
from probe_localizer import ProbeLocalizer, ProbeLocation, check_wrong_region
from vlm_analyzer import analyze_frame

guidance_bp = Blueprint("guidance", __name__)
logger = logging.getLogger(__name__)

# Reuse the same per-session localisers defined in localization.py
from routes.localization import _get_localiser


@guidance_bp.route("/api/guidance", methods=["POST"])
def guidance():
    """
    Generate active guidance.

    POST body (JSON):
    {
        "session_id":        str  (optional),

        // Option A – pass a single scanner reading to update localiser:
        "segment_id":        int,
        "segment_dist":      float,
        "is_front":          bool,
        "x":                 int   (optional),
        "y":                 int   (optional),

        // Option B – pass pre-computed location:
        "location": {
            "region": str, "leg": str, "segment_type": str,
            "is_front": bool, "segment_id": int, "segment_dist": float,
            "x": int|null, "y": int|null, "confidence": float
        },

        "expected_region":    str   (optional),
        "current_clip_index": int   (optional, default 0),
        "ep_rp_findings":     dict  (optional — Task-1 output),
        "ultrasound_image":   str   (optional — base64 JPEG/PNG),
        "image_media_type":   str   (optional, default "image/jpeg")
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    session_id = str(data.get("session_id", "default"))

    # --- resolve probe location ---
    loc: ProbeLocation | None = None

    if "location" in data:
        # Pre-computed location passed directly
        ld = data["location"]
        try:
            loc = ProbeLocation(
                region=ld["region"],
                leg=ld["leg"],
                segment_type=ld.get("segment_type", "unknown"),
                is_front=bool(ld["is_front"]),
                segment_id=int(ld["segment_id"]),
                segment_dist=float(ld["segment_dist"]),
                x=ld.get("x"),
                y=ld.get("y"),
                confidence=float(ld.get("confidence", 1.0)),
                stable=bool(ld.get("stable", True)),
                description=ld.get("description", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid location object: {exc}"}), 400

    elif all(k in data for k in ("segment_id", "segment_dist", "is_front")):
        # New scanner reading — update localiser
        try:
            loc = _get_localiser(session_id).update(
                segment_id=int(data["segment_id"]),
                segment_dist=float(data["segment_dist"]),
                is_front=bool(data["is_front"]),
                x=int(data["x"]) if data.get("x") is not None else None,
                y=int(data["y"]) if data.get("y") is not None else None,
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid scanner reading: {exc}"}), 400

    else:
        return jsonify({
            "error": "Provide either a 'location' object or (segment_id, segment_dist, is_front)"
        }), 400

    expected_region = data.get("expected_region")
    current_clip_index = int(data.get("current_clip_index", 0))
    ep_rp_findings = data.get("ep_rp_findings")

    # --- optional VLM analysis ---
    vlm_result = None
    b64_image = data.get("ultrasound_image")
    if b64_image:
        media_type = data.get("image_media_type", "image/jpeg")
        # Strip data-URL prefix if present
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        try:
            vlm_result = analyze_frame(b64_image, loc.region, loc.leg, media_type)
        except Exception as exc:
            logger.error("VLM analysis failed: %s", exc)

    # --- generate active guidance ---
    try:
        result = generate_guidance(
            probe=loc,
            expected_region=expected_region,
            ep_rp_findings=ep_rp_findings,
            vlm_assessment=vlm_result,
            current_clip_index=current_clip_index,
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception("Guidance generation error")
        return jsonify({"error": str(exc)}), 500

    return jsonify(result.to_dict())


@guidance_bp.route("/api/guidance-regions", methods=["GET"])
def guidance_regions():
    """Return metadata for all guidance regions."""
    return jsonify([
        {
            "name": name,
            "full_name": data["full_name"],
            "patient_orientation": data["patient_orientation"],
            "step_count": len(data["steps"]),
            "steps": [
                {"index": i, "description": s}
                for i, s in enumerate(data["steps"])
            ],
        }
        for name, data in GUIDANCE.items()
    ])


@guidance_bp.route("/api/guidance-clips/<region>", methods=["GET"])
def guidance_clips(region: str):
    """Return all clips for a specific region."""
    r = get_region(region)
    if not r:
        return jsonify({"error": f"Unknown region: {region}"}), 404
    return jsonify(r)

@guidance_bp.route("/api/wrong-region", methods=["POST"])
def wrong_region_check():
    """
    Lightweight endpoint: just check if a region is wrong for the expected step.

    POST body: {"current_region": str, "expected_region": str}
    """
    data = request.get_json(force=True, silent=True) or {}
    current = data.get("current_region", "")
    expected = data.get("expected_region")
    result = check_wrong_region(current, expected)
    return jsonify(result.to_dict())