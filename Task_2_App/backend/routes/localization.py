"""
/api/localize  — ingest one scanner reading and return smoothed probe location.
/api/localize/batch — ingest a list of readings and return the stable location.
/api/localize/reset — reset the session localiser window.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from probe_localizer import ProbeLocalizer, classify_single, check_wrong_region

localization_bp = Blueprint("localization", __name__)

# Per-session localisers.  Key = session_id string.
_localisers: dict[str, ProbeLocalizer] = {}


def _get_localiser(session_id: str) -> ProbeLocalizer:
    if session_id not in _localisers:
        _localisers[session_id] = ProbeLocalizer()
    return _localisers[session_id]


@localization_bp.route("/api/localize", methods=["POST"])
def localize():
    """
    POST body (JSON):
    {
        "session_id":    str  (optional, defaults to "default"),
        "segment_id":    int  (0–3),
        "segment_dist":  float (0.0–1.0),
        "is_front":      bool,
        "x":             int  (optional),
        "y":             int  (optional),
        "expected_region": str (optional — for wrong-region check)
    }

    Response:
    {
        "location": { ... ProbeLocation fields ... },
        "wrong_region": { "is_wrong": bool, "reason": str|null, "suggestion": str|null }
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    session_id = str(data.get("session_id", "default"))
    segment_id = data.get("segment_id")
    segment_dist = data.get("segment_dist")
    is_front = data.get("is_front")

    if segment_id is None or segment_dist is None or is_front is None:
        return jsonify({"error": "segment_id, segment_dist, and is_front are required"}), 400

    try:
        segment_id = int(segment_id)
        segment_dist = float(segment_dist)
        is_front = bool(is_front)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid parameter types: {exc}"}), 400

    x = data.get("x")
    y = data.get("y")
    expected_region = data.get("expected_region")

    loc = _get_localiser(session_id).update(
        segment_id=segment_id,
        segment_dist=segment_dist,
        is_front=is_front,
        x=int(x) if x is not None else None,
        y=int(y) if y is not None else None,
    )
    wrong = check_wrong_region(loc.region, expected_region)

    return jsonify({
        "location": loc.to_dict(),
        "wrong_region": wrong.to_dict(),
    })


@localization_bp.route("/api/localize/batch", methods=["POST"])
def localize_batch():
    """
    POST body (JSON):
    {
        "session_id": str,
        "readings": [
            {"segment_id": int, "segment_dist": float, "is_front": bool, "x": int, "y": int},
            ...
        ],
        "expected_region": str (optional)
    }

    Feeds all readings through the localiser window; returns the final stable location.
    """
    data = request.get_json(force=True, silent=True) or {}
    session_id = str(data.get("session_id", "default"))
    readings = data.get("readings", [])
    expected_region = data.get("expected_region")

    if not isinstance(readings, list) or len(readings) == 0:
        return jsonify({"error": "readings must be a non-empty list"}), 400

    localiser = _get_localiser(session_id)
    loc = None
    errors = []

    for i, r in enumerate(readings):
        try:
            loc = localiser.update(
                segment_id=int(r["segment_id"]),
                segment_dist=float(r["segment_dist"]),
                is_front=bool(r["is_front"]),
                x=int(r["x"]) if r.get("x") is not None else None,
                y=int(r["y"]) if r.get("y") is not None else None,
            )
        except Exception as exc:
            errors.append(f"Reading {i}: {exc}")

    if loc is None:
        return jsonify({"error": "No valid readings processed", "details": errors}), 400

    wrong = check_wrong_region(loc.region, expected_region)

    return jsonify({
        "location": loc.to_dict(),
        "wrong_region": wrong.to_dict(),
        "readings_processed": len(readings) - len(errors),
        "errors": errors,
    })


@localization_bp.route("/api/localize/reset", methods=["POST"])
def localize_reset():
    """Reset the sliding window for a session."""
    data = request.get_json(force=True, silent=True) or {}
    session_id = str(data.get("session_id", "default"))
    if session_id in _localisers:
        _localisers[session_id].reset()
    return jsonify({"ok": True, "session_id": session_id})


@localization_bp.route("/api/localize/single", methods=["POST"])
def localize_single():
    """
    Stateless single-reading classification (no smoothing window).

    POST body: same fields as /api/localize minus session_id.
    """
    data = request.get_json(force=True, silent=True) or {}
    try:
        region = classify_single(
            int(data["segment_id"]),
            float(data["segment_dist"]),
            bool(data["is_front"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    from probe_localizer import REGION_FULL_NAMES, SEGMENT_META, get_leg
    leg, seg_type = SEGMENT_META.get(int(data["segment_id"]), ("unknown", "unknown"))

    return jsonify({
        "region": region,
        "region_full_name": REGION_FULL_NAMES.get(region, region),
        "leg": leg,
        "segment_type": seg_type,
    })
