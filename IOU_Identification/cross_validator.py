"""
Cross-validate CV-ensemble and VLM bounding boxes.
Decision table:
  IoU >= 0.85  → accept CV box (fast, reproducible)          confidence 0.95
  IoU  0.60–0.85 → re-query VLM with CV box as hint          confidence 0.88
  IoU < 0.60   → trust VLM, flag in audit log                confidence 0.60
  Only CV      → use CV box with reduced confidence           confidence 0.75
  Only VLM     → use VLM box with reduced confidence          confidence 0.70
  Neither      → raise RuntimeError
"""

import json
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from cv_ensemble import iou as _iou


AuditEntry = dict


def _append_audit(path: str, entry: AuditEntry) -> None:
    entries: list = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                entries = json.load(f)
        except Exception:
            entries = []
    entries.append(entry)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def validate_roi(
    cv_box: Optional[tuple],
    vlm_box: Optional[tuple],
    frames: list,
    vlm_query_fn: Callable,          # vlm_agent.query_roi signature
    audit_log_path: str = "audit_log.json",
) -> tuple[tuple, str, float]:
    """
    Returns (final_roi, method_label, confidence_score).
    Raises RuntimeError if neither source produced a result.
    """
    entry: AuditEntry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cv_box":    cv_box,
        "vlm_box":   vlm_box,
    }

    def _done(box, method, conf):
        entry.update(result=method, confidence=conf)
        _append_audit(audit_log_path, entry)
        return box, method, conf

    if cv_box is None and vlm_box is None:
        entry["result"] = "FAILED"
        _append_audit(audit_log_path, entry)
        raise RuntimeError(
            "Both CV ensemble and VLM failed to detect a ROI. "
            "Check audit_log.json for details."
        )

    if cv_box is None:
        print("[Validator] CV failed — using VLM box only")
        return _done(vlm_box, "vlm_only", 0.70)

    if vlm_box is None:
        print("[Validator] VLM failed — using CV box only")
        return _done(cv_box, "cv_only", 0.75)

    score = _iou(cv_box, vlm_box)
    entry["iou"] = round(score, 3)
    print(f"[Validator] IoU between CV and VLM: {score:.3f}")

    if score >= 0.85:
        print("[Validator] Strong agreement → accepting CV box")
        return _done(cv_box, "cv+vlm_agree", 0.95)

    if score >= 0.60:
        print("[Validator] Moderate disagreement → re-querying VLM with CV hint")
        corrected = vlm_query_fn(frames[:2], hint_box=cv_box)
        if corrected:
            score2 = _iou(cv_box, corrected)
            print(f"[Validator] After hint, IoU: {score2:.3f}")
            if score2 >= 0.80:
                return _done(cv_box, "cv+vlm_corrected", 0.88)
        # Hint did not help — fall back to original VLM box
        return _done(vlm_box, "vlm_fallback", 0.72)

    # Low agreement
    entry["flag"] = "Low IoU — manual review recommended"
    print(f"[Validator] WARNING: low IoU ({score:.3f}) — using VLM box. "
          "Check audit_log.json")
    return _done(vlm_box, "vlm_low_iou_flagged", 0.60)
