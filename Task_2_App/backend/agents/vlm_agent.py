"""
VLM Frame Analysis Agent.

Thin wrapper around vlm_analyzer.analyze_frame() that exposes a consistent
interface to the guidance orchestrator.  Returns (vlm_dict, summary_text)
so the caller decides how to use each form.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def analyze(
    base64_frame: Optional[str],
    region: str,
    leg: str,
    media_type: str = "image/jpeg",
) -> tuple[Optional[dict], str]:
    """
    Run VLM analysis on a base64-encoded ultrasound frame.

    Args:
        base64_frame:  Base64 JPEG/PNG string, or None if no frame available.
        region:        Current anatomical region ("SFJ", "GSV-THI", etc.).
        leg:           "right" or "left".
        media_type:    MIME type of the image (default "image/jpeg").

    Returns:
        (vlm_dict, summary_text) where vlm_dict is None if no frame provided.
    """
    if not base64_frame:
        return None, "No frame provided."

    try:
        from vlm_analyzer import analyze_frame
        assessment = analyze_frame(base64_frame, region, leg, media_type)
        return assessment.to_dict(), assessment.summary()
    except Exception as exc:
        logger.error("VLM analysis failed: %s", exc)
        return None, f"Frame analysis unavailable: {exc}"
