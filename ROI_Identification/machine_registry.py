"""
Optional speed-up layer: cache detected ROI per machine type.

Fingerprint = (resolution, quantised hue-histogram of coloured pixels)
The fingerprint is stable across sessions for the same hardware/UI layout
but does NOT require knowing the machine brand or model.

A new video that fingerprints as a known machine skips Stages 2–3 of
the pipeline entirely (zero CV / VLM cost).
"""

import hashlib
import json
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(_HERE, "machine_registry.json")


# -- Fingerprinting -----------------------------------------------------------

def _fingerprint(frame: np.ndarray, width: int, height: int) -> str:
    """
    Stable, fuzzy fingerprint based on resolution + UI-chrome colour signature.
    Downsamples to 32x32 so content variation across frames is ironed out.
    """
    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)

    # Identify coloured (non-grayscale) pixels — these are UI chrome
    b = small[:, :, 0].astype(np.int16)
    g = small[:, :, 1].astype(np.int16)
    r = small[:, :, 2].astype(np.int16)
    dev = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    colour_mask = dev > 20

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hues = hsv[:, :, 0][colour_mask]

    # 8-bucket hue histogram (quantised to reduce sensitivity to minor colour shifts)
    hist = np.zeros(8, dtype=np.int32)
    if len(hues):
        for h in hues:
            hist[int(h) // 32] += 1
        hist = (hist * 100 // len(hues)).astype(np.int32)  # normalise to % counts

    sig = f"{width}x{height}|{hist.tolist()}"
    return hashlib.md5(sig.encode()).hexdigest()[:14]


# -- Registry I/O -------------------------------------------------------------

def _load() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# -- Public API ---------------------------------------------------------------

def lookup(frame: np.ndarray, width: int, height: int) -> tuple | None:
    """Return cached (x1,y1,x2,y2) for this machine, or None if unseen."""
    registry = _load()
    fp = _fingerprint(frame, width, height)
    entry = registry.get(fp)
    if entry:
        r = entry["roi"]
        print(f"[Registry] Cache hit for fingerprint {fp} -> {r}")
        return (r["x1"], r["y1"], r["x2"], r["y2"])
    return None


def register(
    frame: np.ndarray,
    width: int,
    height: int,
    roi: tuple,
    method: str,
    confidence: float,
    video_name: str,
) -> None:
    """Persist a newly detected ROI for future videos from the same machine."""
    registry = _load()
    fp = _fingerprint(frame, width, height)
    x1, y1, x2, y2 = roi

    existing = registry.get(fp)
    if existing:
        # Accumulate which videos we've verified this fingerprint on
        seen = existing.get("verified_on", [])
        if video_name not in seen:
            seen.append(video_name)
        existing["verified_on"] = seen
        existing["last_confidence"] = round(confidence, 3)
    else:
        registry[fp] = {
            "resolution":       f"{width}x{height}",
            "roi":              {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "detection_method": method,
            "confidence":       round(confidence, 3),
            "verified_on":      [video_name],
        }
        print(f"[Registry] Registered new fingerprint {fp} -> ROI {roi}")

    _save(registry)
