"""
Ultrasound Vein Detector - VLM-based Vein Classification

Calls a Vision Language Model via the HuggingFace Inference API for
ultrasound image understanding and vein classification relative to fascia.
Uses GLM-4.5V by default (EchoVLM is not available via serverless inference).

Classifies veins as:
  - N1 (Deep): below fascia
  - N2 (GSV): within/near fascia
  - N3 (Superficial): above fascia
"""

import base64
import logging
import os
import re
import cv2
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VeinTracker:
    """Tracks veins across video frames using centroid proximity.

    Once a vein is first classified (N1/N2/N3), its classification is locked
    so it stays consistent throughout the video. Veins are matched frame-to-frame
    by finding the nearest centroid within a distance threshold.
    """

    def __init__(self, max_distance: float = 60, missing_tolerance: int = 8):
        """
        Args:
            max_distance: Max pixel distance between centroids to consider a match.
            missing_tolerance: Number of consecutive frames a vein can be missing
                               before it's dropped from tracking.
        """
        self.max_distance = max_distance
        self.missing_tolerance = missing_tolerance
        self.tracks = {}  # track_id -> {center, classification, vein_type, color, last_seen, confidence}
        self.next_id = 0

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """Match new detections to existing tracks, lock classifications.

        Args:
            detections: List of detection dicts from _classify_blobs
            frame_idx: Current frame index (for aging out lost tracks)

        Returns:
            Updated detections with stable track_id and locked classification.
        """
        if not detections and not self.tracks:
            return detections

        # Remove stale tracks
        stale = [tid for tid, t in self.tracks.items()
                 if (frame_idx - t["last_seen"]) > self.missing_tolerance]
        for tid in stale:
            del self.tracks[tid]

        if not detections:
            return detections

        # Build cost matrix: distance from each detection to each track
        track_ids = list(self.tracks.keys())
        matched_det = set()
        matched_trk = set()

        if track_ids:
            import numpy as np
            det_centers = np.array([d["center"] for d in detections])
            trk_centers = np.array([self.tracks[tid]["center"] for tid in track_ids])

            # Pairwise distances
            dists = np.linalg.norm(det_centers[:, None, :] - trk_centers[None, :, :], axis=2)

            # Greedy matching: pick closest pairs first
            while True:
                if dists.size == 0:
                    break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                min_dist = dists[min_idx]
                if min_dist > self.max_distance:
                    break
                di, ti = min_idx
                if di in matched_det or ti in matched_trk:
                    dists[di, ti] = float('inf')
                    continue

                # Match: lock classification from the track
                tid = track_ids[ti]
                track = self.tracks[tid]
                detections[di]["track_id"] = tid
                detections[di]["classification"] = track["classification"]
                detections[di]["vein_type"] = track["vein_type"]
                detections[di]["color"] = track["color"]

                # Update track position (smoothed)
                alpha = 0.6  # weight toward new position
                old_c = track["center"]
                new_c = detections[di]["center"]
                track["center"] = [
                    alpha * new_c[0] + (1 - alpha) * old_c[0],
                    alpha * new_c[1] + (1 - alpha) * old_c[1],
                ]
                track["last_seen"] = frame_idx

                matched_det.add(di)
                matched_trk.add(ti)
                dists[di, :] = float('inf')
                dists[:, ti] = float('inf')

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det:
                tid = self.next_id
                self.next_id += 1
                det["track_id"] = tid
                self.tracks[tid] = {
                    "center": list(det["center"]),
                    "classification": det["classification"],
                    "vein_type": det["vein_type"],
                    "color": det["color"],
                    "confidence": det["confidence"],
                    "last_seen": frame_idx,
                }

        return detections

    def reset(self):
        """Reset all tracks (e.g., for a new video)."""
        self.tracks.clear()
        self.next_id = 0


class UltrasoundVeinDetector:
    """Detects and classifies veins in ultrasound images using a VLM via HF Inference API."""

    # EchoVLM is not available on serverless inference, so we default to GLM-4.5V
    DEFAULT_MODEL = "zai-org/GLM-4.5V"

    def __init__(self, model_name: str = None, **kwargs):
        from huggingface_hub import InferenceClient

        self.model_name = model_name or self.DEFAULT_MODEL
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or None  # Set via environment variable
        )
        self.client = InferenceClient(model=self.model_name, token=hf_token)
        logger.info(f"UltrasoundVeinDetector initialized (API mode: {self.model_name})")

    def _extract_ultrasound_roi(self, frame: np.ndarray) -> tuple:
        """Extract the actual ultrasound image region from machine UI frames.

        Ultrasound machines display the scan in a bright rectangular region
        surrounded by dark UI borders with text/controls. This method finds
        that region.

        Returns:
            (roi_frame, x_offset, y_offset) — cropped frame and its offset
            in the original. If no ROI found, returns the original frame with (0,0).
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold: ultrasound region is generally brighter than the black UI border
        # Use a low threshold to separate the scan area from the dark surround
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find the largest contour (the ultrasound image area)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, 0, 0

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Must be at least 15% of the frame to be the scan area
        if area < h * w * 0.15:
            return frame, 0, 0

        x, y, rw, rh = cv2.boundingRect(largest)

        # Add small padding inward to exclude edge artifacts
        pad = 5
        x = min(x + pad, w - 1)
        y = min(y + pad, h - 1)
        rw = max(rw - 2 * pad, 50)
        rh = max(rh - 2 * pad, 50)

        roi = frame[y:y+rh, x:x+rw]
        return roi, x, y

    def detect_and_classify_frame_fast(
        self,
        frame: np.ndarray,
        fascia_center_y: Optional[int] = None,
        tracker: Optional['VeinTracker'] = None,
        frame_idx: int = 0,
    ) -> Dict:
        """Fast CV-only detection — no VLM call. For video frame processing.

        Args:
            tracker: Optional VeinTracker for cross-frame tracking. When provided,
                     vein IDs and classifications are locked across frames.
            frame_idx: Current frame index (used by tracker for aging).
        """
        original_frame = frame
        roi, x_off, y_off = self._extract_ultrasound_roi(frame)
        frame = roi
        height, width = frame.shape[:2]
        dark_blobs = self._find_dark_blobs_cv(frame)

        if fascia_center_y is not None:
            margin = int(height * 0.04)
            fascia_bounds = (fascia_center_y - margin, fascia_center_y + margin)
        else:
            fascia_bounds = self._detect_fascia_y(frame)

        detections = self._classify_blobs(dark_blobs, fascia_bounds, height, width)

        # Map detections and fascia back to original frame coordinates
        if x_off != 0 or y_off != 0:
            for d in detections:
                d["center"] = [d["center"][0] + x_off, d["center"][1] + y_off]
                d["bbox"] = [d["bbox"][0] + x_off, d["bbox"][1] + y_off,
                             d["bbox"][2] + x_off, d["bbox"][3] + y_off]
            fascia_bounds = (fascia_bounds[0] + y_off, fascia_bounds[1] + y_off)

        # Apply cross-frame tracking (locks classification after first detection)
        if tracker is not None:
            detections = tracker.update(detections, frame_idx)

        annotated = self._annotate_frame(original_frame.copy(), detections, fascia_bounds)

        return {
            "status": "success" if detections else "no_veins",
            "detections": detections,
            "annotated_frame": annotated,
            "frame_shape": original_frame.shape,
            "num_veins": len(detections),
            "fascia_bounds": (int(fascia_bounds[0]), int(fascia_bounds[1])),
        }

    def detect_and_classify_frame(
        self,
        frame: np.ndarray,
        fascia_center_y: Optional[int] = None,
    ) -> Dict:
        """
        Detect and classify veins in an ultrasound frame using EchoVLM via API.

        Args:
            frame: Input BGR image (numpy array)
            fascia_center_y: Optional hint for fascia Y position

        Returns:
            Dict with detections, annotated_frame, status, num_veins
        """
        original_frame = frame
        roi, x_off, y_off = self._extract_ultrasound_roi(frame)
        frame = roi
        height, width = frame.shape[:2]

        # Step 1: Use OpenCV to find dark blobs (exact pixel coordinates)
        dark_blobs = self._find_dark_blobs_cv(frame)
        logger.info(f"CV found {len(dark_blobs)} dark blob candidates")

        if not dark_blobs:
            return {
                "status": "no_veins",
                "detections": [],
                "annotated_frame": self._annotate_frame(frame.copy(), []),
                "frame_shape": frame.shape,
                "num_veins": 0,
            }

        # Step 2: Detect fascia (yellow line) upper/lower boundaries
        if fascia_center_y is not None:
            margin = int(height * 0.04)
            fascia_bounds = (fascia_center_y - margin, fascia_center_y + margin)
        else:
            fascia_bounds = self._detect_fascia_y(frame)
        logger.info(f"Fascia bounds: upper={fascia_bounds[0]}, lower={fascia_bounds[1]}")

        # Step 3: Use VLM to validate — ask it to describe dark shapes
        _, buffer = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_b64}"

        vlm_confirmed = True
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": (
                                "Describe the dark round shapes in this image. "
                                "How many are there? Where is each one located "
                                "(top/middle/bottom, left/center/right)? "
                                "How large is each relative to the image?"
                            )},
                        ],
                    },
                ],
                max_tokens=500,
                temperature=0.2,
            )
            vlm_text = response.choices[0].message.content or ""
            logger.info(f"VLM description: {repr(vlm_text)}")

            # VLM tends to only count the most prominent shapes, so use it
            # as a floor — keep at least VLM count, but allow 1 extra from CV
            # to catch subtler shapes the VLM misses
            vlm_count = self._extract_count(vlm_text)
            if vlm_count is not None and vlm_count > 0:
                max_keep = vlm_count + 1
                if len(dark_blobs) > max_keep:
                    dark_blobs = dark_blobs[:max_keep]
                    logger.info(f"VLM saw {vlm_count}, keeping top {max_keep} CV candidates")
            elif vlm_count == 0:
                dark_blobs = []
                logger.info("VLM says no dark shapes — clearing candidates")
        except Exception as e:
            logger.warning(f"VLM validation call failed (using CV results only): {e}")

        # Step 4: Classify each blob relative to fascia
        detections = self._classify_blobs(dark_blobs, fascia_bounds, height, width)
        logger.info(f"Classified {len(detections)} veins")

        # Map back to original frame coordinates
        if x_off != 0 or y_off != 0:
            for d in detections:
                d["center"] = [d["center"][0] + x_off, d["center"][1] + y_off]
                d["bbox"] = [d["bbox"][0] + x_off, d["bbox"][1] + y_off,
                             d["bbox"][2] + x_off, d["bbox"][3] + y_off]
            fascia_bounds = (fascia_bounds[0] + y_off, fascia_bounds[1] + y_off)

        # Annotate on original frame
        annotated = self._annotate_frame(original_frame.copy(), detections, fascia_bounds)

        return {
            "status": "success" if detections else "no_veins",
            "detections": detections,
            "annotated_frame": annotated,
            "frame_shape": frame.shape,
            "num_veins": len(detections),
        }

    def _find_dark_blobs_cv(self, frame: np.ndarray) -> List[Dict]:
        """Use OpenCV to find dark circular/oval blobs — returns sorted by confidence."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Exclude colored annotations (yellow fascia lines, text overlays)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, np.array([10, 50, 100]), np.array([45, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 50, 100]), np.array([85, 255, 255]))
        exclude = cv2.bitwise_or(color_mask, green_mask)
        exclude = cv2.dilate(exclude, np.ones((15, 15), np.uint8))
        # Also exclude bright regions
        bright = cv2.inRange(gray, 200, 255)
        exclude = cv2.bitwise_or(exclude, cv2.dilate(bright, np.ones((11, 11), np.uint8)))

        # Adaptive dark threshold
        valid_pixels = gray[exclude == 0]
        if len(valid_pixels) == 0:
            return []
        dark_thresh = np.percentile(valid_pixels, 18)
        dark_mask = ((gray < dark_thresh) & (exclude == 0)).astype(np.uint8) * 255

        # Remove borders (generous to exclude edge artifacts)
        border = int(min(height, width) * 0.06)
        dark_mask[:border, :] = 0
        dark_mask[-border:, :] = 0
        dark_mask[:, :border] = 0
        dark_mask[:, -border:] = 0

        # Morphological cleanup
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=3)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = height * width
        min_area = frame_area * 0.002
        max_area = frame_area * 0.12

        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area or len(contour) < 5:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 3.0:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.15:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Contrast check
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(roi_mask, [contour], -1, 255, -1)
            mean_int = cv2.mean(gray, mask=roi_mask)[0]
            surr_mask = cv2.dilate(roi_mask, np.ones((15, 15), np.uint8)) - roi_mask
            surr_int = cv2.mean(gray, mask=surr_mask)[0] if cv2.countNonZero(surr_mask) > 0 else mean_int
            contrast = (surr_int - mean_int) / max(surr_int, 1)
            if contrast < 0.10:
                continue

            confidence = min(0.95, 0.3 + circularity * 0.25 + contrast * 0.3 + 0.15)
            blobs.append({
                "center": [float(cx), float(cy)],
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "width": float(w), "height": float(h),
                "confidence": float(confidence),
            })

        blobs.sort(key=lambda b: b["confidence"], reverse=True)
        return blobs

    def _detect_fascia_y(self, frame: np.ndarray) -> tuple:
        """Detect fascia boundaries in an ultrasound image.

        Tries two strategies:
        1. Yellow annotation detection (for annotated teaching images)
        2. Hyperechoic band detection (for raw ultrasound — fascia appears
           as a bright horizontal echogenic layer in B-mode)

        Medical context: In transverse ultrasound of the leg, the saphenous
        fascia and muscular fascia appear as bright (hyperechoic) horizontal
        bands. The saphenous vein sits between them ("Egyptian eye" sign).

        Returns:
            (upper_y, lower_y) tuple
        """
        h, w = frame.shape[:2]

        # --- Strategy 1: Yellow annotation (annotated images) ---
        result = self._detect_fascia_yellow(frame)
        if result is not None:
            logger.debug("Fascia detected via yellow annotation")
            return result

        # --- Strategy 2: Hyperechoic band detection (raw ultrasound) ---
        result = self._detect_fascia_hyperechoic(frame)
        if result is not None:
            logger.debug("Fascia detected via hyperechoic band")
            return result

        # Fallback: middle of image
        margin = int(h * 0.04)
        logger.debug("Fascia detection fallback: using image center")
        return (h // 2 - margin, h // 2 + margin)

    def _detect_fascia_yellow(self, frame: np.ndarray) -> Optional[tuple]:
        """Detect yellow fascia annotations (for annotated teaching images)."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([15, 80, 150]), np.array([35, 255, 255]))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)), iterations=2)

        # Keep only wide connected components (real fascia lines span the image)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(yellow, connectivity=8)
        min_width = w * 0.20
        fascia_mask = np.zeros_like(yellow)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_WIDTH] >= min_width and stats[i, cv2.CC_STAT_AREA] > 50:
                fascia_mask[labels == i] = 255

        uppers, lowers = [], []
        for col in range(w):
            col_ys = np.where(fascia_mask[:, col] > 0)[0]
            if len(col_ys) > 0:
                uppers.append(col_ys.min())
                lowers.append(col_ys.max())

        if len(uppers) > int(w * 0.1):
            upper_y = int(np.median(uppers))
            lower_y = int(np.median(lowers))
            if (lower_y - upper_y) > h * 0.6:
                center = (upper_y + lower_y) // 2
                margin = int(h * 0.04)
                return (center - margin, center + margin)
            return (upper_y, lower_y)

        return None

    def _detect_fascia_hyperechoic(self, frame: np.ndarray) -> Optional[tuple]:
        """Detect fascia as a bright horizontal band in raw B-mode ultrasound.

        In B-mode ultrasound of the leg:
        - Fascia appears as hyperechoic (bright) continuous horizontal lines
        - There are typically two layers: superficial fascia and deep/muscular fascia
        - The saphenous compartment sits between them ("Egyptian eye" sign)

        We combine two signals:
        1. Row-wise mean intensity (bright rows = candidate fascia)
        2. Vertical gradient magnitude (fascia has sharp bright-dark transitions)
        3. Horizontal continuity check (fascia spans most of the image width)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Signal 1: Horizontal line enhancement ---
        # Morphological closing with wide horizontal kernel enhances horizontal structures
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 10), 1))
        horiz_enhanced = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, horiz_kernel)

        # --- Signal 2: Vertical gradient (Sobel-Y) to detect bright-dark boundaries ---
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_mag = np.abs(sobel_y)
        # Row-wise mean of gradient magnitude — fascia rows have strong vertical edges
        grad_row_means = np.mean(gradient_mag, axis=1).astype(np.float32)

        # Row-wise mean intensity from horizontal-enhanced image
        row_means = np.mean(horiz_enhanced, axis=1).astype(np.float32)

        # Smooth both profiles
        from scipy.ndimage import uniform_filter1d
        smooth_size = max(5, h // 40)
        row_smooth = uniform_filter1d(row_means, size=smooth_size)
        grad_smooth = uniform_filter1d(grad_row_means, size=smooth_size)

        # Normalize both to 0-1
        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-6)

        row_norm = normalize(row_smooth)
        grad_norm = normalize(grad_smooth)

        # Combined score: bright rows with strong vertical gradient
        combined = 0.6 * row_norm + 0.4 * grad_norm

        # Search in the middle portion of the image (fascia is not at very top/bottom)
        search_top = int(h * 0.15)
        search_bot = int(h * 0.85)
        search_region = combined[search_top:search_bot]

        if len(search_region) == 0:
            return None

        # Adaptive threshold: take the top 20% of combined scores
        threshold = np.percentile(search_region, 80)

        # Find rows that exceed the threshold
        bright_rows = np.where(search_region > threshold)[0] + search_top

        if len(bright_rows) < 3:
            return None

        # Cluster the bright rows into bands (groups of consecutive or near rows)
        gap_threshold = max(8, h // 30)
        bands = []
        current_band = [bright_rows[0]]
        for i in range(1, len(bright_rows)):
            if bright_rows[i] - bright_rows[i-1] <= gap_threshold:
                current_band.append(bright_rows[i])
            else:
                if len(current_band) >= 2:
                    bands.append(current_band)
                current_band = [bright_rows[i]]
        if len(current_band) >= 2:
            bands.append(current_band)

        if not bands:
            return None

        # Score each band: intensity * gradient * horizontal extent
        scored_bands = []
        for band in bands:
            band_intensity = np.mean(row_smooth[band])
            band_gradient = np.mean(grad_smooth[band])
            horiz_extent = self._check_band_horizontal_extent(gray, band, w)

            # Skip bands that don't span enough width (likely artifacts)
            if horiz_extent < 0.3:
                continue

            score = band_intensity * (1 + band_gradient) * horiz_extent
            scored_bands.append((score, band))

        if not scored_bands:
            return None

        scored_bands.sort(key=lambda x: x[0], reverse=True)

        # If we found 2+ qualifying bands, use the top two as upper and lower fascia
        if len(scored_bands) >= 2:
            band_a = scored_bands[0][1]
            band_b = scored_bands[1][1]
            y_a = (min(band_a) + max(band_a)) // 2
            y_b = (min(band_b) + max(band_b)) // 2
            upper_y = min(y_a, y_b)
            lower_y = max(y_a, y_b)

            # Sanity check: the two bands shouldn't be too far apart
            # (fascia layers are typically within ~30% of image height of each other)
            if (lower_y - upper_y) > h * 0.45:
                # Too far apart — just use the best band
                best_band = scored_bands[0][1]
                upper_y = int(min(best_band))
                lower_y = int(max(best_band))
            else:
                logger.debug(f"Found dual fascia bands at y={upper_y} and y={lower_y}")
                return (upper_y, lower_y)

        else:
            best_band = scored_bands[0][1]
            upper_y = int(min(best_band))
            lower_y = int(max(best_band))

        # If the band is very thin, add a margin
        margin = int(h * 0.04)
        if (lower_y - upper_y) < margin:
            center = (upper_y + lower_y) // 2
            upper_y = center - margin
            lower_y = center + margin

        return (max(0, upper_y), min(h - 1, lower_y))

    def _check_band_horizontal_extent(self, gray: np.ndarray, band_rows: list, width: int) -> float:
        """Check what fraction of image width a bright band spans.

        Returns a score 0-1 indicating horizontal extent. Real fascia spans
        most of the ultrasound width; artifacts are typically localized.
        """
        band_strip = gray[min(band_rows):max(band_rows)+1, :]
        col_means = np.mean(band_strip, axis=0)

        # Use adaptive threshold: columns brighter than the row's overall mean
        overall_mean = np.mean(col_means)
        bright_cols = np.sum(col_means > overall_mean * 0.85)

        # Real fascia should span at least 40% of width for a good score
        extent = bright_cols / width
        return min(1.0, extent / 0.5)  # 50% width = score 1.0

    def _extract_count(self, vlm_text: str) -> Optional[int]:
        """Extract number of dark shapes from VLM natural language response."""
        # Look for patterns like "1 dark", "2 dark", "there is 1", "there are 3"
        text_lower = vlm_text.lower()
        for pattern in [r"(\d+)\s+dark", r"there\s+(?:is|are)\s+(\d+)", r"(\d+)\s+(?:round|circular|oval)"]:
            m = re.search(pattern, text_lower)
            if m:
                return int(m.group(1))
        # Word numbers
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "no": 0, "zero": 0}
        for word, num in word_map.items():
            if word in text_lower:
                return num
        return None

    def _classify_blobs(self, blobs: List[Dict], fascia_bounds: tuple, height: int, width: int) -> List[Dict]:
        """Classify blobs as N1/N2/N3 based on position relative to fascia boundaries."""
        classification_map = {
            "N1": ("N1_deep", "below_fascia", (0, 255, 0)),
            "N2": ("N2_gsv", "within_fascia", (255, 0, 255)),
            "N3": ("N3_superficial", "above_fascia", (0, 165, 255)),
        }
        upper_y, lower_y = fascia_bounds
        detections = []
        for i, blob in enumerate(blobs):
            cy = blob["center"][1]
            if cy < upper_y:
                cls = "N3"
                dist = float(upper_y - cy)
            elif cy > lower_y:
                cls = "N1"
                dist = float(cy - lower_y)
            else:
                cls = "N2"
                dist = 0.0
            vein_type, position, color = classification_map[cls]
            detections.append({
                "id": i, "blob_id": i,
                "classification": cls, "vein_type": vein_type,
                "position": position, "confidence": blob["confidence"],
                "center": blob["center"], "bbox": blob["bbox"],
                "width": blob["width"], "height": blob["height"],
                "radius": max(blob["width"], blob["height"]) / 2.0,
                "distance_to_fascia": dist, "fascia_distance": dist,
                "in_fascia_region": cls == "N2", "color": color,
            })
        return detections

    def _parse_detections(
        self, model_output: str, height: int, width: int
    ) -> List[Dict]:
        """Parse structured vein detections from VLM text output (fallback)."""
        detections = []
        classification_map = {
            "N1": ("N1_deep", "below_fascia", (0, 255, 0)),       # Green
            "N2": ("N2_gsv", "within_fascia", (255, 0, 255)),     # Magenta
            "N3": ("N3_superficial", "above_fascia", (0, 165, 255)),  # Orange
        }

        # Match lines like: VEIN 100 200 150 280 N2 0.85
        pattern = r"VEIN\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(N[123])\s+([\d.]+)"
        matches = re.findall(pattern, model_output)

        for i, match in enumerate(matches):
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            classification = match[4]
            confidence = float(match[5])

            # Clamp to image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            confidence = max(0.0, min(1.0, confidence))
            vein_type, position, color = classification_map.get(
                classification, ("N2_gsv", "within_fascia", (255, 0, 255))
            )

            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append({
                "id": i,
                "blob_id": i,
                "classification": classification,
                "vein_type": vein_type,
                "position": position,
                "confidence": confidence,
                "center": [cx, cy],
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "width": float(w),
                "height": float(h),
                "radius": max(w, h) / 2.0,
                "distance_to_fascia": 0.0,
                "fascia_distance": 0.0,
                "in_fascia_region": classification == "N2",
                "color": color,
            })

        return detections

    def _annotate_frame(
        self, frame: np.ndarray, classified_veins: List[Dict],
        fascia_bounds: Optional[tuple] = None,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, fascia lines, and legend on the frame."""
        width = frame.shape[1]

        # Draw fascia boundary lines
        if fascia_bounds:
            upper_y, lower_y = fascia_bounds
            cv2.line(frame, (0, upper_y), (width, upper_y), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (0, lower_y), (width, lower_y), (0, 255, 255), 1, cv2.LINE_AA)
            # Label
            cv2.putText(frame, "Fascia", (width - 70, upper_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        for vein in classified_veins:
            x1, y1, x2, y2 = [int(v) for v in vein["bbox"]]
            color = tuple(vein["color"])
            label = f"{vein['classification']} ({vein['confidence']:.0%})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        # Legend
        legend_items = [
            ("N3 - Superficial", (0, 165, 255)),
            ("N2 - GSV", (255, 0, 255)),
            ("N1 - Deep", (0, 255, 0)),
        ]
        y_off = 25
        for text, color in legend_items:
            cv2.putText(frame, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_off += 22

        return frame


# # ==========================================================================
# # ORIGINAL CV-BASED IMPLEMENTATION (commented out for revert if needed)
# # ==========================================================================
# #
# # """
# # Ultrasound Vein Detector - Fascia-Aware Vein Classification
# #
# # Strategy:
# # 1. Detect the yellow/colored fascia annotation via HSV color filtering
# # 2. Detect dark blob structures (veins) via intensity-based detection
# # 3. Classify veins as N1/N2/N3 based on position relative to fascia contour
# # 4. Annotate output with colored bounding boxes and labels
# # """
# #
# # import logging
# # import cv2
# # import numpy as np
# # from typing import Dict, List, Optional, Tuple
# #
# # logger = logging.getLogger(__name__)
# #
# #
# # class UltrasoundVeinDetector:
# #     """Detects and classifies veins relative to annotated fascia in ultrasound images."""
# #
# #     FASCIA_MARGIN_RATIO = 0.04
# #
# #     FASCIA_HSV_RANGES = [
# #         ((15, 80, 150), (35, 255, 255)),
# #         ((25, 100, 200), (45, 255, 255)),
# #     ]
# #
# #     MIN_VEIN_AREA_RATIO = 0.003
# #     MAX_VEIN_AREA_RATIO = 0.05
# #     MIN_CIRCULARITY = 0.55
# #     MAX_ASPECT_RATIO = 2.0
# #     DARK_THRESHOLD_PERCENTILE = 18
# #     MIN_CONTRAST_RATIO = 0.25
# #     MIN_SOLIDITY = 0.65
# #
# #     def __init__(self, **kwargs):
# #         logger.info("UltrasoundVeinDetector initialized")
# #
# #     def detect_and_classify_frame(
# #         self,
# #         frame: np.ndarray,
# #         fascia_center_y: Optional[int] = None
# #     ) -> Dict:
# #         height, width = frame.shape[:2]
# #         fascia_mask, fascia_upper_y, fascia_lower_y = self._detect_fascia_annotation(frame)
# #
# #         if fascia_upper_y is None and fascia_center_y is not None:
# #             margin = int(height * self.FASCIA_MARGIN_RATIO)
# #             fascia_upper_y = np.full(width, fascia_center_y - margin, dtype=np.float32)
# #             fascia_lower_y = np.full(width, fascia_center_y + margin, dtype=np.float32)
# #             fascia_mask = np.zeros((height, width), dtype=np.uint8)
# #             y = fascia_center_y
# #             fascia_mask[max(0, y - 5):min(height, y + 5), :] = 255
# #
# #         if fascia_upper_y is None:
# #             margin = int(height * self.FASCIA_MARGIN_RATIO)
# #             fascia_upper_y = np.full(width, height // 2 - margin, dtype=np.float32)
# #             fascia_lower_y = np.full(width, height // 2 + margin, dtype=np.float32)
# #             fascia_mask = np.zeros((height, width), dtype=np.uint8)
# #
# #         vein_candidates = self._detect_dark_blobs(frame, fascia_mask)
# #
# #         classified = []
# #         for i, vein in enumerate(vein_candidates):
# #             cx, cy = vein["center"]
# #             x_idx = max(0, min(int(cx), width - 1))
# #             local_upper = fascia_upper_y[x_idx]
# #             local_lower = fascia_lower_y[x_idx]
# #
# #             if cy < local_upper:
# #                 classification = "N3"
# #                 vein_type = "N3_superficial"
# #                 position = "above_fascia"
# #                 color = (0, 165, 255)
# #                 dist_to_fascia = float(local_upper - cy)
# #             elif cy > local_lower:
# #                 classification = "N1"
# #                 vein_type = "N1_deep"
# #                 position = "below_fascia"
# #                 color = (0, 255, 0)
# #                 dist_to_fascia = float(cy - local_lower)
# #             else:
# #                 classification = "N2"
# #                 vein_type = "N2_gsv"
# #                 position = "within_fascia"
# #                 color = (255, 0, 255)
# #                 dist_to_fascia = 0.0
# #
# #             classified.append({
# #                 "id": i,
# #                 "blob_id": i,
# #                 "classification": classification,
# #                 "vein_type": vein_type,
# #                 "position": position,
# #                 "confidence": vein["confidence"],
# #                 "center": vein["center"],
# #                 "bbox": vein["bbox"],
# #                 "width": vein["width"],
# #                 "height": vein["height"],
# #                 "radius": max(vein["width"], vein["height"]) / 2,
# #                 "distance_to_fascia": dist_to_fascia,
# #                 "fascia_distance": dist_to_fascia,
# #                 "in_fascia_region": classification == "N2",
# #                 "color": color,
# #             })
# #
# #         annotated = self._annotate_frame(frame.copy(), classified, fascia_upper_y, fascia_lower_y)
# #
# #         return {
# #             "status": "success" if classified else "no_veins",
# #             "detections": classified,
# #             "annotated_frame": annotated,
# #             "frame_shape": frame.shape,
# #             "num_veins": len(classified),
# #         }
# #
# #     def _detect_fascia_annotation(self, frame):
# #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# #         height, width = frame.shape[:2]
# #         combined_mask = np.zeros((height, width), dtype=np.uint8)
# #         for lo, hi in self.FASCIA_HSV_RANGES:
# #             mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
# #             combined_mask = cv2.bitwise_or(combined_mask, mask)
# #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
# #         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
# #         combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
# #         fascia_pixel_count = cv2.countNonZero(combined_mask)
# #         if fascia_pixel_count < width * 0.1:
# #             return None, None, None
# #         fascia_upper = np.full(width, np.nan, dtype=np.float32)
# #         fascia_lower = np.full(width, np.nan, dtype=np.float32)
# #         for col in range(width):
# #             ys = np.where(combined_mask[:, col] > 0)[0]
# #             if len(ys) > 0:
# #                 fascia_upper[col] = np.min(ys)
# #                 fascia_lower[col] = np.max(ys)
# #         valid = ~np.isnan(fascia_upper)
# #         if valid.sum() < 10:
# #             return None, None, None
# #         xs = np.arange(width)
# #         fascia_upper = np.interp(xs, xs[valid], fascia_upper[valid]).astype(np.float32)
# #         fascia_lower = np.interp(xs, xs[valid], fascia_lower[valid]).astype(np.float32)
# #         median_gap = np.median(fascia_lower - fascia_upper)
# #         fallback_margin = int(height * self.FASCIA_MARGIN_RATIO)
# #         if median_gap < fallback_margin:
# #             center = (fascia_upper + fascia_lower) / 2
# #             fascia_upper = center - fallback_margin
# #             fascia_lower = center + fallback_margin
# #         return combined_mask, fascia_upper, fascia_lower
# #
# #     def _detect_dark_blobs(self, frame, fascia_mask):
# #         # ... (full blob detection code omitted for brevity, see git history)
# #         pass
# #
# #     def _annotate_frame(self, frame, classified_veins, fascia_upper_y=None, fascia_lower_y=None):
# #         # ... (full annotation code omitted for brevity, see git history)
# #         pass
