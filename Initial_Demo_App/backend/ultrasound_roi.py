"""
Ultrasound ROI (Region of Interest) Detection & Cropping
Detects and extracts the ultrasound scan area from frames with black borders
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UltrasoundROI:
    """Detect and crop ultrasound region from frames"""

    @staticmethod
    def detect_roi(frame: np.ndarray, threshold: int = 30) -> tuple:
        """
        Detect the ultrasound region (non-black area) in frame.

        Args:
            frame: Input frame (BGR or RGB)
            threshold: Intensity threshold to distinguish from black background (0-255)

        Returns:
            Tuple of (x, y, w, h) bounding box, or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Find non-black pixels (ultrasound region)
        # Black background typically has very low intensity
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No ultrasound region detected in frame")
            return None

        # Get largest contour (the ultrasound region)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ensure minimum size (reject noise)
        min_size = 100
        if w < min_size or h < min_size:
            logger.warning(f"Detected region too small: {w}x{h}")
            return None

        return x, y, w, h

    @staticmethod
    def crop_to_roi(frame: np.ndarray, roi: tuple = None, auto_detect: bool = True) -> tuple:
        """
        Crop frame to ultrasound ROI.

        Args:
            frame: Input frame
            roi: (x, y, w, h) bounding box, or None to auto-detect
            auto_detect: Auto-detect ROI if not provided

        Returns:
            Tuple of (cropped_frame, roi)
        """
        if roi is None:
            if auto_detect:
                roi = UltrasoundROI.detect_roi(frame)
            if roi is None:
                return frame, None

        x, y, w, h = roi
        cropped = frame[y:y+h, x:x+w]

        return cropped, roi

    @staticmethod
    def find_center_square_roi(frame: np.ndarray, padding_percent: float = 0.05) -> tuple:
        """
        Find center square ROI (assumes ultrasound is roughly centered and square).
        Useful for ultrasounds with consistent format.

        Args:
            frame: Input frame
            padding_percent: Padding percentage from detected region

        Returns:
            Tuple of (x, y, w, h)
        """
        roi = UltrasoundROI.detect_roi(frame)
        if roi is None:
            return None

        x, y, w, h = roi

        # Make it square (use smaller dimension)
        size = min(w, h)

        # Center the square
        center_x = x + w // 2
        center_y = y + h // 2

        square_x = max(0, center_x - size // 2)
        square_y = max(0, center_y - size // 2)

        # Add padding
        padding = int(size * padding_percent)
        square_x = max(0, square_x - padding)
        square_y = max(0, square_y - padding)
        size = min(size + 2 * padding, min(frame.shape[1], frame.shape[0]))

        # Ensure within bounds
        square_x = min(square_x, frame.shape[1] - size)
        square_y = min(square_y, frame.shape[0] - size)

        return square_x, square_y, size, size

    @staticmethod
    def crop_video_frames(frames: list, crop_mode: str = 'none') -> list:
        """
        Crop multiple frames using specified mode.

        Args:
            frames: List of frame arrays
            crop_mode: 'none', 'auto', or 'square'

        Returns:
            List of cropped frames
        """
        if crop_mode == 'none':
            return frames

        cropped_frames = []
        roi = None  # Cache ROI from first frame

        for i, frame in enumerate(frames):
            if crop_mode == 'auto':
                cropped, roi = UltrasoundROI.crop_to_roi(frame, roi, auto_detect=(i == 0))
            elif crop_mode == 'square':
                if roi is None:
                    roi = UltrasoundROI.find_center_square_roi(frame)
                if roi:
                    cropped, _ = UltrasoundROI.crop_to_roi(frame, roi, auto_detect=False)
                else:
                    cropped = frame
            else:
                cropped = frame

            cropped_frames.append(cropped)

        return cropped_frames


def apply_roi_to_frame(frame: np.ndarray, crop_mode: str = 'none', roi: tuple = None) -> tuple:
    """
    Utility function to apply ROI cropping to a single frame.

    Args:
        frame: Input frame
        crop_mode: 'none', 'auto', or 'square'
        roi: Cached ROI from previous frame (for consistency)

    Returns:
        Tuple of (cropped_frame, roi)
    """
    if crop_mode == 'none':
        return frame, roi

    if crop_mode == 'auto':
        if roi is None:
            roi = UltrasoundROI.detect_roi(frame)
        if roi:
            cropped, _ = UltrasoundROI.crop_to_roi(frame, roi, auto_detect=False)
            return cropped, roi
        else:
            return frame, None

    elif crop_mode == 'square':
        if roi is None:
            roi = UltrasoundROI.find_center_square_roi(frame)
        if roi:
            cropped, _ = UltrasoundROI.crop_to_roi(frame, roi, auto_detect=False)
            return cropped, roi
        else:
            return frame, None

    return frame, roi
