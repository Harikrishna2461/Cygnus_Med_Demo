"""
Curved Fascia Detection using Contour Following and Brightness Analysis.
Detects fascia as two bright curved boundaries (not straight lines).
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Any
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d


class CurvedFasciaDetector:
    """
    Detects fascia as two bright curved boundaries.
    Works with wavy, undulating fascia patterns in ultrasound.
    """
    
    def __init__(self):
        self.fascia_mask = None
        self.upper_curve = None
        self.lower_curve = None
        self.confidence = 0.0
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast and denoise.
        
        Args:
            image: Input ultrasound image
            
        Returns:
            Enhanced grayscale image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter for denoising while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def detect_bright_boundaries(self, image: np.ndarray,
                                  brightness_threshold: int = 150) -> np.ndarray:
        """
        Detect bright regions (candidate fascia).
        Fascia appears as bright band = high intensity.
        
        Args:
            image: Enhanced grayscale image
            brightness_threshold: Minimum intensity for fascia
            
        Returns:
            Binary mask of bright regions
        """
        _, bright_mask = cv2.threshold(image, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        return bright_mask
    
    def find_horizontal_band(self, image: np.ndarray,
                            min_width_coverage: float = 0.4) -> Tuple[int, int]:
        """
        Find the main horizontal fascia band using intensity peaks.
        Fascia appears as bright line(s), look for local maxima along y-axis.
        
        Args:
            image: Enhanced grayscale image (not bright mask)
            min_width_coverage: Minimum width coverage for a valid band
            
        Returns:
            (y_start, y_end) bounds of fascia band
        """
        h, w = image.shape
        
        # Project brightness onto y-axis (average intensity per row)
        y_projection = np.mean(image, axis=1)
        
        # Find local brightness peaks (fascia appears as bright band)
        # Use gradient to find transitions
        y_gradient = np.diff(y_projection)
        
        # Smooth gradient
        y_gradient_smooth = gaussian_filter1d(y_gradient, sigma=2)
        
        # Find zero crossings (peaks in original, transitions in smoothed)
        peaks = []
        for i in range(1, len(y_gradient_smooth)-1):
            if y_gradient_smooth[i-1] > 0 and y_gradient_smooth[i] < 0:
                # Peak at y_projection[i]
                peaks.append(i)
        
        if len(peaks) == 0:
            # Fallback: find brightest region
            brightest_y = np.argmax(y_projection)
            y_start = max(0, brightest_y - 30)
            y_end = min(h-1, brightest_y + 30)
        else:
            # Use the most prominent peaks (those with highest intensity)
            peak_intensities = [(i, y_projection[i]) for i in peaks]
            peak_intensities.sort(key=lambda x: x[1], reverse=True)
            
            # Find the brightest continuous region containing top peaks
            if len(peak_intensities) >= 2:
                y1, int1 = peak_intensities[0]
                y2, int2 = peak_intensities[1]
                
                y_start = min(y1, y2) - 5
                y_end = max(y1, y2) + 5
            else:
                y_start = peak_intensities[0][0] - 10
                y_end = peak_intensities[0][0] + 10
            
            y_start = max(0, y_start)
            y_end = min(h-1, y_end)
        
        return int(y_start), int(y_end)
    
    def extract_boundary_curve(self, image: np.ndarray,
                               y_start: int, y_end: int,
                               direction: str = 'upper') -> np.ndarray:
        """
        Extract upper or lower boundary curve of fascia band.
        Finds the bright boundary line at each x position.
        
        Args:
            image: Enhanced grayscale image
            y_start: Top of fascia band
            y_end: Bottom of fascia band
            direction: 'upper' or 'lower'
            
        Returns:
            Array of (x, y) coordinates forming the curve
        """
        h, w = image.shape
        
        # Extract band region with some padding
        band_top = max(0, y_start - 10)
        band_bottom = min(h-1, y_end + 10)
        band = image[band_top:band_bottom+1, :]
        
        band_height = band.shape[0]
        
        if direction == 'upper':
            # Find brightest pixel in upper half of band at each x
            search_region = band[:band_height//2, :]
        else:
            # Find brightest pixel in lower half of band at each x
            search_region = band[band_height//2:, :]
        
        boundary_y_local = []
        for x in range(w):
            col = search_region[:, x]
            
            # Find local maxima (bright pixels)
            if np.max(col) > 80:  # Has bright pixels
                # Find brightest cluster (may have multiple peaks)
                bright_mask = col > (np.max(col) - 30)  # Within 30 of max
                bright_indices = np.where(bright_mask)[0]
                
                if len(bright_indices) > 0:
                    # Use center of bright region
                    y_local = int(np.mean(bright_indices))
                    boundary_y_local.append(y_local)
                else:
                    boundary_y_local.append(None)
            else:
                boundary_y_local.append(None)
        
        # Convert local indices back to global
        boundary_y = []
        for y_local in boundary_y_local:
            if y_local is not None:
                if direction == 'upper':
                    y_global = band_top + y_local
                else:
                    y_global = band_top + (band_height // 2) + y_local
                boundary_y.append(y_global)
            else:
                boundary_y.append(None)
        
        # Smooth the curve
        boundary_y = self._smooth_curve(boundary_y, window=7)
        
        x_coords = np.arange(w)
        valid_indices = [i for i, y in enumerate(boundary_y) if y is not None and 0 <= y < h]
        
        if len(valid_indices) < 20:  # Need enough valid points
            return None
        
        curve = np.column_stack([
            x_coords[valid_indices],
            [boundary_y[i] for i in valid_indices]
        ]).astype(np.int32)
        
        return curve
    
    def _smooth_curve(self, y_values: List, window: int = 5) -> List:
        """
        Smooth curve using median filter and interpolation.
        
        Args:
            y_values: Y-coordinates (may contain None for missing values)
            window: Smoothing window size
            
        Returns:
            Smoothed y-coordinates
        """
        # Extract valid values
        valid_mask = [v is not None for v in y_values]
        valid_indices = np.where(valid_mask)[0]
        valid_values = [y_values[i] for i in valid_indices]
        
        if len(valid_values) < window:
            return y_values
        
        # Apply median filter
        valid_values = ndimage.median_filter(valid_values, size=window)
        
        # Interpolate back to full length
        smoothed = np.interp(
            np.arange(len(y_values)),
            valid_indices,
            valid_values
        )
        
        return smoothed.tolist()
    
    def validate_fascia_curves(self, image: np.ndarray,
                               upper_curve: np.ndarray,
                               lower_curve: np.ndarray) -> float:
        """
        Validate fascia curves by checking spacing and intensity.
        
        Args:
            image: Original image
            upper_curve: Upper boundary points (N, 2)
            lower_curve: Lower boundary points (N, 2)
            
        Returns:
            Confidence score (0-1)
        """
        if upper_curve is None or lower_curve is None:
            return 0.0
        
        upper_y = upper_curve[:, 1].astype(float)
        lower_y = lower_curve[:, 1].astype(float)
        
        # Check: lower should be consistently below upper
        valid_pairs = lower_y > upper_y
        if not np.any(valid_pairs):
            return 0.0
        
        # Check spacing
        spacing = lower_y[valid_pairs] - upper_y[valid_pairs]
        
        if len(spacing) == 0:
            return 0.0
        
        avg_spacing = np.mean(spacing)
        spacing_std = np.std(spacing)
        
        # Band should have consistent spacing (2-50 pixels, low variance)
        if avg_spacing < 2 or avg_spacing > 50:
            return 0.0
        
        # Penalize high variance in spacing
        spacing_consistency = 1.0 if spacing_std < 10 else max(0, 1.0 - spacing_std/20)
        
        # Soft scoring for optimal spacing
        optimal_spacing = 15
        spacing_score = 1.0 - (abs(avg_spacing - optimal_spacing) / 20)
        spacing_score = max(0, min(1, spacing_score))
        
        # Check intensity contrast (optional - may not always work)
        h, w = image.shape
        
        try:
            # Sample regions to check contrast
            sample_x = np.linspace(10, w-10, 10).astype(int)
            above_samples = []
            below_samples = []
            
            for x in sample_x:
                for y_idx, y in enumerate(upper_y[:10]):  # Sample upper curve
                    y_int = int(y)
                    if 0 < y_int - 10 < h:
                        above_samples.append(image[y_int - 10, x])
                
                for y in lower_y[-10:]:  # Sample lower curve
                    y_int = int(y)
                    if y_int + 10 < h:
                        below_samples.append(image[y_int + 10, x])
            
            if len(above_samples) > 0 and len(below_samples) > 0:
                above_mean = np.mean(above_samples)
                below_mean = np.mean(below_samples)
                intensity_diff = above_mean - below_mean
                
                # Fascia: above is bright, below is dark (above > below)
                if intensity_diff > 0:
                    intensity_score = min(intensity_diff / 80, 1.0)
                else:
                    intensity_score = 0.0
            else:
                intensity_score = 0.5  # Neutral if can't check
        except:
            intensity_score = 0.5
        
        # Combined confidence (emphasis on spacing since that's most reliable)
        confidence = (spacing_score * 0.6 + spacing_consistency * 0.2 + intensity_score * 0.2)
        
        return max(0, min(1, confidence))
    
    def create_fascia_mask(self, shape: Tuple[int, int],
                          upper_curve: np.ndarray,
                          lower_curve: np.ndarray) -> np.ndarray:
        """
        Create binary mask of fascia from curves.
        
        Args:
            shape: (height, width)
            upper_curve: Upper boundary points
            lower_curve: Lower boundary points
            
        Returns:
            Binary mask with fascia = 255
        """
        mask = np.zeros(shape, dtype=np.uint8)
        
        if upper_curve is None or lower_curve is None:
            return mask
        
        h, w = shape
        
        # Draw curves
        cv2.polylines(mask, [upper_curve], False, 255, 2)
        cv2.polylines(mask, [lower_curve], False, 255, 2)
        
        # Fill region between curves
        # Create filled region by drawing lines from upper to lower
        for i in range(min(len(upper_curve), len(lower_curve))):
            pt1 = tuple(upper_curve[i])
            pt2 = tuple(lower_curve[i])
            cv2.line(mask, pt1, pt2, 255, 1)
        
        return mask
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main detection pipeline for curved fascia.
        
        Args:
            image: Ultrasound image
            
        Returns:
            Dict with mask, curves, confidence, etc.
        """
        h, w = image.shape[:2]
        
        # Step 1: Enhance
        enhanced = self.enhance_image(image)
        
        # Step 2: Find horizontal band (using enhanced image for intensity peaks)
        y_start, y_end = self.find_horizontal_band(enhanced)
        
        if y_start is None:
            return {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'upper_curve': None,
                'lower_curve': None,
                'confidence': 0.0,
                'status': 'no_bright_band_found'
            }
        
        # Step 4: Extract boundary curves
        upper_curve = self.extract_boundary_curve(enhanced, y_start, y_end, 'upper')
        lower_curve = self.extract_boundary_curve(enhanced, y_start, y_end, 'lower')
        
        if upper_curve is None or lower_curve is None:
            return {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'upper_curve': None,
                'lower_curve': None,
                'confidence': 0.0,
                'status': 'failed_to_extract_curves'
            }
        
        # Step 5: Validate
        confidence = self.validate_fascia_curves(enhanced, upper_curve, lower_curve)
        
        # Step 6: Create mask
        mask = self.create_fascia_mask((h, w), upper_curve, lower_curve)
        
        self.upper_curve = upper_curve
        self.lower_curve = lower_curve
        self.confidence = confidence
        self.fascia_mask = mask
        
        return {
            'mask': mask,
            'upper_curve': upper_curve,
            'lower_curve': lower_curve,
            'confidence': confidence,
            'band_bounds': (y_start, y_end),
            'status': 'success'
        }
    
    def get_mask(self) -> np.ndarray:
        """Return fascia mask."""
        if self.fascia_mask is None:
            return np.zeros((256, 256), dtype=np.uint8)
        return self.fascia_mask


if __name__ == '__main__':
    """Test curved fascia detector"""
    # Create synthetic curved fascia
    test_img = np.ones((256, 256), dtype=np.uint8) * 80
    
    # Bright region (above fascia)
    test_img[:80, :] = 160
    
    # Curved upper fascia
    x = np.arange(256)
    upper_y = 90 + 5 * np.sin(x / 50)
    for i, y in enumerate(upper_y):
        y_int = int(y)
        if 0 <= y_int < 256:
            test_img[y_int-1:y_int+2, i] = 220
    
    # Curved lower fascia (20 pixels below upper)
    lower_y = upper_y + 15
    for i, y in enumerate(lower_y):
        y_int = int(y)
        if 0 <= y_int < 256:
            test_img[y_int-1:y_int+2, i] = 220
    
    # Dark region (below fascia)
    test_img[120:, :] = 50
    
    test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    
    detector = CurvedFasciaDetector()
    result = detector.detect(test_img)
    
    print(f"Curved fascia detection:")
    print(f"  Status: {result['status']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Upper curve: {result['upper_curve'].shape if result['upper_curve'] is not None else 'None'}")
    print(f"  Lower curve: {result['lower_curve'].shape if result['lower_curve'] is not None else 'None'}")
