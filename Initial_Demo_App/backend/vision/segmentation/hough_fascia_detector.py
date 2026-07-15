"""
Hough Transform-based Fascia Detection
Detects parallel horizontal lines in ultrasound images using OpenCV's Hough Transform
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any


class HoughFasciaDetector:
    """
    Detects fascia as parallel horizontal bright lines using Hough Transform.
    
    Fascia characteristics:
    - Two parallel bright horizontal lines
    - ~2-4 pixels thick each
    - >70% image width  
    - Between bright fat (above) and dark muscle (below)
    """
    
    def __init__(self):
        self.fascia_mask = None
        self.fascia_centerline = None
        self.confidence = 0.0
        self.upper_line = None
        self.lower_line = None
        
    def denoise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Denoise image using bilateral filtering (preserves edges).
        
        Args:
            image: Input ultrasound image
            kernel_size: Filter kernel size
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter: denoise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def detect_horizontal_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect horizontal edges using Sobel filter (vertical gradient).
        Horizontal lines have strong vertical gradients.
        
        Args:
            image: Denoised ultrasound image
            
        Returns:
            Edge magnitude image
        """
        # Sobel in Y direction (vertical gradient) detects horizontal edges
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Convert to absolute values and uint8
        sobelY = cv2.convertScaleAbs(sobelY)
        
        return sobelY
    
    def detect_lines_hough(self, edges: np.ndarray, 
                          threshold: int = 50,
                          min_length: int = 100,
                          max_gap: int = 10) -> list:
        """
        Detect long horizontal lines using Probabilistic Hough Transform.
        Filters for truly horizontal lines and removes duplicates.
        
        Args:
            edges: Edge-detected image
            threshold: Hough voting threshold
            min_length: Minimum line length (pixels)
            max_gap: Maximum gap between line segments
            
        Returns:
            List of lines as [(x1,y1,x2,y2), ...]
        """
        # Apply median filter to edges first
        edges = cv2.medianBlur(edges, 5)
        
        # Threshold edges
        _, edges_binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
        
        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges_binary,
            rho=1,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        
        if lines is None:
            return []
        
        # Filter for near-horizontal lines (within ±10° of horizontal)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Accept lines that are nearly horizontal (0° or 180° ± 10°)
            if angle < 10 or angle > 170:
                horizontal_lines.append((x1, y1, x2, y2))
        
        # Remove duplicate lines (same y-coordinate)
        # Group by y-position (with 2-pixel tolerance)
        y_groups = {}
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            avg_y = (y1 + y2) / 2
            
            # Find matching group
            found_group = False
            for y_key in list(y_groups.keys()):
                if abs(avg_y - y_key) <= 2:  # Within 2 pixels
                    y_groups[y_key].append(line)
                    found_group = True
                    break
            
            if not found_group:
                y_groups[avg_y] = [line]
        
        # Keep best line from each y-group
        unique_lines = []
        for y_key, group in y_groups.items():
            # Choose line with longest length
            best_line = max(group, key=lambda l: abs(l[2] - l[0]))
            unique_lines.append(best_line)
        
        return unique_lines
    
    def cluster_parallel_lines(self, lines: list, 
                              y_tolerance: int = 3,
                              min_cluster_size: int = 2) -> list:
        """
        Cluster nearby parallel horizontal lines into bands.
        Filters out noise by requiring minimum cluster size.
        
        Args:
            lines: List of detected lines
            y_tolerance: Vertical distance tolerance for clustering (in pixels)
            min_cluster_size: Minimum lines per cluster
            
        Returns:
            List of line clusters (each cluster is list of parallel lines)
        """
        if not lines:
            return []
        
        # Extract y-coordinates of lines
        y_coords = []
        for x1, y1, x2, y2 in lines:
            avg_y = (y1 + y2) / 2
            y_coords.append(avg_y)
        
        # Sort by y-coordinate
        sorted_indices = sorted(range(len(y_coords)), key=lambda i: y_coords[i])
        sorted_lines = [lines[i] for i in sorted_indices]
        sorted_y = [y_coords[i] for i in sorted_indices]
        
        # Cluster lines by proximity - use stricter tolerance
        clusters = []
        current_cluster = [sorted_lines[0]]
        
        for i in range(1, len(sorted_lines)):
            # Check distance from last line in current cluster
            distance_from_last = sorted_y[i] - sorted_y[i-1]
            
            if distance_from_last <= y_tolerance:
                # Same cluster
                current_cluster.append(sorted_lines[i])
            else:
                # Gap detected - start new cluster if current is large enough
                if len(current_cluster) >= min_cluster_size:
                    clusters.append(current_cluster)
                current_cluster = [sorted_lines[i]]
        
        # Add final cluster
        if len(current_cluster) >= min_cluster_size:
            clusters.append(current_cluster)
        
        return clusters
    
    def merge_line_cluster(self, cluster: list) -> Tuple[float, float, float]:
        """
        Merge parallel lines in a cluster into single centerline.
        
        Args:
            cluster: List of lines in cluster
            
        Returns:
            (y_pos, x_start, x_end) of merged line
        """
        y_positions = []
        x_starts = []
        x_ends = []
        
        for x1, y1, x2, y2 in cluster:
            y_positions.append((y1 + y2) / 2)
            x_starts.append(min(x1, x2))
            x_ends.append(max(x1, x2))
        
        y_pos = np.mean(y_positions)
        x_start = np.min(x_starts)
        x_end = np.max(x_ends)
        
        return y_pos, x_start, x_end
    
    def validate_fascia_band(self, image: np.ndarray,
                            upper_line: Tuple[float, float, float],
                            lower_line: Tuple[float, float, float]) -> float:
        """
        Validate fascia band by checking intensity difference.
        Fascia sits between bright fat (above) and dark muscle (below).
        
        Args:
            image: Original ultrasound image
            upper_line: (y, x_start, x_end) of upper fascia line
            lower_line: (y, x_start, x_end) of lower fascia line
            
        Returns:
            Confidence score (0-1)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = image.shape
        y_upper, x_start_u, x_end_u = upper_line
        y_lower, x_start_l, x_end_l = lower_line
        
        # Sample regions above, fascia band, and below
        x_start = max(int(min(x_start_u, x_start_l)), 0)
        x_end = min(int(max(x_end_u, x_end_l)), w - 1)
        
        y_upper_int = int(y_upper)
        y_lower_int = int(y_lower)
        
        if y_upper_int < 10 or y_lower_int > h - 10 or y_upper_int >= y_lower_int:
            return 0.0
        
        # Sample intensity above (bright fat)
        above_region = image[max(0, y_upper_int - 20):y_upper_int, x_start:x_end]
        intensity_above = np.mean(above_region) if above_region.size > 0 else 0
        
        # Sample intensity below (dark muscle)
        below_region = image[y_lower_int:min(h, y_lower_int + 20), x_start:x_end]
        intensity_below = np.mean(below_region) if below_region.size > 0 else 0
        
        # Fascia should have above brighter than below
        if intensity_above > intensity_below:
            intensity_diff = intensity_above - intensity_below
            # Normalize to 0-1
            confidence = min(intensity_diff / 100, 1.0)
        else:
            confidence = 0.0
        
        return confidence
    
    def create_fascia_mask(self, shape: Tuple[int, int],
                          upper_line: Tuple[float, float, float],
                          lower_line: Tuple[float, float, float],
                          band_thickness: int = 3) -> np.ndarray:
        """
        Create binary mask of fascia band.
        
        Args:
            shape: (height, width) of mask
            upper_line: (y, x_start, x_end)
            lower_line: (y, x_start, x_end)
            band_thickness: Thickness of fascia lines
            
        Returns:
            Binary mask with fascia = 255, background = 0
        """
        mask = np.zeros(shape, dtype=np.uint8)
        h, w = shape
        
        y_upper, x_start_u, x_end_u = upper_line
        y_lower, x_start_l, x_end_l = lower_line
        
        # Draw upper line
        cv2.line(mask,
                (int(x_start_u), int(y_upper)),
                (int(x_end_u), int(y_upper)),
                255, band_thickness)
        
        # Draw lower line
        cv2.line(mask,
                (int(x_start_l), int(y_lower)),
                (int(x_end_l), int(y_lower)),
                255, band_thickness)
        
        # Fill region between lines
        y_upper_int = int(y_upper)
        y_lower_int = int(y_lower)
        x_start = max(int(min(x_start_u, x_start_l)), 0)
        x_end = min(int(max(x_end_u, x_end_l)), w)
        
        if y_upper_int < y_lower_int:
            mask[y_upper_int:y_lower_int, x_start:x_end] = 255
        
        return mask
    
    def create_centerline(self, shape: Tuple[int, int],
                         upper_line: Tuple[float, float, float],
                         lower_line: Tuple[float, float, float]) -> np.ndarray:
        """
        Create centerline between upper and lower fascia lines.
        
        Args:
            shape: (height, width) of image
            upper_line: (y, x_start, x_end)
            lower_line: (y, x_start, x_end)
            
        Returns:
            Binary mask with centerline = 255
        """
        centerline = np.zeros(shape, dtype=np.uint8)
        
        y_upper, x_start_u, x_end_u = upper_line
        y_lower, x_start_l, x_end_l = lower_line
        
        # Calculate centerline
        y_center = (y_upper + y_lower) / 2
        x_start = max(min(x_start_u, x_start_l), 0)
        x_end = min(max(x_end_u, x_end_l), shape[1])
        
        cv2.line(centerline,
                (int(x_start), int(y_center)),
                (int(x_end), int(y_center)),
                255, 1)
        
        return centerline
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main detection pipeline: denoise → edges → Hough → cluster → validate.
        
        Args:
            image: Ultrasound image (BGR or grayscale)
            
        Returns:
            Dict with:
              - 'mask': Binary fascia band mask
              - 'centerline': Fascia centerline
              - 'confidence': Confidence score
              - 'upper_line': (y, x_start, x_end)
              - 'lower_line': (y, x_start, x_end)
        """
        # Step 1: Denoise
        denoised = self.denoise(image)
        
        # Step 2: Detect horizontal edges
        edges = self.detect_horizontal_edges(denoised)
        
        # Step 3: Hough Transform to find lines
        lines = self.detect_lines_hough(edges, threshold=40, min_length=80, max_gap=15)
        
        if len(lines) < 2:
            # Not enough lines detected
            h, w = denoised.shape if len(denoised.shape) == 2 else denoised.shape[:2]
            result = {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'centerline': np.zeros((h, w), dtype=np.uint8),
                'confidence': 0.0,
                'upper_line': None,
                'lower_line': None,
                'num_lines_detected': len(lines)
            }
            return result
        
        # Step 4: Cluster parallel lines
        clusters = self.cluster_parallel_lines(lines, y_tolerance=8)
        
        if len(clusters) < 2:
            # Need at least 2 bands for fascia
            h, w = denoised.shape if len(denoised.shape) == 2 else denoised.shape[:2]
            result = {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'centerline': np.zeros((h, w), dtype=np.uint8),
                'confidence': 0.0,
                'upper_line': None,
                'lower_line': None,
                'num_clusters': len(clusters)
            }
            return result
        
        # Step 5: Merge clusters into centerlines
        merged_clusters = []
        for cluster in clusters[:5]:  # Use first 5 clusters
            merged = self.merge_line_cluster(cluster)
            merged_clusters.append(merged)
        
        # Sort by y position
        merged_clusters.sort(key=lambda x: x[0])
        
        # Find best pair of adjacent clusters
        best_pair = None
        best_confidence = 0.0
        
        for i in range(len(merged_clusters) - 1):
            upper = merged_clusters[i]
            lower = merged_clusters[i + 1]
            
            # Check line separation (2-40 pixels for fascia band)
            y_diff = lower[0] - upper[0]
            if y_diff < 2 or y_diff > 40:
                continue
            
            # Check width coverage - at least 50% of image width
            x_union_start = max(upper[1], lower[1])
            x_union_end = min(upper[2], lower[2])
            if x_union_end <= x_union_start:
                continue
            
            coverage = (x_union_end - x_union_start) / image.shape[1]
            if coverage < 0.5:  # Relaxed from 0.7 to 0.5
                continue
            
            # Validate intensity
            conf = self.validate_fascia_band(image, upper, lower)
            
            if conf > best_confidence:
                best_confidence = conf
                best_pair = (upper, lower)
        
        if best_pair is None:
            h, w = denoised.shape if len(denoised.shape) == 2 else denoised.shape[:2]
            result = {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'centerline': np.zeros((h, w), dtype=np.uint8),
                'confidence': 0.0,
                'upper_line': None,
                'lower_line': None,
                'pairs_evaluated': len(merged_clusters) - 1
            }
            return result
        
        upper_line, lower_line = best_pair
        self.upper_line = upper_line
        self.lower_line = lower_line
        self.confidence = best_confidence
        
        # Step 6: Create mask and centerline
        h, w = denoised.shape if len(denoised.shape) == 2 else denoised.shape[:2]
        mask = self.create_fascia_mask((h, w), upper_line, lower_line, band_thickness=2)
        centerline = self.create_centerline((h, w), upper_line, lower_line)
        
        self.fascia_mask = mask
        self.fascia_centerline = centerline
        
        result = {
            'mask': mask,
            'centerline': centerline,
            'confidence': best_confidence,
            'upper_line': upper_line,
            'lower_line': lower_line,
            'num_lines_detected': len(lines),
            'num_clusters': len(clusters)
        }
        
        return result
    
    def get_mask(self) -> np.ndarray:
        """Return fascia mask as tensor-compatible array."""
        if self.fascia_mask is None:
            return np.zeros((256, 256), dtype=np.uint8)
        return self.fascia_mask


if __name__ == '__main__':
    """Test the detector"""
    # Create synthetic ultrasound-like test image
    test_img = np.ones((256, 256, 3), dtype=np.uint8) * 100
    
    # Bright region (fat) - top 80 pixels
    test_img[:80, :] = 180
    
    # Fascia lines
    test_img[90:92, 20:200] = 220  # Upper fascia
    test_img[100:102, 20:200] = 220  # Lower fascia
    
    # Dark region (muscle) - bottom
    test_img[110:, :] = 50
    
    detector = HoughFasciaDetector()
    result = detector.detect(test_img)
    
    print(f"Fascia detected: confidence={result['confidence']:.3f}")
    print(f"Upper line: {result['upper_line']}")
    print(f"Lower line: {result['lower_line']}")
    print(f"Mask shape: {result['mask'].shape}")
    print(f"Lines detected: {result.get('num_lines_detected', '?')}")
    print(f"Clusters found: {result.get('num_clusters', '?')}")
