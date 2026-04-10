"""
Visualization Module for Vein and Fascia Annotations

Creates annotated visualizations with veins, fascia, and classification results.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class UltrasoundVisualizer:
    """High-quality visualization of segmentation and classification results"""
    
    # Color definitions (BGR format for OpenCV) - BRIGHT colors for dark ultrasound images
    COLORS = {
        "fascia": (0, 255, 255),            # Cyan - BRIGHT for fascia outline
        "deep_vein": (255, 0, 0),           # Blue - BRIGHT bounding box
        "superficial_vein": (0, 255, 0),    # Green - BRIGHT bounding box
        "perforator_vein": (0, 165, 255),   # Orange - BRIGHT bounding box
        "gsv": (255, 0, 255),               # Magenta
        "unknown": (255, 255, 0)            # Yellow - BRIGHT for unknowns
    }
    
    ALPHA_MASK = 0.5  # Lower transparency to keep ultrasound visible
    ALPHA_CONTOUR = 0.7
    UPSCALE_FACTOR = 1.0  # No upscaling - keep original size for tight bounding boxes
    
    def __init__(self, font_scale: float = 0.5, thickness: int = 1):
        """
        Initialize visualizer
        
        Args:
            font_scale: Font size scale (reduced for tighter display)
            thickness: Line thickness (reduced for clarity)
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def visualize_segmentation(
        self,
        frame: np.ndarray,
        fascia_mask: np.ndarray,
        vein_masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Create segmentation visualization
        
        Args:
            frame: Original ultrasound frame
            fascia_mask: Binary fascia mask
            vein_masks: List of binary vein masks
        
        Returns:
            Annotated frame (upscaled)
        """
        output = frame.copy().astype(np.uint8)
        output = cv2.convertScaleAbs(output, alpha=1.2, beta=20)
        
        # UPSCALE the image
        h, w = output.shape[:2]
        new_h = int(h * self.UPSCALE_FACTOR)
        new_w = int(w * self.UPSCALE_FACTOR)
        output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Scale and draw fascia
        fascia_mask_scaled = cv2.resize(
            (fascia_mask > 0).astype(np.uint8) * 255,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )
        self._draw_fascia_boundary(output, fascia_mask_scaled)
        
        # Draw each vein as bounding box
        for i, vein_mask in enumerate(vein_masks):
            vein_mask_scaled = cv2.resize(
                (vein_mask > 0).astype(np.uint8) * 255,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            contours, _ = cv2.findContours(
                vein_mask_scaled.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(contours[0])
                
                # Draw bounding box with yellow color
                color = (0, 255, 255)  # Cyan
                cv2.rectangle(output, (x, y), (x + w_box, y + h_box), color, 2)
                
                # Draw vein ID label
                cv2.putText(
                    output, f"V{i}",
                    (x + 5, y + 20),
                    self.font, self.font_scale,
                    color, 2
                )
        
        return output
    
    def visualize_classification(
        self,
        frame: np.ndarray,
        fascia_mask: np.ndarray,
        veins: List[Dict]
    ) -> np.ndarray:
        """
        Create clean classification visualization matching reference style.
        
        Args:
            frame: Original ultrasound frame
            fascia_mask: Binary fascia mask
            veins: List of classified vein dictionaries
        
        Returns:
            Annotated frame with clean fascia lines and vein boxes
        """
        output = frame.copy().astype(np.uint8)
        
        # Minimal brightness enhancement
        output = cv2.convertScaleAbs(output, alpha=1.1, beta=10)
        
        h, w = output.shape[:2]
        
        # Draw fascia boundary first (TWO lines)
        self._draw_fascia_boundary(output, fascia_mask)
        
        # Draw vein bounding boxes
        for i, vein in enumerate(veins):
            vein_mask = vein.get('mask')
            if vein_mask is None:
                continue
            
            classification = vein.get('classification', {})
            vein_type = classification.get('primary_classification', 'unknown')
            n_level = classification.get('n_level', '')
            confidence = classification.get('confidence', 0)
            
            # Get color for this vein type
            color = self._get_color_for_type(vein_type)
            
            # Find bounding box around vein
            contours, _ = cv2.findContours(
                vein_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(contours[0])
                
                # Draw BRIGHT bounding box (thick, clearly visible)
                cv2.rectangle(output, (x, y), (x + w_box, y + h_box), color, 3)
                
                # Draw label: "ID [type]"
                label = f"{vein.get('vein_id', f'V{i}')} [{n_level}]"
                
                # Draw label with dark background for contrast
                label_size = cv2.getTextSize(label, self.font, self.font_scale, 1)[0]
                label_x = x + 5
                label_y = y + label_size[1] + 5
                
                # Dark background rectangle
                cv2.rectangle(
                    output,
                    (label_x - 3, label_y - label_size[1] - 3),
                    (label_x + label_size[0] + 3, label_y + 3),
                    (0, 0, 0),
                    -1
                )
                
                # White text
                cv2.putText(
                    output, label,
                    (label_x, label_y),
                    self.font, self.font_scale,
                    (255, 255, 255), 1
                )
        
        return output
    
    def _draw_fascia_boundary(self, image: np.ndarray, fascia_mask: np.ndarray):
        """
        Draw fascia as TWO parallel bright horizontal lines (top and bottom of band).
        
        Args:
            image: Output image to draw on
            fascia_mask: Binary fascia mask
        """
        # Find the rows where fascia exists
        fascia_rows = np.where(np.sum(fascia_mask > 0, axis=1) > image.shape[1] * 0.2)[0]
        
        if len(fascia_rows) == 0:
            return
        
        # Get top and bottom boundaries
        fascia_top = int(fascia_rows[0])
        fascia_bottom = int(fascia_rows[-1])
        
        fascia_color = self.COLORS["fascia"]  # Bright cyan (0, 255, 255)
        
        # Draw TWO lines (marking top and bottom of fascia band)
        cv2.line(image, (0, fascia_top), (image.shape[1], fascia_top), fascia_color, 3)
        cv2.line(image, (0, fascia_bottom), (image.shape[1], fascia_bottom), fascia_color, 3)
        
        logger.info(f"[VIZ] Fascia band: top={fascia_top}, bottom={fascia_bottom}")
    
    def _draw_label_with_background(
        self,
        image: np.ndarray,
        main_label: str,
        detail_label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int]
    ):
        """
        Draw text labels with strong background for legibility
        
        Args:
            image: Output image
            main_label: Main label text (vein ID)
            detail_label: Detail label text (type, level, confidence)
            position: (x, y) position
            color: Text color
        """
        x, y = position
        font = self.font
        font_scale = self.font_scale
        thickness = 2
        
        # Get text sizes
        (main_width, main_height), main_baseline = cv2.getTextSize(
            main_label, font, font_scale, thickness
        )
        (detail_width, detail_height), detail_baseline = cv2.getTextSize(
            detail_label, font, font_scale * 0.9, 1
        )
        
        max_width = max(main_width, detail_width)
        total_height = main_height + detail_height + 15
        
        # Draw strong black background
        cv2.rectangle(
            image,
            (x - 8, y - main_height - 10),
            (x + max_width + 8, y + detail_height + 5),
            (0, 0, 0),  # Black background
            -1  # Filled
        )
        
        # Draw colored border
        cv2.rectangle(
            image,
            (x - 8, y - main_height - 10),
            (x + max_width + 8, y + detail_height + 5),
            color,
            2  # Border thickness
        )
        
        # Draw main label
        cv2.putText(image, main_label, (x, y), font, font_scale, color, thickness)
        
        # Draw detail label below
        cv2.putText(image, detail_label, (x, y + detail_height + 5), 
                   font, font_scale * 0.9, (255, 255, 255), 1)
    
    def visualize_detailed_analysis(
        self,
        frame: np.ndarray,
        fascia_mask: np.ndarray,
        veins: List[Dict],
        include_measurements: bool = True
    ) -> np.ndarray:
        """
        Create detailed analysis visualization with measurements
        
        Args:
            frame: Original ultrasound frame
            fascia_mask: Binary fascia mask
            veins: List of analyzed veins
            include_measurements: Whether to show distance measurements
        
        Returns:
            Detailed annotated frame (upscaled)
        """
        output = self.visualize_classification(frame, fascia_mask, veins)
        
        if not include_measurements:
            return output
        
        # Scale coordinates for upscaled image
        h, w = frame.shape[:2]
        scaled_w = int(w * self.UPSCALE_FACTOR)
        scaled_h = int(h * self.UPSCALE_FACTOR)
        
        # Scale fascia mask
        fascia_mask_scaled = cv2.resize(
            (fascia_mask > 0).astype(np.uint8) * 255,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Add measurement lines
        for vein in veins:
            spatial = vein.get('spatial_analysis', {})
            vein_centroid = spatial.get('vein_centroid')
            
            if vein_centroid is None:
                continue
            
            # Scale vein centroid
            cx = int(vein_centroid[0] * self.UPSCALE_FACTOR)
            cy = int(vein_centroid[1] * self.UPSCALE_FACTOR)
            
            # Draw perpendicular line to fascia
            fascia_y = self._get_fascia_y_at_x(fascia_mask_scaled, cx)
            
            if fascia_y > 0 and fascia_y != cy:
                # Draw measurement line with bright yellow
                cv2.line(output, (cx, cy), (cx, fascia_y), (0, 255, 255), 2)
                
                distance_mm = spatial.get('distance_to_fascia_mm', 0)
                
                # Mark endpoints
                cv2.circle(output, (cx, cy), 5, (0, 255, 255), -1)
                cv2.circle(output, (cx, fascia_y), 5, (0, 255, 255), -1)
                
                # Label distance with white background
                mid_y = (cy + fascia_y) // 2
                label = f"{distance_mm:.1f}mm"
                
                (text_width, text_height), _ = cv2.getTextSize(
                    label, self.font, self.font_scale * 0.8, 1
                )
                
                # Draw background
                cv2.rectangle(
                    output,
                    (cx + 10, mid_y - text_height - 5),
                    (cx + 10 + text_width, mid_y + 5),
                    (0, 0, 0), -1
                )
                
                # Draw text
                cv2.putText(
                    output, label,
                    (cx + 10, mid_y),
                    self.font, self.font_scale * 0.8,
                    (0, 255, 255), 1
                )
        
        return output
    
    def create_comparison_grid(
        self,
        frame: np.ndarray,
        segmentation_viz: np.ndarray,
        classification_viz: np.ndarray,
        detailed_viz: np.ndarray
    ) -> np.ndarray:
        """
        Create a 2x2 grid comparison of different visualizations
        
        Args:
            frame: Original frame
            segmentation_viz: Segmentation visualization (already upscaled)
            classification_viz: Classification visualization (already upscaled)
            detailed_viz: Detailed analysis visualization (already upscaled)
        
        Returns:
            Combined grid image
        """
        # Upscale original frame to match other visualizations
        h, w = frame.shape[:2]
        new_h = int(h * self.UPSCALE_FACTOR)
        new_w = int(w * self.UPSCALE_FACTOR)
        frame_upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # All images should now be the same size, just create the grid
        target_h = frame_upscaled.shape[0]
        target_w = frame_upscaled.shape[1]
        
        # Resize all to target size if needed
        seg_resized = cv2.resize(segmentation_viz, (target_w, target_h)) if segmentation_viz.shape != frame_upscaled.shape else segmentation_viz
        cls_resized = cv2.resize(classification_viz, (target_w, target_h)) if classification_viz.shape != frame_upscaled.shape else classification_viz
        det_resized = cv2.resize(detailed_viz, (target_w, target_h)) if detailed_viz.shape != frame_upscaled.shape else detailed_viz
        
        # Create grid
        top_row = np.hstack([frame_upscaled, seg_resized])
        bottom_row = np.hstack([cls_resized, det_resized])
        grid = np.vstack([top_row, bottom_row])
        
        # Add labels
        self._add_grid_labels(grid, (target_w, target_h))
        
        return grid
    
    def _draw_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int],
        label: str = "",
        alpha: float = 0.3
    ) -> np.ndarray:
        """Draw semi-transparent mask on image"""
        mask_uint8 = (mask > 0).astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask_uint8 > 0] = color
        
        # Blend
        return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    def _draw_label(
        self,
        image: np.ndarray,
        label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """Draw text label with background"""
        x, y = position
        font = self.font
        font_scale = self.font_scale * 0.8
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            image,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(image, label, (x, y), font, font_scale, color, thickness)
    
    def _calculate_non_overlapping_positions(
        self,
        image: np.ndarray,
        veins: List[Dict],
        scale_factor: float = 1.0
    ) -> Dict[int, Tuple[int, int]]:
        """
        Calculate label positions that avoid overlaps using spatial separation.
        
        Args:
            image: Output image (for bounds checking)
            veins: List of vein dictionaries with spatial_analysis data
            scale_factor: Scale factor applied to original image
        
        Returns:
            Dictionary mapping vein index to (x, y) label position
        """
        h, w = image.shape[:2]
        
        # Get text size for spacing calculation (adjusted for scale)
        font = self.font
        font_scale = self.font_scale * 0.9
        thickness = 2
        sample_label = "V99"
        (text_width, text_height), baseline = cv2.getTextSize(sample_label, font, font_scale, thickness)
        detail_height = int(text_height * 0.9)
        
        label_height = text_height + detail_height + 20
        label_width = max(text_width, 100) + 20
        
        # Initialize positions with vein centroids (scaled)
        vein_positions = {}
        
        for i, vein in enumerate(veins):
            spatial = vein.get('spatial_analysis', {})
            cx, cy = spatial.get('vein_centroid', (0, 0))
            # Scale coordinates
            cx = int(cx * scale_factor)
            cy = int(cy * scale_factor)
            vein_positions[i] = (cx, cy)
        
        # Use iterative force-directed layout for better spreading
        final_positions = self._iterative_position_layout(
            vein_positions, w, h, label_width, label_height
        )
        
        return final_positions
    
    @staticmethod
    def _iterative_position_layout(
        vein_positions: Dict[int, Tuple[int, int]],
        w: int, h: int,
        label_width: int, label_height: int,
        iterations: int = 10
    ) -> Dict[int, Tuple[int, int]]:
        """
        Use iterative approach to find non-overlapping positions.
        
        Args:
            vein_positions: Dictionary of vein_index -> (x, y) centroid
            w, h: Image width and height
            label_width, label_height: Label dimensions
            iterations: Number of refinement iterations
        
        Returns:
            Dictionary of vein_index -> (label_x, label_y) non-overlapping positions
        """
        # Strategy: Place labels around each vein in priority order
        # Priority positions: right, up-right, up, up-left, left, down-left, down, down-right
        
        margin = 10  # Pixels from vein centroid to label box edge
        positions = {}
        
        # Sort veins by x-position for consistent processing
        sorted_indices = sorted(vein_positions.keys(), key=lambda i: vein_positions[i][0])
        
        for idx in sorted_indices:
            cx, cy = vein_positions[idx]
            
            # Try priority positions around the vein
            priority_offsets = [
                (label_width + margin, 0),              # Right
                (label_width//2 + margin, -label_height - margin),  # Up-right
                (0, -label_height - margin),            # Up
                (-label_width - margin, -label_height - margin),    # Up-left
                (-label_width - margin, 0),             # Left
                (-label_width - margin, label_height + margin),     # Down-left
                (0, label_height + margin),             # Down
                (label_width//2 + margin, label_height + margin),   # Down-right
            ]
            
            best_pos = None
            best_overlap_count = float('inf')
            
            # Try each position
            for dx, dy in priority_offsets:
                test_x = cx + dx
                test_y = cy + dy
                
                # Clamp to image bounds
                test_x = max(10, min(test_x, w - label_width - 10))
                test_y = max(label_height, min(test_y, h - label_height - 10))
                
                # Count overlaps with already-placed labels
                test_bound = (test_x - 5, test_y - label_height - 5,
                            test_x + label_width, test_y + 5)
                
                overlap_count = 0
                for already_placed_idx in list(positions.keys()):
                    if already_placed_idx >= idx:
                        continue
                    
                    placed_x, placed_y = positions[already_placed_idx]
                    placed_bound = (placed_x - 5, placed_y - label_height - 5,
                                   placed_x + label_width, placed_y + 5)
                    
                    if UltrasoundVisualizer._rectangles_overlap(test_bound, [placed_bound], margin=5):
                        overlap_count += 1
                
                # Choose position with least overlaps
                if overlap_count < best_overlap_count:
                    best_overlap_count = overlap_count
                    best_pos = (test_x, test_y)
            
            if best_pos:
                positions[idx] = best_pos
            else:
                # Fallback: place at vein centroid shifted right
                positions[idx] = (cx + 20, cy)
        
        return positions
    
    @staticmethod
    def _rectangles_overlap(rect: Tuple[int, int, int, int], 
                           other_rects: List[Tuple[int, int, int, int]],
                           margin: int = 15) -> bool:
        """
        Check if rectangle overlaps with any rectangle in the list.
        
        Args:
            rect: (x1, y1, x2, y2) for checking
            other_rects: List of (x1, y1, x2, y2) rectangles
            margin: Minimum spacing between labels
        
        Returns:
            True if overlaps with any rectangle (with margin)
        """
        x1a, y1a, x2a, y2a = rect
        
        for x1b, y1b, x2b, y2b in other_rects:
            # Check with margin for better spacing
            if not (x2a + margin < x1b or x1a - margin > x2b or 
                   y2a + margin < y1b or y1a - margin > y2b):
                return True
        
        return False
    
    @staticmethod
    def _get_color_for_type(vein_type: str) -> Tuple[int, int, int]:
        """Get color for vein type"""
        color_map = {
            "deep_vein": (0, 0, 255),
            "superficial_vein": (0, 255, 0),
            "perforator_vein": (0, 165, 255),
            "gsv": (255, 0, 255),
            "unknown": (128, 128, 128)
        }
        return color_map.get(vein_type, (128, 128, 128))
    
    @staticmethod
    def _get_fascia_y_at_x(fascia_mask: np.ndarray, x: int) -> int:
        """Get Y coordinate of fascia at given X coordinate"""
        if x < 0 or x >= fascia_mask.shape[1]:
            return -1
        
        col = fascia_mask[:, x]
        fascia_points = np.where(col > 0)[0]
        
        if len(fascia_points) > 0:
            return int(np.mean(fascia_points))
        
        return -1
    
    @staticmethod
    def _add_grid_labels(grid: np.ndarray, cell_size: Tuple[int, int]):
        """Add labels to grid cells"""
        h, w = cell_size
        labels = ["Original", "Segmentation", "Classification", "Detailed Analysis"]
        positions = [(10, 30), (w + 10, 30), (10, h + 30), (w + 10, h + 30)]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for label, (x, y) in zip(labels, positions):
            cv2.putText(
                grid, label, (x, y),
                font, 0.7, (255, 255, 255), 2
            )
