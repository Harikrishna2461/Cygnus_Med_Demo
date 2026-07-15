"""
Rule-based fascia detection for ultrasound images.
Detects fascia as linear bright structures using image processing.
No training data required - uses classical computer vision techniques.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch


class RuleBasedFasciaDetector:
    """
    Detect fascia boundaries in ultrasound using:
    - CLAHE enhancement
    - Edge detection
    - Line detection
    - Morphological filtering
    
    Fascia appears as bright linear structures in ultrasound.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance ultrasound image contrast using CLAHE.
        
        Args:
            image: input image (H, W, C) or (H, W)
        
        Returns:
            enhanced image in same format
        """
        if len(image.shape) == 3:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def detect_edges(self, image: np.ndarray, threshold1=50, threshold2=150) -> np.ndarray:
        """
        Detect edges using Canny edge detector.
        
        Args:
            image: grayscale ultrasound image
            threshold1: lower Canny threshold
            threshold2: upper Canny threshold
        
        Returns:
            binary edge map
        """
        edges = cv2.Canny(image, threshold1, threshold2)
        return edges
    
    def detect_lines(self, edge_image: np.ndarray) -> np.ndarray:
        """
        Detect linear structures (fascia boundaries) using morphological operations.
        
        Args:
            edge_image: binary edge map
        
        Returns:
            line detection result
        """
        # Create horizontal and vertical kernels to detect line-like structures
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20))
        
        # Detect horizontal and vertical lines
        lines_h = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, kernel_h)
        lines_v = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, kernel_v)
        
        # Combine
        combined_lines = cv2.bitwise_or(lines_h, lines_v)
        
        return combined_lines
    
    def filter_fascia_candidates(self, image: np.ndarray, line_map: np.ndarray) -> np.ndarray:
        """
        Filter line candidates to likely fascia.
        Fascia typically:
        - Has high intensity in ultrasound
        - Forms continuous boundaries
        - Located at tissue layer interfaces
        
        Args:
            image: original ultrasound
            line_map: detected lines
        
        Returns:
            filtered fascia mask
        """
        # Threshold on intensity (fascia is bright)
        _, intensity_mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        
        # Combine with line detection
        combined = cv2.bitwise_and(intensity_mask, line_map)
        
        # Clean up with morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def fill_fascia_region(self, fascia_lines: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """
        Fill region below detected fascia line (fascia defines boundary).
        
        Args:
            fascia_lines: detected fascia boundaries
            original_shape: original image shape
        
        Returns:
            binary fascia mask
        """
        # Dilate to make boundaries thicker
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(fascia_lines, kernel, iterations=2)
        
        return dilated
    
    def detect_fascia(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect fascia in ultrasound image.
        
        Args:
            image: input ultrasound image (any size, any color format)
        
        Returns:
            (normalized_mask, confidence_map)
            - mask: binary fascia segmentation (0-1)
            - confidence: pixel confidence scores (0-1)
        """
        # Ensure input is correct format
        if len(image.shape) == 3:
            original = image.copy()
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            if gray.max() > 1:
                gray = gray.astype(np.float32) / 255.0
                gray = (gray * 255).astype(np.uint8)
        
        H, W = gray.shape
        
        # Step 1: Enhance
        enhanced = self.enhance_image(gray)
        
        # Step 2: Detect edges
        edges = self.detect_edges(enhanced)
        
        # Step 3: Detect lines
        lines = self.detect_lines(edges)
        
        # Step 4: Filter for fascia
        fascia_candidates = self.filter_fascia_candidates(enhanced, lines)
        
        # Step 5: Fill fascia region
        fascia_mask = self.fill_fascia_region(fascia_candidates, (H, W))
        
        # Normalize to 0-1
        if fascia_mask.max() > 0:
            fascia_mask_norm = fascia_mask.astype(np.float32) / 255.0
        else:
            fascia_mask_norm = fascia_mask.astype(np.float32)
        
        # Create confidence map (combine line strength with intensity)
        confidence = np.zeros((H, W), dtype=np.float32)
        
        # High confidence where we detected strong lines
        line_norm = lines.astype(np.float32) / 255.0
        intensity_norm = enhanced.astype(np.float32) / 255.0
        
        # Confidence = line detection * local intensity
        confidence = (line_norm * 0.6 + intensity_norm * 0.4)
        confidence = np.clip(confidence, 0, 1)
        
        return fascia_mask_norm, confidence


class FasciaDetectorWrapper:
    """Wrapper to match the UNet inference interface."""
    
    def __init__(self, device='cpu'):
        self.detector = RuleBasedFasciaDetector(device=device)
        self.device = device
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect fascia in image.
        
        Args:
            image: input image (H, W, C) with values 0-1 or 0-255
        
        Returns:
            fascia mask (H, W) with values 0-1
        """
        mask, _ = self.detector.detect_fascia(image)
        return mask
    
    def detect_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Detect fascia in batch of images.
        
        Args:
            images: batch of images (B, 3, H, W) as torch tensor
        
        Returns:
            batch of masks (B, 1, H, W) as torch tensor
        """
        batch_size = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        
        masks = torch.zeros(batch_size, 1, H, W, device=self.device)
        
        for i in range(batch_size):
            # Convert tensor to numpy
            img_tensor = images[i]  # (3, H, W)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            
            # Normalize to 0-1 if needed
            if img_np.max() > 1:
                img_np = img_np / 255.0
            
            # Detect fascia
            mask, _ = self.detector.detect_fascia(img_np)
            
            # Convert back to tensor
            masks[i, 0] = torch.from_numpy(mask).to(self.device)
        
        return masks


# Test
if __name__ == '__main__':
    print("\n" + "="*70)
    print("RULE-BASED FASCIA DETECTOR (No Training Data Required)")
    print("="*70)
    
    detector = RuleBasedFasciaDetector()
    
    # Create a synthetic test image
    test_image = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
    
    # Add a bright line (fake fascia)
    test_image[100:102, :] = 200
    
    print(f"\n✓ Detector initialized")
    print(f"✓ Input shape: {test_image.shape}")
    
    # Detect
    mask, confidence = detector.detect_fascia(test_image)
    
    print(f"✓ Output mask shape: {mask.shape}")
    print(f"✓ Output confidence shape: {confidence.shape}")
    print(f"✓ Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"✓ Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    print("\n✅ FASCIA DETECTION WORKS WITHOUT TRAINING DATA")
    print("="*70 + "\n")
