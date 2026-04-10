"""
Deep Learning Fascia Detector with Blob Classification
Two-stage approach: Fascia segmentation -> Vein blob detection -> Classification
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class SimpleUNet(nn.Module):
    """Lightweight U-Net for fascia segmentation (can be pretrained)"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(64, 128)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Output
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1_pool = self.pool1(e1)
        e2 = self.enc2(e1_pool)
        e2_pool = self.pool2(e2)
        
        # Bottleneck
        bn = self.bottleneck(e2_pool)
        
        # Decoder
        d2 = self.upconv2(bn)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out(d1)
        return torch.sigmoid(out)


class DeepLearningFasciaDetector:
    """
    Deep learning-based fascia and vein detector
    Stage 1: Segment fascia using U-Net
    Stage 2: Detect veins below fascia
    Stage 3: Classify vein types
    """
    
    def __init__(self):
        """Initialize with pretrained or untrained U-Net"""
        logger.info("[DL-Detector] Initializing deep learning fascia detector...")
        
        # Try to use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[DL-Detector] Using device: {self.device}")
        
        # Initialize U-Net (can be pretrained later)
        self.fascia_model = SimpleUNet(in_channels=1, out_channels=1).to(self.device)
        self.fascia_model.eval()  # Inference mode
        
        logger.info("[DL-Detector] ✓ Detector ready (pretrained model can be loaded)")
    
    def load_pretrained_fascia_model(self, model_path: str):
        """Load a pretrained fascia segmentation model"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.fascia_model.load_state_dict(state_dict)
            logger.info(f"[DL-Detector] Loaded pretrained fascia model from {model_path}")
        except Exception as e:
            logger.warning(f"[DL-Detector] Could not load pretrained model: {e}")
    
    def segment_fascia_dl(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment fascia using deep learning
        Returns: (fascia_mask, fascia_y)
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize to [0, 1]
        gray_norm = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        # Resize to model input size (256x256 for efficiency)
        input_size = 256
        resized = cv2.resize(gray_norm, (input_size, input_size))
        
        # Convert to tensor
        tensor_input = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.fascia_model(tensor_input)
        
        # Convert back to numpy
        pred = output.cpu().squeeze().numpy()
        
        # Resize back to original size
        fascia_seg = cv2.resize(pred, (w, h))
        
        # Threshold
        _, fascia_mask = cv2.threshold(fascia_seg, 0.5, 255, cv2.THRESH_BINARY)
        fascia_mask = fascia_mask.astype(np.uint8)
        
        # Find fascia y-position
        row_sums = np.sum(fascia_mask > 0, axis=1)
        if np.max(row_sums) > 0:
            fascia_y = int(np.argmax(row_sums))
        else:
            fascia_y = h // 2
        
        logger.info(f"[DL-Fascia] Segmented fascia at y≈{fascia_y}")
        
        return fascia_mask, fascia_y
    
    def detect_veins_below_fascia(self, image: np.ndarray, fascia_y: int) -> List[Dict]:
        """
        Detect vein blobs below the fascia using adaptive thresholding
        Args:
            image: Original ultrasound image
            fascia_y: Y position of fascia from DL segmentation
        
        Returns:
            List of vein detections with bounding boxes
        """
        h, w = image.shape[:2]
        
        # Get region of interest below fascia
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Invert (veins are dark -> bright after inversion)
        gray_inv = cv2.bitwise_not(gray)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
        enhanced = clahe.apply(gray_inv)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, blockSize=19, C=7)
        
        # Morphological cleanup
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours (blobs)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        veins = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Only consider blobs below fascia and with reasonable size
            if y > fascia_y and 380 <= area <= 5200:
                solidity = area / cv2.contourArea(cv2.convexHull(cnt))
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
                aspect = float(bw) / max(bh, 1)
                
                # Apply filters
                if solidity > 0.54 and circularity > 0.32 and aspect < 7.0:
                    veins.append({
                        'bbox': (x, y, bw, bh),
                        'area': area,
                        'solidity': solidity,
                        'circularity': circularity,
                        'y_pos': y
                    })
        
        # Sort by y position (top to bottom)
        veins = sorted(veins, key=lambda v: v['y_pos'])
        
        logger.info(f"[DL-Veins] Detected {len(veins)} vein candidates")
        
        return veins
    
    def classify_veins(self, veins: List[Dict]) -> List[Dict]:
        """
        Classify detected veins by position (TRB, N1, N2, etc.)
        """
        classified = []
        
        for i, vein in enumerate(veins):
            if i == 0:
                label = "N1"  # First vein
            elif i == 1:
                label = "N2"  # Second vein
            else:
                label = f"ON{i+1}"  # Other nerves
            
            vein['label'] = label
            classified.append(vein)
        
        return classified
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Full detection pipeline: Fascia segmentation -> Vein detection -> Classification
        """
        h, w = image.shape[:2]
        logger.info(f"[DL-Pipeline] Processing {h}x{w} image...")
        
        # Stage 1: Segment fascia with DL
        fascia_mask, fascia_y = self.segment_fascia_dl(image)
        
        # Stage 2: Detect veins below fascia
        vein_candidates = self.detect_veins_below_fascia(image, fascia_y)
        
        # Stage 3: Classify veins
        veins = self.classify_veins(vein_candidates[:2])  # Keep top 2
        
        # Create visualization
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw fascia
        cv2.line(vis_image, (0, fascia_y), (w, fascia_y), (0, 255, 255), 2)
        
        # Draw veins
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue
        for i, vein in enumerate(veins):
            x, y, bw, bh = vein['bbox']
            color = colors[i % len(colors)]
            cv2.rectangle(vis_image, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(vis_image, vein['label'], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return {
            'fascia_y': int(fascia_y),
            'fascia_mask': fascia_mask,
            'veins': veins,
            'visualization': vis_image
        }
