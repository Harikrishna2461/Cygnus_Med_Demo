"""
Segment Anything Model (SAM) Wrapper for Fascia and Vein Segmentation

Provides unified interface for segmenting fascia and veins from ultrasound images.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class SAMSegmenter:
    """
    Wrapper for Segment Anything Model
    
    Supports both hardware-accelerated and CPU-based segmentation
    """
    
    def __init__(self, model_type: str = "vit_b", device: str = "cpu"):
        """
        Initialize SAM segmenter
        
        Args:
            model_type: SAM model variant ('vit_b', 'vit_l', 'vit_h', 'mobile_sam')
                       vit_b is reliable and works with memory optimizations
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self._model_initialized = False
        self.use_fallback = False  # Flag to use classical CV fallback
        self.fallback_detector = None
        # DON'T call _initialize_model here - lazy load on first use
    
    def _ensure_model_loaded(self):
        """Lazily initialize SAM model only when first needed (memory efficient)"""
        if self._model_initialized:
            logger.debug(f"[SAM] Model already initialized (use_fallback={self.use_fallback})")
            return  # Already loaded
        
        logger.info(f"[SAM] First use detected - checking device {self.device}")
        
        # Check if we're on CPU - if so, use fallback immediately (avoid OOM crashes)
        if self.device == "cpu":
            logger.warning("⚠️ SAM on CPU detected - activating fallback detector to avoid memory crashes")
            try:
                from vision.segmentation.fallback_detector import FallbackVeinDetector
                self.fallback_detector = FallbackVeinDetector()
                self.use_fallback = True
                self._model_initialized = True
                logger.info("[SAM] ✓ Fallback detector activated for CPU inference")
                return
            except ImportError as e:
                logger.error(f"[SAM] Failed to import FallbackVeinDetector: {e}")
                pass
        
        logger.info(f"[SAM] Attempting to initialize SAM model on device {self.device}...")
        self._initialize_model()
        self._model_initialized = True
        logger.info("[SAM] Model initialization complete")
    
    def _initialize_model(self):
        """Initialize SAM model with memory optimization for CPU"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import os
            import urllib.request
            import ssl
            import torch
            import gc
            
            logger.info(f"Initializing SAM ({self.model_type}) on {self.device} with memory optimization...")
            
            # Aggressive garbage collection BEFORE loading
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Create weights directory
            os.makedirs("./weights", exist_ok=True)
            
            # Map model to download URLs
            model_urls = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            }
            
            # Use consistent .pth extension
            weights_path = f"./weights/sam_{self.model_type}.pth"
            
            # Download weights if not present
            if not os.path.exists(weights_path):
                if self.model_type not in model_urls:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                
                logger.info(f"Downloading SAM {self.model_type} weights ({os.path.getsize(weights_path) if os.path.exists(weights_path) else 'downloading'}MB)...")
                url = model_urls[self.model_type]
                
                # Bypass SSL verification for downloading (common issue on macOS)
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                try:
                    with urllib.request.urlopen(url, context=ssl_context) as response:
                        with open(weights_path, 'wb') as out_file:
                            out_file.write(response.read())
                    logger.info(f"✓ Downloaded to {weights_path}")
                except Exception as e:
                    logger.error(f"Download failed: {e}")
                    raise
            
            # Load model with memory optimization
            logger.info(f"Loading SAM from {weights_path}...")
            
            # Disable gradient tracking during model load (huge memory savings)
            with torch.no_grad():
                # Load weights on CPU first
                sam = sam_model_registry[self.model_type](checkpoint=weights_path)
                
                # Set to eval mode IMMEDIATELY before any other operations
                sam.eval()
                
                # Move to device IMMEDIATELY
                sam.to(device=self.device)
                
                # Optimize for CPU: use half precision (float16) to save memory
                if self.device == "cpu":
                    logger.info("Converting to float16 for memory efficiency...")
                    try:
                        # Convert ALL module buffers and parameters to float16
                        sam.half()
                    except RuntimeError as e:
                        logger.warning(f"Half precision conversion failed: {e}, using float32")
                else:
                    # For CUDA, just ensure it's on device
                    pass
                
                # Create predictor while still in no_grad context
                self.predictor = SamPredictor(sam)
            
            # Aggressive garbage collection AFTER loading
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"✓ SAM {self.model_type} ready on {self.device} (memory optimized with float16)")
        
        except Exception as e:
            logger.critical(f"FATAL: Cannot initialize SAM: {e}")
            # FALLBACK: Use classical CV detector instead of crashing
            logger.warning("⚠️ SAM initialization failed, activating fallback vein detector...")
            try:
                from vision.segmentation.fallback_detector import FallbackVeinDetector
                self.fallback_detector = FallbackVeinDetector()
                self.use_fallback = True
                logger.info("✓ Fallback detector ready (classical CV techniques)")
                return  # Don't raise exception - use fallback instead
            except ImportError:
                raise RuntimeError(f"SAM initialization failed: {e}")
    
    def segment_fascia(self, image: np.ndarray, point_prompts: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Segment fascia from ultrasound image
        
        Args:
            image: Input image (BGR or RGB)
            point_prompts: Optional list of (x, y) points within fascia region
        
        Returns:
            Binary mask of fascia (0/1)
        """
        logger.info(f"[FASCIA] Segmentation requested on {image.shape} image")
        
        # FIRST: Ensure model is loaded (sets up fallback if needed)
        self._ensure_model_loaded()
        
        logger.info(f"[FASCIA] After model init: use_fallback={self.use_fallback}, has_fallback_detector={self.fallback_detector is not None}")
        
        # THEN: Check if we should use fallback
        if self.use_fallback and self.fallback_detector is not None:
            logger.info("[FASCIA] Using fallback detector (classical CV)")
            return self.fallback_detector.segment_fascia(image)
        
        if self.predictor is None:
            raise RuntimeError("SAM model not initialized")
        
        import torch
        
        # Use no_grad context to disable gradient computation (saves memory)
        with torch.no_grad():
            # CRITICAL: Aggressively downsample image to 256 max dimension for CPU
            # This is the single biggest memory optimization
            h, w = image.shape[:2]
            max_dim = 256  # Ultra-aggressive downsampling for 14GB laptop
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(f"Downsampled image: {h}x{w} → {new_h}x{new_w} for SAM inference")
            else:
                image_resized = image
            
            # Convert image to RGB if needed
            if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
                if image_resized.max() > 1:  # BGR range
                    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_resized
            else:
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
            
            # Set image for SAM
            self.predictor.set_image(image_rgb)
            
            # Default: segment middle-lower fascia region if no prompts given
            if point_prompts is None:
                h_resized, w_resized = image_resized.shape[:2]
                # Point in middle of image, lower half (typical fascia location)
                point_prompts = [(w_resized // 2, int(h_resized * 0.65))]
            else:
                # Scale prompts to resized image coordinates if image was downsampled
                h_orig, w_orig = h, w
                h_resized, w_resized = image_resized.shape[:2]
                if h_orig != h_resized or w_orig != w_resized:
                    scale_h = h_resized / h_orig
                    scale_w = w_resized / w_orig
                    point_prompts = [(int(x * scale_w), int(y * scale_h)) for x, y in point_prompts]
            
            # Convert prompts to numpy array
            points = np.array(point_prompts)
            labels = np.ones(len(point_prompts))
            
            # Get predictions
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
            
            # Return best mask (highest confidence)
            return masks[0].astype(np.uint8) * 255
    
    def segment_veins(self, image: np.ndarray, fascia_mask: np.ndarray = None, num_masks: int = 5) -> List[Dict]:
        """
        Segment multiple veins from ultrasound image using smart prompting.
        
        Strategy: Find dark blobs (actual veins), use them as point prompts for SAM,
        then get accurate segmentations. Avoids random background noise.
        
        Args:
            image: Input image
            fascia_mask: Optional fascia mask to guide vein detection
            num_masks: Unused (kept for API compatibility)
        
        Returns:
            List of vein segmentation results with metadata
        """
        logger.info(f"[VEINS] Smart prompt-based segmentation on {image.shape} image")
        
        # FIRST: Ensure model is loaded (sets up fallback if needed)
        self._ensure_model_loaded()
        
        # THEN: Check if we should use fallback
        if self.use_fallback and self.fallback_detector is not None:
            logger.info("[VEINS] Using fallback detector (classical CV)")
            return self.fallback_detector.segment_veins(image, fascia_mask=fascia_mask, num_masks=num_masks)
        
        if self.predictor is None:
            raise RuntimeError("SAM model not initialized")
        
        import torch
        
        # Use no_grad context to disable gradient computation (saves memory)
        with torch.no_grad():
            # CRITICAL: Aggressively downsample image to 256 max dimension for CPU
            h, w = image.shape[:2]
            max_dim = 256  # Ultra-aggressive downsampling for 14GB laptop
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(f"Downsampled for vein segmentation: {h}x{w} → {new_h}x{new_w}")
            else:
                image_resized = image
            
            # Convert BGR to RGB
            if len(image_resized.shape) == 3:
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_resized
            
            # STEP 1: Find dark blobs (actual veins on ultrasound)
            dark_blobs = self._find_dark_blob_prompts(image_resized)
            logger.info(f"[VEINS] Found {len(dark_blobs)} dark blob candidates for prompting")
            
            if len(dark_blobs) == 0:
                logger.warning("[VEINS] No dark blobs found - falling back to auto-segmentation")
                self.predictor.set_image(image_rgb)
                masks, scores, _ = self.predictor.predict(multimask_output=True)
                vein_masks = []
                for mask, score in zip(masks, scores):
                    if self._is_vein_like(mask):
                        vein_masks.append({
                            "mask": mask.astype(np.uint8),
                            "confidence": float(score),
                            "vein_id": f"V{len(vein_masks)}",
                            "properties": self._compute_mask_properties(mask)
                        })
                logger.info(f"✓ Auto-segmented {len(vein_masks)} veins")
                return vein_masks
            
            # STEP 2: Use blob centers as prompts for SAM
            self.predictor.set_image(image_rgb)
            vein_masks = []
            processed_centroids = set()
            
            for blob_center in dark_blobs:
                # Avoid duplicate prompts within 20 pixels
                cx, cy = blob_center
                skip = False
                for px, py in processed_centroids:
                    if abs(cx - px) < 20 and abs(cy - py) < 20:
                        skip = True
                        break
                if skip:
                    continue
                
                processed_centroids.add((cx, cy))
                
                try:
                    # Prompt SAM with blob center point
                    points = np.array([[cx, cy]])
                    labels = np.array([1])
                    
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=False
                    )
                    
                    mask = masks[0]
                    score = scores[0]
                    
                    # Strict filtering: only accept if it looks like a real vein
                    if self._is_vein_like(mask) and score > 0.5:
                        vein_masks.append({
                            "mask": mask.astype(np.uint8),
                            "confidence": float(score),
                            "vein_id": f"V{len(vein_masks)}",
                            "properties": self._compute_mask_properties(mask)
                        })
                        logger.info(f"  ✓ Vein {len(vein_masks)} (confidence={score:.2f})")
                    else:
                        logger.debug(f"  ✗ Rejected (is_vein_like={self._is_vein_like(mask)}, score={score:.2f})")
                
                except Exception as e:
                    logger.warning(f"  Could not prompt SAM at {blob_center}: {e}")
            
            logger.info(f"✓ Smart prompt segmentation: {len(vein_masks)} valid veins")
            return vein_masks
    
    def _detect_fascia_hints(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect hint points for fascia using image analysis
        
        Fascia typically appears as a bright linear structure
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find prominent horizontal lines (fascia often appears as linear structure)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        hints = []
        if lines is not None:
            for line in lines[:3]:  # Use top 3 lines
                x1, y1, x2, y2 = line[0]
                # Take middle point of line
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                hints.append((cx, cy))
        
        return hints if hints else []
    
    def _find_dark_blob_prompts(self, image: np.ndarray, min_area: int = 200, max_area: int = 40000) -> List[Tuple[int, int]]:
        """
        Find dark blob regions (veins) on ultrasound.
        
        Veins appear as dark (hypoechoic) regions. This method:
        1. Converts to grayscale
        2. Inverts (dark → bright)
        3. Applies threshold to find blob regions
        4. Filters by size and shape
        5. Returns centroids as point prompts for SAM
        
        Args:
            image: Input grayscale or color image
            min_area: Minimum blob size in pixels
            max_area: Maximum blob size in pixels
        
        Returns:
            List of (x, y) centroids for dark blobs
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Invert image so dark regions (veins) become bright
        inverted = 255 - gray
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        
        # Threshold to get binary mask of dark regions
        # Use Otsu's method automatically
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find connected components
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        prompts = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filter: must be reasonable (not noise, not huge backgrounds)
            if area < min_area or area > max_area:
                continue
            
            # Fit ellipse for shape analysis
            if len(contour) < 5:
                continue
            
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major_axis, minor_axis), _ = ellipse
            
            # Check aspect ratio (veins are somewhat elongated but not extreme)
            aspect_ratio = major_axis / (minor_axis + 1e-6)
            if aspect_ratio < 1.1 or aspect_ratio > 6.0:
                continue
            
            # Check solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.3:  # Reject fragmented shapes
                    continue
            
            # This is a good vein-like blob - use centroid as prompt
            prompts.append((int(cx), int(cy)))
        
        logger.debug(f"[DARK_BLOBS] Found {len(prompts)} dark blob candidates")
        return prompts
    
    def _is_vein_like(self, mask: np.ndarray, min_area: int = 500) -> bool:
        """
        Check if mask resembles a vein
        
        Veins are typically:
        - Medium to large size (not noise, not huge artifacts)
        - Elongated / tubular shape
        - Solidity > 0.5 (compact, not fragmented)
        """
        area = np.sum(mask)
        
        # Much stricter: min 500 pixels (was 50)
        if area < min_area or area > 50000:
            return False
        
        # Compute elongation using contours
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Compute solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        
        # Reject fragmented or irregular shapes
        if solidity < 0.4:
            return False
        
        # Fit ellipse if enough points
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (major_axis, minor_axis), _ = ellipse
            
            # Veins are typically more elongated (high aspect ratio)
            # But not extremely elongated (that's a line, not a vein)
            aspect_ratio = major_axis / (minor_axis + 1e-6)
            is_tubular = 1.3 < aspect_ratio < 5.0  # Stricter bounds
            
            return is_tubular
        
        return False  # Reject if not enough points to determine shape
    
    def _compute_mask_properties(self, mask: np.ndarray) -> Dict:
        """Compute shape properties of mask"""
        area = np.sum(mask)
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        properties = {
            "area": int(area),
            "centroid": self._compute_centroid(mask)
        }
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            properties["perimeter"] = float(cv2.arcLength(contour, True))
            properties["num_points"] = len(contour)
        
        return properties
    
    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of binary mask"""
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return (0, 0)
        
        cx = float(np.mean(x_coords))
        cy = float(np.mean(y_coords))
        
        return (cx, cy)
