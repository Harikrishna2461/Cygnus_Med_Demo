"""
Vision Language Model (VLM) based Vein Classifier with RAG Integration

Uses LLaVA (via Ollama) to classify veins in ultrasound images with medical knowledge.
Integrates with RAG system for evidence-based classifications.
"""

import logging
import cv2
import numpy as np
import json
import requests
import base64
from typing import Dict, List, Tuple, Optional
from io import BytesIO
import os

logger = logging.getLogger(__name__)


class VLMVeinClassifier:
    """
    Classify veins using Vision Language Model (LLaVA via Ollama)
    with RAG-enhanced medical knowledge.
    """

    # Classification rules
    CLASSIFICATION_RULES = {
        'N1_deep': {
            'label': 'Deep Vein',
            'color': (0, 255, 0),  # Green
            'description': 'Vein below the fascia layer'
        },
        'N2_gsv': {
            'label': 'GSV/N2',
            'color': (255, 0, 255),  # Magenta
            'description': 'Great Saphenous Vein - within or very near fascia'
        },
        'N3_superficial': {
            'label': 'Superficial Vein',
            'color': (0, 165, 255),  # Orange
            'description': 'Vein above fascia, near skin surface'
        },
        'TRB': {
            'label': 'Tributary',
            'color': (0, 255, 255),  # Cyan
            'description': 'Tributary branch vessel'
        }
    }

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        vlm_model: str = None,
        enable_rag: bool = True,
        rag_retriever=None
    ):
        """
        Initialize VLM classifier
        
        Args:
            ollama_base_url: Base URL for Ollama service
            vlm_model: Vision model name (llava:7b, llava:13b, mistral, etc). If None, auto-selects available model
            enable_rag: Whether to use RAG for medical knowledge
            rag_retriever: Optional RAG retriever instance
        """
        self.ollama_base_url = ollama_base_url
        self.enable_rag = enable_rag
        self.rag_retriever = rag_retriever
        
        # Auto-select model if not specified
        if vlm_model is None:
            self.vlm_model = self._select_available_model()
        else:
            self.vlm_model = vlm_model
        
        logger.info(f"Using VLM model: {self.vlm_model}")

    def _select_available_model(self):
        """Auto-select best available vision model"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                logger.info(f"Available models: {model_names}")
                
                # Preference order - vision models first!
                preference = ['llava', 'mistral', 'llama3.2', 'neural-chat']
                for pref in preference:
                    for model in model_names:
                        if pref in model.lower():
                            logger.info(f"✓ Auto-selected model: {model}")
                            return model
                
                # Fallback: use first available
                if model_names:
                    logger.info(f"✓ Using fallback model: {model_names[0]}")
                    return model_names[0]
                
                logger.warning("⚠ No models available in Ollama")
                return "llava"
            else:
                logger.warning("⚠ Could not query Ollama models, using llava")
                return "llava"
        except Exception as e:
            logger.warning(f"⚠ Model selection failed: {e}, using llava")
            return "llava"

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 for VLM API"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode()

    def _get_rag_context(self, query: str) -> str:
        """Get medical knowledge from RAG system"""
        if not self.enable_rag or self.rag_retriever is None:
            return ""

        try:
            results = self.rag_retriever(query, top_k=2)
            context = "\n".join([f"- {r.get('content', '')[:200]}" for r in results])
            return context
        except Exception as e:
            logger.warning(f"⚠ RAG retrieval failed: {e}")
            return ""

    def classify_veins_in_image(
        self,
        image: np.ndarray,
        fascia_mask: Optional[np.ndarray] = None,
        fascia_center_y: Optional[int] = None
    ) -> Dict:
        """
        Classify all veins in ultrasound image using VLM vision analysis
        
        Uses LLaVA to directly identify veins from the ultrasound image.
        
        Args:
            image: Input ultrasound image (BGR)
            fascia_mask: Optional binary mask of fascia region
            fascia_center_y: Y-coordinate of fascia center line
        
        Returns:
            Dictionary with:
            - detected_veins: List of vein detections with classifications
            - annotated_image: Image with bounding boxes and labels
            - summary: Classification summary
            - raw_vlm_response: Raw VLM analysis (debug)
        """
        logger.info("Starting LLaVA-based vein classification with direct image analysis")

        # Step 1: Get VLM to directly identify veins in the image
        vlm_analysis = self._get_vlm_vein_detection(image, fascia_center_y)
        
        if not vlm_analysis.get("veins"):
            logger.warning("VLM could not identify any veins in the image")
            return {
                "status": "no_veins_detected",
                "detected_veins": [],
                "annotated_image": image,
                "summary": {
                    "total_veins": 0,
                    "by_type": {}
                },
                "raw_vlm_response": vlm_analysis.get("response", "")
            }

        # Step 2: Convert VLM detections to classification format
        classifications = self._vlm_detections_to_classifications(vlm_analysis, fascia_center_y)

        # Step 3: Create annotated image with VLM-identified veins
        annotated_image = self._create_annotated_image(image, classifications)

        # Step 4: Create summary
        summary = self._create_summary(classifications)

        return {
            "status": "success",
            "detected_veins": [self._classification_to_dict(c) for c in classifications],
            "annotated_image": annotated_image,
            "summary": summary,
            "raw_vlm_response": vlm_analysis.get("response", "")
        }

    def _detect_blobs(self, image: np.ndarray, min_area: int = 30) -> List[Dict]:
        """
        Detect blobs/veins in ultrasound image with relaxed parameters
        
        Returns:
            List of blob detections with center, radius, area
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply strong contrast enhancement for ultrasound
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Blob detection with relaxed parameters for ultrasound
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area  # Relaxed from 50
        params.maxArea = image.shape[0] * image.shape[1] // 2
        params.filterByCircularity = True
        params.minCircularity = 0.4  # Relaxed from 0.5
        params.filterByInertia = True
        params.minInertiaRatio = 0.3  # Relaxed from 0.4
        params.filterByColor = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(enhanced)
        
        # Also try alternative detection method if no blobs found
        if not keypoints:
            logger.info("No blobs found with SimpleBlobDetector, trying edge detection")
            # Try edge-based detection
            edges = cv2.Canny(enhanced, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > min_area:
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        radius = int(np.sqrt(area / np.pi))
                        keypoints.append(type('KP', (), {'pt': (cx, cy), 'size': radius * 2})())
        
        blobs = []
        for i, kp in enumerate(keypoints):
            blobs.append({
                'id': i,
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'radius': int(kp.size / 2) if hasattr(kp, 'size') else 10,
                'keypoint': kp
            })
        
        return blobs

    def _apply_rule_based_classification(
        self,
        blobs: List[Dict],
        fascia_mask: Optional[np.ndarray],
        fascia_center_y: Optional[int]
    ) -> List[Dict]:
        """
        Apply spatial rules to classify veins
        
        Rules:
        - Within fascia (±10px) or very near = N2 (GSV)
        - Above fascia, closer to top = N3 (Superficial)
        - Below fascia = N1 (Deep)
        """
        classifications = []
        
        for blob in blobs:
            center_y = blob['center'][1]
            
            if fascia_center_y is None:
                # No fascia info - classify by position
                if center_y < len(blobs) // 3:  # heuristic
                    vein_type = 'N3_superficial'
                    position = 'upper'
                else:
                    vein_type = 'N1_deep'
                    position = 'lower'
                distance_to_fascia = None
            else:
                distance = center_y - fascia_center_y
                
                if abs(distance) <= 10:  # Within 10 pixels
                    vein_type = 'N2_gsv'
                    position = 'within_fascia'
                    confidence_base = 0.8
                elif distance < 0:
                    vein_type = 'N3_superficial'
                    position = 'above_fascia'
                    confidence_base = 0.8
                else:
                    vein_type = 'N1_deep'
                    position = 'below_fascia'
                    confidence_base = 0.8
                
                distance_to_fascia = distance
            
            # Check if in fascia region (extra confidence)
            in_fascia_region = False
            if fascia_mask is not None:
                cx, cy = blob['center']
                if 0 <= cy < fascia_mask.shape[0] and 0 <= cx < fascia_mask.shape[1]:
                    if fascia_mask[cy, cx] > 0:
                        in_fascia_region = True
            
            classifications.append({
                'blob_id': blob['id'],
                'center': blob['center'],
                'radius': blob['radius'],
                'vein_type': vein_type,
                'vein_label': self.CLASSIFICATION_RULES[vein_type]['label'],
                'position': position,
                'distance_to_fascia': distance_to_fascia,
                'in_fascia_region': in_fascia_region,
                'confidence': 0.7,  # Will be enhanced
                'color': self.CLASSIFICATION_RULES[vein_type]['color']
            })
        
        return classifications

    def _get_vlm_analysis(
        self,
        image: np.ndarray,
        blobs: List[Dict],
        fascia_mask: Optional[np.ndarray],
        fascia_center_y: Optional[int]
    ) -> Dict:
        """
        Get VLM analysis of the ultrasound image
        
        Uses Vision Language Model (LLaVA) to:
        1. Identify anatomical structures
        2. Confirm vein classifications
        3. Provide medical reasoning
        """
        
        # Vision model analysis
        try:
            # Create visualization with blobs marked
            viz_image = image.copy()
            for blob in blobs:
                cv2.circle(viz_image, blob['center'], blob['radius'], (0, 255, 0), 2)
            
            # Encode image
            image_b64 = self._encode_image_to_base64(viz_image)
            
            # Build prompt with RAG context
            rag_context = self._get_rag_context("vein classification fascia ultrasound")
            
            prompt = f"""You are an expert ultrasound technician analyzing a vein ultrasound image.

Basic Data Format : 
    - The input is a stream of live video which is segmented into images.

MEDICAL KNOWLEDGE:
{rag_context}

CLASSIFICATION RULES:
- Veins WITHIN or very near fascia (within ~10pixels) = GSV/N2 (Great Saphenous Vein)
- Veins ABOVE fascia and closer to the top of image = Superficial Veins/N3
- Veins BELOW fascia = Deep Veins/N1
- Assign IDs at the start of the stream to each of the veins and track them across frames (e.g., N1, N2 etc for all the veins) and the classification should not change for the veins later in video.Basically the vein classification should be remembered.

TASK:
1. Identify the fascia layer in this image
2. Locate each marked blob and classify based on position to fascia
3. For each vein, provide:
    - Location (x, y approximate)
    - Classification (N1/N2/N3)
    - Confidence (0-100%)
    - Brief reasoning

Respond in JSON format:
{{
    "fascia_identified": true/false,
    "fascia_y_estimate": y_coordinate,
    "veins": [
        {{"location": "x,y", "type": "N1/N2/N3", "confidence": 85, "reasoning": "..."}}
    ]
}}"""
            
            # Call VLM with timeout
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.vlm_model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Try to extract JSON
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        vlm_json = json.loads(json_str)
                        logger.info("✓ VLM analysis successful")
                        return {
                            "response": response_text,
                            "parsed": vlm_json,
                            "status": "success"
                        }
                except json.JSONDecodeError:
                    logger.warning("⚠ Could not parse VLM JSON response")
                    return {
                        "response": response_text,
                        "parsed": {},
                        "status": "parsing_failed"
                    }
            else:
                logger.warning(f"⚠ VLM API error: {response.status_code}")
                return {"status": "api_error"}
        
        except Exception as e:
            logger.warning(f"⚠ VLM analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    def _enhance_with_vlm(
        self,
        classifications: List[Dict],
        vlm_analysis: Dict,
        image: np.ndarray
    ) -> List[Dict]:
        """
        Enhance rule-based classifications with VLM predictions
        
        Increases confidence for classifications that match VLM analysis
        """
        vlm_data = vlm_analysis.get('parsed', {})
        vlm_veins = vlm_data.get('veins', [])
        
        for classification in classifications:
            base_confidence = classification['confidence']
            
            # Try to match with VLM prediction
            for vlm_vein in vlm_veins:
                vlm_type = vlm_vein.get('type', '')
                vlm_confidence = vlm_vein.get('confidence', 50) / 100.0
                
                # If VLM agrees on type, boost confidence
                if vlm_type.lower() in classification['vein_type'].lower():
                    classification['confidence'] = min(
                        0.99,
                        base_confidence * 0.7 + vlm_confidence * 0.3
                    )
                    classification['vlm_enhanced'] = True
                    break
        
        return classifications

    def _create_annotated_image(
        self,
        image: np.ndarray,
        classifications: List[Dict]
    ) -> np.ndarray:
        """
        Create annotated image with bounding boxes and labels
        """
        annotated = image.copy()
        
        for classification in classifications:
            center = (int(classification['center'][0]), int(classification['center'][1]))
            radius = int(classification['radius'])
            color = classification['color']
            label = classification['vein_label']
            confidence = classification['confidence']
            
            # Draw circle around vein
            cv2.circle(annotated, center, radius, color, 2)
            cv2.circle(annotated, center, 3, color, -1)
            
            # Draw label with background for better visibility
            label_text = f"{label}"
            conf_text = f"{int(confidence * 100)}%"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size
            label_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            conf_size = cv2.getTextSize(conf_text, font, 0.5, 1)[0]
            
            # Position text near blob
            text_x = center[0] + radius + 10
            text_y = center[1] - 5
            
            # Draw background rectangle for label
            padding = 4
            cv2.rectangle(
                annotated,
                (text_x - padding, text_y - label_size[1] - padding),
                (text_x + label_size[0] + padding, text_y + padding),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
            
            # Draw confidence below label
            conf_y = text_y + 20
            cv2.putText(
                annotated,
                conf_text,
                (text_x, conf_y),
                font,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated

    def _create_summary(self, classifications: List[Dict]) -> Dict:
        """Create summary of classifications"""
        summary = {
            'total_veins': len(classifications),
            'by_type': {}
        }
        
        for vein_type in self.CLASSIFICATION_RULES.keys():
            count = len([c for c in classifications if c['vein_type'] == vein_type])
            if count > 0:
                summary['by_type'][vein_type] = {
                    'count': count,
                    'label': self.CLASSIFICATION_RULES[vein_type]['label']
                }
        
        # Average confidence
        if classifications:
            avg_conf = sum(c['confidence'] for c in classifications) / len(classifications)
            summary['average_confidence'] = round(avg_conf, 2)
        
        return summary

    def _get_vlm_vein_detection(
        self,
        image: np.ndarray,
        fascia_center_y: Optional[int]
    ) -> Dict:
        """
        Use LLaVA to directly identify and locate veins in the ultrasound image.
        
        Returns coordinates and classifications from VLM analysis.
        """
        try:
            # Encode image
            image_b64 = self._encode_image_to_base64(image)
            height, width = image.shape[:2]
            
            # Get medical context
            rag_context = self._get_rag_context("vein classification fascia ultrasound GSV deep superficial")
            
            prompt = f"""You are an expert ultrasound technician analyzing a vein ultrasound image.

CLINICAL CONTEXT FOR GSV IDENTIFICATION:
- GSV (Great Saphenous Vein): Large tubular dark structure, often horizontal. Located at/near fascia (usually middle of image).
- Deep Veins (N1): Located deeper, at bottom of image. Smaller, less prominent.
- Superficial Veins (N3): Located near surface, at top of image.
- Look for: tubular structures, horizontal lines, paired walls indicating vessels.

IMAGE DIMENSIONS: {width}x{height} pixels

Identify ALL veins. For each, describe:
- Region: top/middle/bottom AND left/center/right (MIDDLE VEINS ARE OFTEN GSV!)
- Size: small/medium/large
- Type: "vein", "superficial_vein", "deep_vein", or "GSV" 
- Shape: tubular/elongated/oval
- Visibility: clear/moderate/dim
- Description: specific visual details

JSON Response (no markdown):
{{
    "analysis": "brief overall analysis of what you see",
    "vessels": [
        {{
            "region": "description of location (e.g., 'bottom right corner', 'center of image', 'upper left area')",
            "size": "small/medium/large",
            "type": "vein" or "superficial_vein" or "deep_vein" or "GSV",
            "shape": "tubular/elongated/round - describe the actual shape observed",
            "visibility": "clear/moderate/dim",
            "description": "specific visual description of this vessel"
        }},
        ...
    ],
    "fascia_estimate": "describe where fascia appears to be if visible (e.g., 'line at middle of image', 'not clearly visible')"
}}"""
            
            logger.info(f"Sending image to LLaVA for vein detection (model: {self.vlm_model})")
            
            # Try with extended timeout for vision models (they are slow)
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.vlm_model,
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False,
                        "temperature": 0.1  # Low temp for consistent output
                    },
                    timeout=240  # Vision models need time: Extended from 120 to 240 seconds
                )
            except requests.exceptions.Timeout:
                logger.warning(f"⚠ LLaVA timed out. Falling back to faster model...")
                # Fall back to image processing only
                fallback_vessels = self._detect_veins_with_image_processing(image)
                return {
                    "response": "LLaVA timeout - using image processing fallback",
                    "veins": fallback_vessels,
                    "status": "timeout_fallback" if fallback_vessels else "no_vessels"
                }
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                logger.info(f"LLaVA response received ({len(response_text)} chars)")
                
                # Extract JSON from response - handle markdown code blocks
                try:
                    import re
                    # Remove markdown code blocks if present
                    json_text = response_text
                    if '```' in json_text:
                        # Extract content between ``` markers
                        match = re.search(r'```(?:json)?\s*({.*?})\s*```', json_text, re.DOTALL)
                        if match:
                            json_text = match.group(1)
                    
                    # Parse JSON
                    json_start = json_text.find('{')
                    json_end = json_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = json_text[json_start:json_end]
                        vlm_data = json.loads(json_str)
                        
                        # Convert descriptive vessels to coordinate-based detections
                        # Use image processing to find actual vessels based on LLaVA's descriptions
                        vessels = self._convert_llava_descriptions_to_detections(
                            vlm_data.get('vessels', []),
                            image,
                            vlm_data.get('fascia_estimate', '')
                        )
                        
                        logger.info(f"✓ LLaVA identified {len(vessels)} vessel regions")
                        return {
                            "response": response_text,
                            "veins": vessels,
                            "fascia_analysis": vlm_data.get('fascia_estimate'),
                            "status": "success"
                        }
                except Exception as e:
                    logger.warning(f"⚠ Issue processing LLaVA response: {e}")
                    logger.warning(f"Raw response: {response_text[:500]}")
                    # Fall back to using image processing
                    fallback_vessels = self._detect_veins_with_image_processing(image)
                    return {
                        "response": response_text,
                        "veins": fallback_vessels,
                        "status": "partial" if fallback_vessels else "no_vessels"
                    }
            else:
                logger.error(f"⚠ LLaVA API error: {response.status_code}")
                # Use image processing fallback
                fallback_vessels = self._detect_veins_with_image_processing(image)
                return {"status": "api_error_fallback", "veins": fallback_vessels}
        
        except Exception as e:
            logger.error(f"⚠ VLM vein detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Final fallback to image processing
            fallback_vessels = self._detect_veins_with_image_processing(image)
            return {
                "status": "error_fallback" if fallback_vessels else "error",
                "error": str(e),
                "veins": fallback_vessels if fallback_vessels else []
            }
    
    def _vlm_detections_to_classifications(
        self,
        vlm_analysis: Dict,
        fascia_center_y: Optional[int]
    ) -> List[Dict]:
        """
        Convert VLM regional vein detections to classification format.
        Handles both direct coordinate detections and region-based detections.
        """
        classifications = []
        vlm_veins = vlm_analysis.get('veins', [])
        
        logger.info(f"Converting {len(vlm_veins)} VLM detections to classifications")
        
        for i, vein in enumerate(vlm_veins):
            try:
                # Extract data from VLM response
                cx = vein.get('center_x', 0)
                cy = vein.get('center_y', 0)
                diameter = vein.get('diameter', 40)
                vlm_type = vein.get('type', 'N2_gsv')
                vlm_confidence = vein.get('confidence', 0.5) / 100.0  # Convert from 0-100 to 0-1
                description = vein.get('description', '')
                
                # Use provided fascia center or estimate from image
                if fascia_center_y is None:
                    fascia_center_y = 200  # Default estimate
                
                # Calculate distance to fascia
                distance_to_fascia = cy - fascia_center_y
                in_fascia_region = abs(distance_to_fascia) < (diameter / 2 + 20)
                
                # Classification
                classification = {
                    'blob_id': i,
                    'center': (cx, cy),
                    'radius': diameter / 2,
                    'vein_type': vlm_type,
                    'vein_label': self.CLASSIFICATION_RULES.get(vlm_type, {}).get('label', vlm_type),
                    'position': self._get_position_label(vlm_type, distance_to_fascia),
                    'distance_to_fascia': distance_to_fascia,
                    'in_fascia_region': in_fascia_region,
                    'confidence': min(0.95, vlm_confidence),
                    'color': self.CLASSIFICATION_RULES.get(vlm_type, {}).get('color', (255, 255, 255)),
                    'vlm_description': description
                }
                
                classifications.append(classification)
                logger.info(f"  Vein {i}: {vlm_type} at ({cx}, {cy}) confidence={vlm_confidence:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to process VLM vein {i}: {e}")
        
        return classifications
        
        for i, vein_data in enumerate(vlm_veins):
            try:
                center_x = vein_data.get('center_x', 0)
                center_y = vein_data.get('center_y', 0)
                diameter = vein_data.get('diameter', 10)
                vlm_type = vein_data.get('type', 'N2_gsv')
                vlm_confidence = vein_data.get('confidence', 50) / 100.0
                description = vein_data.get('description', '')
                
                # Determine distance to fascia
                distance_to_fascia = None
                if fascia_center_y is not None:
                    distance_to_fascia = center_y - fascia_center_y
                
                classification = {
                    'blob_id': i,
                    'center':  (int(center_x), int(center_y)),
                    'radius': int(diameter / 2),
                    'vein_type': vlm_type,
                    'vein_label': self.CLASSIFICATION_RULES.get(vlm_type, {}).get('label', vlm_type),
                    'position': self._get_position_label(vlm_type, distance_to_fascia),
                    'distance_to_fascia': distance_to_fascia,
                    'in_fascia_region': False,
                    'confidence': min(0.99, vlm_confidence * 0.9),  # Slightly discount for uncertainty
                    'color': self.CLASSIFICATION_RULES.get(vlm_type, {}).get('color', (255, 255, 255)),
                    'vlm_description': description
                }
                
                classifications.append(classification)
            except Exception as e:
                logger.warning(f"Failed to process VLM vein {i}: {e}")
        
        return classifications
    
    def _get_position_label(self, vein_type: str, distance_to_fascia: Optional[float]) -> str:
        """Get position label based on vein type and distance to fascia."""
        if vein_type == 'N1_deep':
            return 'below_fascia'
        elif vein_type == 'N2_gsv':
            return 'within_fascia'
        elif vein_type == 'N3_superficial':
            return 'above_fascia'
        return 'unknown'

    def _convert_llava_descriptions_to_detections(
        self,
        vessel_descriptions: List[Dict],
        image: np.ndarray,
        fascia_analysis: str
    ) -> List[Dict]:
        """
        Convert LLaVA's descriptive vessel locations to actual detections with coordinates.
        Uses image processing to find dark tubular structures matching the descriptions.
        """
        detections = []
        height, width = image.shape[:2]
        
        if not vessel_descriptions:
            # Try image processing as fallback
            return self._detect_veins_with_image_processing(image)
        
        # Process each vessel description
        for idx, vessel in enumerate(vessel_descriptions):
            try:
                region = vessel.get('region', '').lower()
                size = vessel.get('size', 'medium').lower()
                vessel_type = vessel.get('type', 'vein').lower()
                
                # Map region to approximate coordinates
                # "bottom right" -> (0.7*width, 0.7*height), etc.
                region_center = self._region_to_coordinates(region, width, height)
                
                # Map size to approximate diameter
                size_diameter = self._size_to_diameter(size, width, height)
                
                # Classify based on vertical position
                vein_type = self._classify_by_position(region, vessel_type)
                
                detection = {
                    'center_x': int(region_center[0]),
                    'center_y': int(region_center[1]),
                    'diameter': int(size_diameter),
                    'type': vein_type,
                    'confidence': 0.7,  # Reduced confidence due to descriptive nature
                    'description': vessel.get('description', ''),
                    'regional': True  # Mark as region-based
                }
                
                detections.append(detection)
                logger.info(f"Detected vessel {idx+1}: {vein_type} at ({region_center[0]:.0f}, {region_center[1]:.0f})")
                
            except Exception as e:
                logger.warning(f"Failed to process vessel description {idx}: {e}")
        
        # If we got some detections, fine-tune them with image processing
        if detections:
            detections = self._refine_detections_with_image_processing(image, detections)
        else:
            # Fall back to pure image processing
            detections = self._detect_veins_with_image_processing(image)
        
        return detections
    
    def _region_to_coordinates(self, region: str, width: int, height: int) -> Tuple[float, float]:
        """Convert region description to approximate coordinates."""
        # Default to center
        x, y = width / 2, height / 2
        
        # Vertical positioning
        if 'top' in region or 'upper' in region:
            y = height * 0.25
        elif 'bottom' in region or 'lower' in region:
            y = height * 0.75
        elif 'middle' in region or 'center' in region:
            y = height * 0.5
        
        # Horizontal positioning
        if 'left' in region:
            x = width * 0.25
        elif 'right' in region:
            x = width * 0.75
        elif 'center' in region or 'middle' in region:
            x = width * 0.5
        
        return (x, y)
    
    def _size_to_diameter(self, size: str, width: int, height: int) -> float:
        """Convert size description to approximate diameter."""
        imag_diagonal = (width ** 2 + height ** 2) ** 0.5
        
        if 'small' in size.lower():
            return imag_diagonal * 0.03
        elif 'medium' in size.lower():
            return imag_diagonal * 0.06
        elif 'large' in size.lower():
            return imag_diagonal * 0.1
        else:
            return imag_diagonal * 0.05  # Default
    
    def _classify_by_position(self, region: str, vessel_type: str) -> str:
        """Classify vein type based on position and characteristics."""
        region_lower = region.lower()
        vessel_lower = vessel_type.lower()
        
        # Explicit GSV/type classification from LLaVA
        if 'gsv' in vessel_lower:
            return 'N2_gsv'  # GSV is always N2 (within/at fascia)
        elif 'deep' in vessel_lower:
            return 'N1_deep'
        elif 'superficial' in vessel_lower and 'deep' not in vessel_lower:
            return 'N3_superficial'
        
        # Position-based secondary classification
        if 'bottom' in region_lower or 'lower' in region_lower:
            return 'N1_deep'  # Bottom/lower = deeper
        elif 'top' in region_lower or 'upper' in region_lower:
            return 'N3_superficial'  # Top/upper = superficial
        else:
            # Middle or center = default to GSV (N2)
            return 'N2_gsv'
    
    def _refine_detections_with_image_processing(
        self,
        image: np.ndarray,
        initial_detections: List[Dict]
    ) -> List[Dict]:
        """Use image processing to refine and verify LLaVA detections."""
        try:
            # Find dark regions in image (veins appear dark in ultrasound)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find dark structures
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            refined = []
            for detection in initial_detections:
                cx, cy = detection['center_x'], detection['center_y']
                search_diameter = detection['diameter'] * 1.5
                
                # Find contour closest to predicted location
                best_contour = None
                best_dist = float('inf')
                
                for contour in contours:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        ccx = M['m10'] / M['m00']
                        ccy = M['m01'] / M['m00']
                        dist = (ccx - cx) ** 2 + (ccy - cy) ** 2
                        
                        if dist < search_diameter ** 2 and dist < best_dist:
                            best_contour = contour
                            best_dist = dist
                
                if best_contour is not None:
                    # Use actual contour location
                    M = cv2.moments(best_contour)
                    if M['m00'] > 0:
                        detection['center_x'] = int(M['m10'] / M['m00'])
                        detection['center_y'] = int(M['m01'] / M['m00'])
                        detection['confidence'] = min(0.95, detection['confidence'] + 0.1)
                        
                        # Update diameter based on contour
                        (x, y), radius = cv2.minEnclosingCircle(best_contour)
                        detection['diameter'] = int(radius * 2)
                
                refined.append(detection)
            
            return refined
        except Exception as e:
            logger.warning(f"Could not refine detections with image processing: {e}")
            return initial_detections
    
    def _detect_veins_with_image_processing(self, image: np.ndarray) -> List[Dict]:
        """Fallback: Use image processing to detect vein-like structures."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast to find dark structures
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Threshold to find dark structures (veins appear very dark)
            _, binary = cv2.threshold(enhanced, 140, 255, cv2.THRESH_BINARY_INV)
            
            # Use morphological operations to enhance tubular structures
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            min_area = (image.shape[0] * image.shape[1]) * 0.002  # At least 0.2% of image (HIGHER threshold)
            
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                if area < min_area:
                    continue
                
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Fit ellipse to detect tubular structures better
                    if len(contour) > 5:
                        try:
                            ellipse = cv2.fitEllipse(contour)
                            (ex, ey), (ewidth, eheight), angle = ellipse
                            aspect_ratio = max(ewidth, eheight) / (min(ewidth, eheight) + 1e-5)
                        except:
                            aspect_ratio = 1.0
                    else:
                        aspect_ratio = 1.0
                    
                    # ONLY accept elongated structures (aspect ratio > 1.8 for tubular veins)
                    if aspect_ratio < 1.8:
                        logger.debug(f"Skipping blob: aspect_ratio={aspect_ratio:.2f} (too circular)")
                        continue
                    
                    # Classify based on position (rough thirds)
                    image_height = image.shape[0]
                    if cy < image_height * 0.35:
                        vtype = 'N3_superficial'  # Top third
                    elif cy < image_height * 0.65:
                        vtype = 'N2_gsv'  # Middle third (at/near fascia)
                    else:
                        vtype = 'N1_deep'  # Bottom third
                    
                    # High confidence only for large, clearly tubular structures
                    base_confidence = min(0.85, 0.4 + (area / (image.shape[0] * image.shape[1])) * 15)
                    if aspect_ratio > 2.5:
                        base_confidence = min(0.95, base_confidence + 0.15)  # Boost for highly tubular
                    
                    detections.append({
                        'center_x': int(cx),
                        'center_y': int(cy),
                        'diameter': int(radius * 2),
                        'type': vtype,
                        'confidence': base_confidence,
                        'description': f'Tubular vessel (AR={aspect_ratio:.2f})',
                        'from_cv': True
                    })
            
            logger.info(f"Image processing detected {len(detections)} vein-like structures (filtered for tubular shapes)")
            return detections
        except Exception as e:
            logger.error(f"Image processing detection failed: {e}")
            return []

    def _classification_to_dict(self, classification: Dict) -> Dict:
        """Convert classification to JSON-serializable dict"""
        return {
            'blob_id': classification['blob_id'],
            'center': list(classification['center']),
            'radius': classification['radius'],
            'vein_type': classification['vein_type'],
            'vein_label': classification['vein_label'],
            'position': classification['position'],
            'distance_to_fascia': classification.get('distance_to_fascia'),
            'confidence': round(classification['confidence'], 2),
            'in_fascia_region': classification.get('in_fascia_region', False),
            'vlm_description': classification.get('vlm_description', '')
        }
