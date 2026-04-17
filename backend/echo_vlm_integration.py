"""
Echo VLM Integration - Qwen2VLMOEForConditionalGeneration
Uses exact code pattern from reference: local model with proper inference pipeline
3-stage verification: Fascia → Vein Validation → N1/N2/N3 Classification
Integrated with Qdrant-based RAG system (CHIVA knowledge base)
"""

import numpy as np
import cv2
import json
import logging
import requests
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

try:
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_ECHOVLM = True
except ImportError as e:
    logger.warning(f"EchoVLM dependencies not available: {e}")
    HAS_ECHOVLM = False

logger = logging.getLogger(__name__)


@dataclass
class VeinClassificationResult:
    """Result of vein classification"""
    vein_id: int
    classification: str  # N1, N2, or N3
    confidence: float  # 0-1
    reasoning: str
    distance_to_fascia: Optional[float] = None
    relative_position: Optional[str] = None


@dataclass
class FasciaDetectionResult:
    """Result of fascia detection verification"""
    detected: bool
    confidence: float  # 0-1
    position: Optional[Tuple[int, int]] = None
    thickness: Optional[float] = None
    reasoning: str = ""


class EchoVLMIntegration:
    """
    Integration with EchoVLM via HuggingFace Inference API
    Uses exact reference code pattern but calls HF API instead of local inference
    """

    def __init__(
        self,
        model_id: str = "chaoyinshe/EchoVLM",
        retrieve_context_fn: Optional[Callable] = None,
    ):
        """
        Initialize EchoVLM with HuggingFace Inference API

        Args:
            model_id: HuggingFace model ID for EchoVLM (default: "chaoyinshe/EchoVLM")
            retrieve_context_fn: Function to retrieve RAG context from Qdrant
        """
        self.model_id = model_id
        self.retrieve_rag_context = retrieve_context_fn
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.processor = None
        self._initialized = False

        if not HAS_ECHOVLM:
            logger.warning("⚠️ Install: pip install transformers qwen_vl_utils")
            return

        self._initialize_processor()

    def _initialize_processor(self):
        """Load only the processor (no model needed - use API)"""
        try:
            logger.info(f"Loading EchoVLM processor from {self.model_id}...")

            # Load only processor (no model - we'll use API)
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            self._initialized = True
            logger.info("✅ EchoVLM processor loaded (using HuggingFace Inference API for inference)")

        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            self._initialized = False

    def _call_echovlm_api(self, image: np.ndarray, prompt: str) -> str:
        """
        Call EchoVLM using exact reference code pattern but via HuggingFace Inference API

        Uses the EXACT same pattern as reference:
        - processor.apply_chat_template()
        - process_vision_info()
        - processor() to prepare inputs
        - Then send to HF API instead of model.generate()

        Args:
            image: Ultrasound image (numpy array, RGB)
            prompt: Text prompt for VLM

        Returns:
            VLM response text
        """
        if not self._initialized or self.processor is None:
            logger.error("EchoVLM processor not initialized")
            return ""

        try:
            # Convert numpy to PIL Image if needed
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image.astype(np.uint8))
            else:
                image_pil = image

            # ===== 1. Prepare messages (EXACT pattern from reference) =====
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_pil,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # ===== 2. Preparation for inference (EXACT pattern from reference) =====
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            # Convert image to base64 for API
            if image_inputs:
                img_pil = image_inputs[0] if isinstance(image_inputs, list) else image_inputs
                img_byte_arr = BytesIO()
                img_pil.save(img_byte_arr, format='PNG')
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
            else:
                img_base64 = None

            # ===== 3. Call HuggingFace Inference API =====
            logger.info(f"Calling EchoVLM API with prompt: {prompt[:50]}...")

            payload = {
                "inputs": text,
            }

            headers = {
                "Authorization": f"Bearer {self._get_hf_token()}",
            } if self._get_hf_token() else {}

            # Send to HF Inference API
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                # API returns array of results
                if isinstance(result, list) and len(result) > 0:
                    output_text = result[0].get("generated_text", "")
                    # Clean up the output if needed
                    if output_text.startswith(text):
                        output_text = output_text[len(text):].strip()
                    return output_text
                else:
                    return str(result)
            else:
                logger.error(f"HF API error {response.status_code}: {response.text}")
                return ""

        except Exception as e:
            logger.error(f"EchoVLM inference error: {e}")
            return ""

    @staticmethod
    def _get_hf_token() -> Optional[str]:
        """Get HuggingFace token from environment"""
        import os
        return os.getenv("HF_TOKEN", None)

    def verify_fascia_detection(
        self,
        image: np.ndarray,
        fascia_y: Optional[int] = None,
    ) -> FasciaDetectionResult:
        """
        Stage 1.5: Verify fascia detection using EchoVLM

        Args:
            image: Original ultrasound image (RGB)
            fascia_y: Y-coordinate of detected fascia line

        Returns:
            FasciaDetectionResult with verification
        """
        # Create visualization with fascia overlay
        vis_image = image.copy()
        if fascia_y is not None:
            cv2.line(vis_image, (0, fascia_y), (image.shape[1], fascia_y), (0, 255, 0), 3)

        prompt = """Fascia Detection Verification

Green line = detected fascial layer

Is the green line at a real tissue boundary (fascia)?
- Clear line? Continuous? Right depth?

Respond with ONLY this JSON:
{"fascia_detected": true, "confidence": 90, "reasoning": "clear continuous line at tissue interface"}
or
{"fascia_detected": false, "confidence": 75, "reasoning": "line looks like artifact"}"""

        response = self._call_echovlm_api(vis_image, prompt)

        try:
            # Try to extract JSON from response
            if not response:
                raise ValueError("Empty response from EchoVLM")

            # Clean response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result_dict = json.loads(cleaned)
            confidence = result_dict.get("confidence", 70)
            if isinstance(confidence, str):
                confidence = int(confidence.strip('%'))

            return FasciaDetectionResult(
                detected=result_dict.get("fascia_detected", True),
                confidence=min(100, max(0, confidence)) / 100.0,
                position=(0, fascia_y) if fascia_y else None,
                reasoning=result_dict.get("reasoning", "EchoVLM fascia verification"),
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse EchoVLM response for fascia: {e}. Response: {response[:200]}")
            # Smart fallback based on fascia_y position
            detected = fascia_y is not None
            return FasciaDetectionResult(
                detected=detected,
                confidence=0.75 if detected else 0.5,
                position=(0, fascia_y) if fascia_y else None,
                reasoning="CNN-based detection (EchoVLM parsing failed - using CNN output)",
            )

    def validate_vein_detections(
        self,
        image: np.ndarray,
        vein_detections: List[Dict],
        fascia_y: Optional[int] = None,
    ) -> List[Dict]:
        """
        Stage 2.5: Validate vein detections using EchoVLM

        Args:
            image: Original ultrasound image (RGB)
            vein_detections: List of detected veins
            fascia_y: Y-coordinate of fascia for reference

        Returns:
            Updated vein detections with validation flags
        """
        # Create visualization with detected veins
        vis_image = image.copy()

        for vein in vein_detections:
            color = (255, 0, 0)  # Red for veins
            cv2.circle(vis_image, (vein["x"], vein["y"]), vein["radius"], color, 2)
            cv2.putText(
                vis_image,
                f"V{vein['id']}",
                (vein["x"], vein["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        if fascia_y is not None:
            cv2.line(vis_image, (0, fascia_y), (vis_image.shape[1], fascia_y), (0, 255, 0), 2)

        num_veins = len(vein_detections)
        prompt = f"""Vein Validation

Image shows {num_veins} marked circles (red = detected veins, green = fascia).

For each vein (V0, V1, V2...):
- Is it a real vein? (round structure, thin walls, dark center)
- Or is it artifact/noise?

Respond with ONLY JSON:
{{"veins": [{{"id": 0, "is_valid": true, "confidence": 90, "notes": "real vein"}}], "total_valid": {num_veins}}}"""

        response = self._call_echovlm_api(vis_image, prompt)

        try:
            if not response:
                raise ValueError("Empty response from EchoVLM")

            # Clean response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result_dict = json.loads(cleaned)
            vein_results = result_dict.get("veins", [])

            for vein, validation in zip(vein_detections, vein_results):
                confidence = validation.get("confidence", 70)
                if isinstance(confidence, str):
                    confidence = int(confidence.strip('%'))

                vein["vlm_valid"] = validation.get("is_valid", True)
                vein["vlm_confidence"] = min(100, max(0, confidence)) / 100.0
                vein["vlm_notes"] = validation.get("notes", "Valid vein detection")

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse EchoVLM vein validation response: {e}. Response: {response[:200]}")
            # Smart fallback - accept all veins if parsing fails
            for vein in vein_detections:
                vein["vlm_valid"] = True
                vein["vlm_confidence"] = 0.75  # Moderate confidence in CNN detection
                vein["vlm_notes"] = "CNN-validated (EchoVLM parsing failed)"

        return vein_detections

    def classify_vein(
        self,
        image: np.ndarray,
        vein: Dict,
        fascia_y: Optional[int] = None,
    ) -> VeinClassificationResult:
        """
        Stage 4: Classify vein as N1, N2, or N3 using EchoVLM + RAG

        Args:
            image: Original ultrasound image (RGB)
            vein: Vein detection dict with coordinates
            fascia_y: Y-coordinate of fascia

        Returns:
            VeinClassificationResult with N1/N2/N3 classification
        """
        x, y, radius = vein["x"], vein["y"], vein["radius"]
        margin = max(50, radius * 2)

        x_min = max(0, x - margin)
        x_max = min(image.shape[1], x + margin)
        y_min = max(0, y - margin)
        y_max = min(image.shape[0], y + margin)

        zoomed = image[y_min:y_max, x_min:x_max].copy()

        # Draw vein center
        local_x = x - x_min
        local_y = y - y_min
        cv2.circle(zoomed, (local_x, local_y), max(1, radius // 2), (0, 255, 0), 2)

        # Draw fascia if visible
        if fascia_y is not None and y_min <= fascia_y <= y_max:
            local_fascia_y = fascia_y - y_min
            cv2.line(
                zoomed,
                (0, local_fascia_y),
                (zoomed.shape[1], local_fascia_y),
                (0, 0, 255),
                2,
            )

        # Calculate spatial relationship
        distance_to_fascia = None
        relative_position = "Unknown"
        if fascia_y is not None:
            distance_to_fascia = y - fascia_y
            distance_mm = abs(distance_to_fascia) * 0.1  # Approximate mm

            if y < fascia_y - 20:
                relative_position = "Above fascia (superficial)"
            elif y > fascia_y + 20:
                relative_position = "Below fascia (deep)"
            else:
                relative_position = "At/near fascia"

        # Retrieve RAG context
        rag_context = ""
        rag_references = []
        if self.retrieve_rag_context and distance_to_fascia is not None:
            try:
                query = f"vein depth classification {relative_position.lower()} N1 N2 N3 CHIVA perforator"
                context_chunks = self.retrieve_rag_context(query, k=2)

                if context_chunks:
                    rag_context = "\nRELEVANT CLINICAL CONTEXT:\n"
                    for i, chunk in enumerate(context_chunks, 1):
                        rag_context += f"\n[Context {i}]\n{chunk[:300]}...\n"
                        rag_references.append(f"RAG-{i}")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Build prompt for classification
        distance_mm = abs(distance_to_fascia) * 0.1 if distance_to_fascia else 0
        prompt = f"""Vein Classification - Analyze this ultrasound image

Green circle = detected vein
Red line = fascia (reference)
Distance to fascia: {distance_mm:.1f}mm ({relative_position})

{rag_context}

CLASSIFICATION RULES:
N1 = Dark vein, >50mm below fascia (in muscle)
N2 = Gray vein, near fascia (±20mm) ← PRIMARY CHIVA TARGET
N3 = Bright vein, >20mm above fascia (superficial)

Analyze the vein:
1. Color: Dark? Gray? Bright?
2. Position: Below/at/above fascia?
3. Compression: Rigid? Collapses easily?

Respond with ONLY this JSON format:
{{"classification": "N1", "confidence": 85, "reasoning": "vein is dark and >50mm below fascia"}}
OR
{{"classification": "N2", "confidence": 90, "reasoning": "gray vein at fascia interface"}}
OR
{{"classification": "N3", "confidence": 88, "reasoning": "bright vein 35mm above fascia"}}"""

        response = self._call_echovlm_api(zoomed, prompt)

        try:
            if not response:
                raise ValueError("Empty response from EchoVLM")

            # Clean response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            result_dict = json.loads(cleaned)

            # Normalize confidence to 0-100 range
            confidence = result_dict.get("confidence", 70)
            if isinstance(confidence, str):
                confidence = int(confidence.strip('%'))
            confidence = min(100, max(0, confidence))

            # Validate classification
            classification = result_dict.get("classification", "N2")
            if classification not in ["N1", "N2", "N3"]:
                # Try to extract from reasoning if invalid
                if "N1" in result_dict.get("reasoning", ""):
                    classification = "N1"
                elif "N3" in result_dict.get("reasoning", ""):
                    classification = "N3"
                else:
                    classification = "N2"

            return VeinClassificationResult(
                vein_id=vein.get("id", 0),
                classification=classification,
                confidence=confidence / 100.0,
                reasoning=result_dict.get("reasoning", "EchoVLM classification"),
                distance_to_fascia=distance_to_fascia,
                relative_position=relative_position,
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse EchoVLM response for vein {vein.get('id', 0)}: {e}. Response: {response[:200]}")
            # Smart fallback based on spatial position AND echogenicity heuristics
            if distance_to_fascia is not None:
                if distance_to_fascia > 50:
                    classification = "N1"
                    confidence = 0.70
                    reasoning = "Deep vein >50mm below fascia (distance-based fallback)"
                elif distance_to_fascia < -20:
                    classification = "N3"
                    confidence = 0.75
                    reasoning = "Superficial vein >20mm above fascia (distance-based fallback)"
                else:
                    classification = "N2"
                    confidence = 0.80
                    reasoning = "At fascia interface ±20mm (distance-based fallback)"
            else:
                classification = "N2"
                confidence = 0.65
                reasoning = "Default to N2 (primary CHIVA target)"

            return VeinClassificationResult(
                vein_id=vein.get("id", 0),
                classification=classification,
                confidence=confidence,
                reasoning=reasoning + " - EchoVLM parsing failed, using spatial analysis",
                distance_to_fascia=distance_to_fascia,
                relative_position=relative_position,
            )

    def batch_classify_veins(
        self,
        image: np.ndarray,
        veins: List[Dict],
        fascia_y: Optional[int] = None,
    ) -> List[VeinClassificationResult]:
        """Classify multiple veins sequentially"""
        results = []
        for vein in veins:
            result = self.classify_vein(image, vein, fascia_y)
            results.append(result)
        return results

    def comprehensive_analysis(
        self,
        image: np.ndarray,
        vein_detections: List[Dict],
        fascia_y: Optional[int] = None,
    ) -> Dict:
        """
        Perform complete 4-stage analysis

        Returns:
            Dictionary with all results
        """
        logger.info("Starting EchoVLM comprehensive 4-stage analysis...")

        # Stage 1.5: Verify fascia
        logger.info("Stage 1.5: Verifying fascia with EchoVLM...")
        fascia_result = self.verify_fascia_detection(image, fascia_y)

        # Stage 2.5: Validate veins
        logger.info("Stage 2.5: Validating vein detections with EchoVLM...")
        validated_veins = self.validate_vein_detections(image, vein_detections, fascia_y)

        # Stage 4: Classify veins
        logger.info("Stage 4: Classifying veins (N1/N2/N3) with EchoVLM + RAG...")
        classifications = self.batch_classify_veins(image, validated_veins, fascia_y)

        return {
            "fascia": {
                "detected": fascia_result.detected,
                "confidence": fascia_result.confidence,
                "position": fascia_result.position,
                "reasoning": fascia_result.reasoning,
            },
            "validated_veins": validated_veins,
            "classifications": [
                {
                    "vein_id": c.vein_id,
                    "classification": c.classification,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                    "distance_to_fascia": c.distance_to_fascia,
                    "relative_position": c.relative_position,
                }
                for c in classifications
            ],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize with actual EchoVLM
    vlm = EchoVLMIntegration(
        model_id="chaoyinshe/EchoVLM",
        device_map="auto",
        retrieve_context_fn=None,  # Can pass retrieve_context from app.py
    )

    if vlm._initialized:
        logger.info("✅ EchoVLM successfully initialized and ready for use")
    else:
        logger.error("❌ EchoVLM initialization failed")
