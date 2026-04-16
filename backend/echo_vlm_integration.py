"""
Echo VLM Integration - Qwen2VLMOEForConditionalGeneration
Actual ultrasound-specialized VLM for clinical decision support
3-stage verification: Fascia → Vein Validation → N1/N2/N3 Classification
Integrated with Qdrant-based RAG system (CHIVA knowledge base)
"""

import torch
import numpy as np
import cv2
import json
import logging
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

try:
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    AutoProcessor = None
    process_vision_info = None

try:
    from EchoVLM import Qwen2VLMOEForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2VLMOEForConditionalGeneration
    except ImportError:
        Qwen2VLMOEForConditionalGeneration = None

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
    """Integration with Qwen2VLM (EchoVLM) for ultrasound interpretation"""

    def __init__(
        self,
        model_id: str = "chaoyinshe/EchoVLM",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        use_flash_attention_2: bool = True,
        retrieve_context_fn: Optional[Callable] = None,
    ):
        """
        Initialize Echo VLM (Qwen2VLMOEForConditionalGeneration)

        Args:
            model_id: HuggingFace model ID for EchoVLM
            device_map: Device mapping ("auto", "cuda", "cpu")
            torch_dtype: Data type (bfloat16, float16, float32)
            use_flash_attention_2: Use Flash Attention 2 for efficiency
            retrieve_context_fn: Function to retrieve RAG context from Qdrant
        """
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_flash_attention_2 = use_flash_attention_2
        self.retrieve_rag_context = retrieve_context_fn

        # Initialize model and processor
        self.model = None
        self.processor = None
        self._initialized = False

        if not HAS_QWEN:
            logger.warning("Qwen VL utilities not available - VLM will not work")
            return

        if Qwen2VLMOEForConditionalGeneration is None:
            logger.warning("Qwen2VLMOEForConditionalGeneration not found - install EchoVLM")
            return

        self._initialize_model()

    def _initialize_model(self):
        """Load model and processor"""
        try:
            logger.info(f"Loading EchoVLM from {self.model_id}...")

            # Map dtype string to torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

            # Load model
            attn_impl = "flash_attention_2" if self.use_flash_attention_2 else None

            self.model = Qwen2VLMOEForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map=self.device_map,
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            self._initialized = True
            logger.info("✅ EchoVLM model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize EchoVLM: {e}")
            self._initialized = False

    def _call_echovlm(self, image: np.ndarray, prompt: str) -> str:
        """
        Call actual EchoVLM (Qwen2VLM) with image and prompt

        Args:
            image: Ultrasound image (numpy array, RGB)
            prompt: Text prompt for VLM

        Returns:
            VLM response text
        """
        if not self._initialized or self.model is None or self.processor is None:
            logger.error("EchoVLM not initialized")
            return ""

        try:
            # Convert numpy to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))

            # Prepare message for EchoVLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                )

            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0] if output_text else ""

        except Exception as e:
            logger.error(f"EchoVLM inference error: {e}")
            return ""

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

        prompt = """Analyze this ultrasound image with the green line marking the detected fascial layer.

TASK: Verify fascial layer detection
1. Is the fascial layer correctly identified? (yes/no)
2. Confidence level (0-100%)
3. Is the anatomical position reasonable? (yes/no)

RESPONSE FORMAT (JSON ONLY):
{
    "fascia_detected": true/false,
    "confidence": 0-100,
    "reasoning": "brief clinical explanation"
}"""

        response = self._call_echovlm(vis_image, prompt)

        try:
            result_dict = json.loads(response)
            return FasciaDetectionResult(
                detected=result_dict.get("fascia_detected", True),
                confidence=result_dict.get("confidence", 70) / 100.0,
                position=(0, fascia_y) if fascia_y else None,
                reasoning=result_dict.get("reasoning", "EchoVLM fascia verification"),
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse EchoVLM response for fascia verification")
            return FasciaDetectionResult(
                detected=True,
                confidence=0.7,
                position=(0, fascia_y) if fascia_y else None,
                reasoning="Automatic detection (EchoVLM parsing failed)",
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
        prompt = f"""Analyze this ultrasound image with marked veins (red circles) and fascial layer (green line).

TASK: Validate vein detections
For each marked vein:
1. Is it a true vein? (yes/no)
2. Confidence (0-100%)
3. Any anatomical concerns?

Total marked veins: {num_veins}

RESPONSE FORMAT (JSON ONLY):
{{
    "veins": [
        {{"id": 1, "is_valid": true, "confidence": 95, "notes": "clear vein structure"}},
        ...
    ],
    "overall_quality": "high"
}}"""

        response = self._call_echovlm(vis_image, prompt)

        try:
            result_dict = json.loads(response)
            vein_results = result_dict.get("veins", [])

            for vein, validation in zip(vein_detections, vein_results):
                vein["vlm_valid"] = validation.get("is_valid", True)
                vein["vlm_confidence"] = validation.get("confidence", 70) / 100.0
                vein["vlm_notes"] = validation.get("notes", "Valid vein detection")

        except json.JSONDecodeError:
            logger.warning("Failed to parse EchoVLM response for vein validation")
            for vein in vein_detections:
                vein["vlm_valid"] = True
                vein["vlm_confidence"] = 0.7
                vein["vlm_notes"] = "EchoVLM parsing failed"

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
        prompt = f"""Analyze this zoomed ultrasound image showing a detected vein (green circle).
Red line (if visible) indicates the fascial layer.

{rag_context}

MEASUREMENT DATA:
- Distance to fascia: {distance_mm:.1f}mm
- Spatial position: {relative_position}
- Vein ID: {vein.get('id', 0)}

CLASSIFICATION RULES:
N1 (Deep): Vein center > 50mm BELOW fascia
  → Low echogenicity (dark), less compressible, in muscle

N2 (At Fascia): Vein within ±20mm of fascia ⭐ IDEAL FOR CHIVA
  → Moderate echogenicity, highly compressible, at interface

N3 (Superficial): Vein center > 20mm ABOVE fascia
  → High echogenicity (bright), fully compressible, in skin layer

TASK: Classify this vein as N1, N2, or N3 with clinical reasoning.

RESPONSE FORMAT (JSON ONLY):
{{
    "classification": "N1" or "N2" or "N3",
    "confidence": 0-100,
    "distance_to_fascia_mm": {distance_mm:.1f},
    "echogenicity": "low" or "moderate" or "high",
    "reasoning": "detailed clinical explanation based on ultrasound features",
    "clinical_significance": "relevance for CHIVA treatment planning"
}}"""

        response = self._call_echovlm(zoomed, prompt)

        try:
            result_dict = json.loads(response)
            return VeinClassificationResult(
                vein_id=vein.get("id", 0),
                classification=result_dict.get("classification", "N2"),
                confidence=result_dict.get("confidence", 70) / 100.0,
                reasoning=result_dict.get("reasoning", "EchoVLM classification"),
                distance_to_fascia=distance_to_fascia,
                relative_position=relative_position,
            )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse EchoVLM response for vein {vein.get('id', 0)}")
            # Fallback based on spatial position
            if distance_to_fascia is not None:
                if distance_to_fascia > 50:
                    classification = "N1"
                elif -20 <= distance_to_fascia <= 20:
                    classification = "N2"
                else:
                    classification = "N3"
            else:
                classification = "N2"

            return VeinClassificationResult(
                vein_id=vein.get("id", 0),
                classification=classification,
                confidence=0.6,
                reasoning="Spatial classification (EchoVLM parsing failed)",
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
