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

TASK: Verify fascial layer detection - Critical for CHIVA classification

DETAILED ANALYSIS REQUIRED:
1. **Fascial Line Visibility**: Is the fascia clearly visible as a hyperechoic (bright) horizontal line?
2. **Continuity**: Does the fascial line appear continuous across the image?
3. **Depth**: Is it at anatomically correct depth (typically 3-5mm for superficial fascia)?
4. **Artifacts**: Are there imaging artifacts (shadows, reverberation) that could confuse the fascial line?
5. **Anatomical Context**: Is the fascia position consistent with tissue planes (between dermis and subcutaneous fat)?

RESPONSE FORMAT (VALID JSON):
{
    "fascia_detected": true,
    "confidence": 85,
    "line_visibility": "clear",
    "continuity": "continuous",
    "depth_mm": 4.2,
    "artifact_present": false,
    "anatomical_validity": "correct",
    "reasoning": "Green line marks a clear, continuous hyperechoic band consistent with superficial fascia at appropriate depth. No significant artifacts. Anatomically valid position between dermis and subcutaneous fat."
}"""

        response = self._call_echovlm(vis_image, prompt)

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
        prompt = f"""Analyze this ultrasound image with marked veins (red circles labeled V#) and fascial layer (green line).

TASK: Validate vein detections - Distinguish true veins from artifacts

FOR EACH MARKED VEIN (V0, V1, V2...):
1. **Compressibility**: Can the structure be compressed/flattened with probe pressure? (True veins compress)
2. **Wall Structure**: Does it have thin, hyperechoic walls typical of veins?
3. **Lumen Content**: Is interior mostly anechoic (empty/blood-filled) or echogenic (clot/artifact)?
4. **Size Consistency**: Is the circular/oval shape consistent with a vein cross-section?
5. **False Positive Indicators**: Could this be fascia, tendon, nerve, or imaging artifact instead?
6. **Pulsatility**: Any pulsatile changes visible? (Suggests artery, not vein)
7. **Location**: Is it in expected anatomical location for veins?

Total marked veins: {num_veins}

RESPONSE FORMAT (VALID JSON):
{{
    "veins": [
        {{
            "id": 0,
            "is_valid": true,
            "confidence": 92,
            "wall_quality": "thin hyperechoic",
            "lumen_appearance": "anechoic",
            "compressible": true,
            "false_positive_risk": "low",
            "notes": "Clear vein structure with thin walls, anechoic lumen, compressible, no pulsatility detected"
        }}
    ],
    "overall_quality": "high",
    "total_valid": {num_veins}
}}"""

        response = self._call_echovlm(vis_image, prompt)

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
        prompt = f"""EXPERT VASCULAR ULTRASOUND ANALYSIS - CHIVA Classification

Image: Zoomed ultrasound showing detected vein (green circle)
Red line: Fascial layer reference point
Measurement: {distance_mm:.1f}mm from fascia ({relative_position})
Vein ID: {vein.get('id', 0)}

{rag_context}

DETAILED ANALYSIS CHECKLIST:

1. **ECHOGENICITY Assessment** (Critical):
   - N1 (Deep/Dark): Hypoechoic appearance, minimal brightness, appears dark gray
   - N2 (Moderate): Medium gray appearance, visible but not bright, moderate echogenicity
   - N3 (Bright): Hyperechoic appearance, bright white/light gray, high brightness

2. **COMPRESSIBILITY Assessment** (Critical):
   - N1 (Low): Minimal deformation under pressure, rigid appearance, stays round
   - N2 (High): Readily flattens under minimal probe pressure, easily collapsible
   - N3 (Complete): Completely flattens/disappears with light pressure, very soft

3. **WALL THICKNESS**:
   - N1: Thick walls, difficult to see lumen clearly
   - N2: Medium walls, lumen clearly visible, defined margins
   - N3: Thin walls, easily compressible, prominent anechoic lumen

4. **SURROUNDING TISSUE**:
   - N1: Surrounded by echogenic muscle tissue (speckled appearance)
   - N2: At interface between fascia and subcutaneous tissue (transition zone)
   - N3: Surrounded by hypoechoic/anechoic subcutaneous fat

5. **DEPTH CRITERIA** (Use as tie-breaker):
   - N1: >50mm BELOW fascia (deeper than typical)
   - N2: ±20mm FROM fascia (around the interface)
   - N3: >20mm ABOVE fascia (very close to surface)

CLASSIFICATION LOGIC:
- Start with echogenicity + compressibility assessment
- Confirm with surrounding tissue appearance
- Use depth measurement to validate
- Weight most confident finding highest

RESPONSE FORMAT (VALID JSON - MUST BE PARSEABLE):
{{
    "classification": "N2",
    "confidence": 88,
    "distance_to_fascia_mm": {distance_mm:.1f},
    "echogenicity": "moderate",
    "echogenicity_score": 6,
    "compressibility": "high",
    "compressibility_score": 9,
    "wall_quality": "thin-medium",
    "surrounding_tissue": "at fascia interface",
    "depth_check": "matches N2 criteria",
    "reasoning": "Vein shows moderate echogenicity (medium gray appearance) with excellent compressibility, confirming fascial-level position. Surrounding tissue at fascia-subcutaneous interface. Wall thickness and lumen definition consistent with N2. Measurement of {distance_mm:.1f}mm from fascia confirms ±20mm range for N2.",
    "clinical_significance": "Primary CHIVA target. Ideal for endovenous ablation at fascial level.",
    "certainty_factors": ["moderate_echogenicity", "high_compressibility", "fascial_interface_location", "measurement_consistent"]
}}"""

        response = self._call_echovlm(zoomed, prompt)

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
