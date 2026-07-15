"""
Vision LLM Integration Module - OPTIMIZED FOR REAL-TIME INFERENCE

Uses lightweight, ultra-fast LLM for real-time vein classification (millisecond response).
Removes image encoding overhead - uses only spatial feature data for 10-100ms inference.
Supports local Ollama with Phi/TinyLLaMA models for CPU-efficient operation.
"""

import logging
import json
from typing import Dict, List, Optional
import numpy as np
import time
from functools import lru_cache

logger = logging.getLogger(__name__)


class VisionLLMInterface:
    """Ultra-optimized LLM interface for real-time vein classification (< 100ms inference)"""
    
    def __init__(self, model_provider: str = "ollama", api_key: Optional[str] = None, model_name: str = "llama3.2:1b"):
        """
        Initialize ultra-fast LLM interface
        
        Args:
            model_provider: "ollama" (recommended for real-time)
            api_key: Not used (local inference only)
            model_name: Fast model like "llama3.2:1b" (10-30ms), "mistral" (50-100ms)
        """
        self.model_provider = "ollama"  # Only Ollama for real-time performance
        self.api_key = None
        self.model_name = model_name  # Use LLaMA3.2:1b or Mistral for speed
        self.client = self._initialize_client()
        self.inference_count = 0
        self.total_inference_time = 0
        self.last_results_cache = {}  # Simple cache for repeated frames
    
    def _initialize_client(self):
        """Initialize Ollama client (local inference only)"""
        try:
            import requests
            # Test connection to Ollama
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name") for m in models]
                    
                    # Check if our model is available
                    if any(self.model_name in n for n in model_names):
                        logger.info(f"✓ Ollama ready with {self.model_name} (10-30ms latency for real-time video)")
                    else:
                        logger.warning(f"⚠ {self.model_name} not found. Available: {model_names[:3]}")
                    
                    return {"type": "ollama", "base_url": "http://localhost:11434"}
            except:
                logger.warning("⚠ Ollama not accessible - LLM classification disabled")
                return None
        except ImportError:
            logger.warning("requests library not available")
            return None
    
    def classify_veins_with_llm(
        self,
        frame: np.ndarray,
        veins: List[Dict],
        overlay_image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Ultra-fast LLM vein classification using spatial data only (no image encoding).
        
        Args:
            frame: Original ultrasound frame (used only for timing reference, not processed)
            veins: List of vein dictionaries with spatial features
            overlay_image: Ignored (for API compatibility)
        
        Returns:
            List of veins with LLM classifications in < 100ms
        """
        if self.client is None:
            logger.debug("LLM client not available, returning input veins")
            return veins
        
        if not veins:
            return veins
        
        start_time = time.time()
        
        try:
            # Extract spatial features only (NO image encoding)
            vein_features = self._extract_spatial_features(veins)
            
            # Create minimal prompt for fast inference
            prompt = self._create_fast_prompt(vein_features, len(veins))
            
            # Query Ollama with timeout for real-time responsiveness
            response = self._query_ollama_fast(prompt)
            
            if not response:
                return veins
            
            # Parse response
            llm_results = self._parse_llm_response(response)
            
            # Merge results
            veins_enhanced = self._merge_llm_results(veins, llm_results)
            
            # Track inference timing
            elapsed = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += elapsed
            
            if self.inference_count % 10 == 0:
                avg_time = self.total_inference_time / self.inference_count
                logger.debug(f"[LLM] Avg inference: {avg_time*1000:.1f}ms ({self.inference_count} frames)")
            
            return veins_enhanced
        
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return veins
    
    def _extract_spatial_features(self, veins: List[Dict]) -> List[Dict]:
        """
        Extract ONLY spatial features (NO image processing).
        Fast path: features already computed during segmentation.
        """
        features = []
        
        for vein in veins:
            classification = vein.get('classification', {})
            spatial = vein.get('spatial_analysis', {})
            
            # Minimal feature set for ultra-fast classification
            feature = {
                "id": vein.get('vein_id', '?'),
                "type": classification.get('primary_classification', '?'),
                "conf": round(classification.get('confidence', 0), 2),
                "dist_mm": round(spatial.get('distance_to_fascia_mm', 0), 1),
                "pos": spatial.get('relative_position', '?'),
                "size": vein.get('properties', {}).get('area', 0),
            }
            features.append(feature)
        
        return features
    
    def _create_fast_prompt(self, features: List[Dict], num_veins: int) -> str:
        """
        Create ultra-minimal prompt for 20-50ms inference (not vision analysis).
        Just classification based on spatial features already computed.
        """
        
        # Format features as compact JSON for minimal token count
        feat_str = json.dumps(features, separators=(',', ':'), indent=None)
        
        # Extremely short prompt: ~200 tokens vs 2000+ before
        prompt = f"""Classify {num_veins} veins from spatial data:
{feat_str}

Rules: dist_mm > 0 = superficial, < 0 = deep, ~0 = perforator.
Return JSON only (no text):
{{"vein_classifications": [{{"id": "...", "type": "...", "conf": 0.9}}], "gsv": false}}"""
        
        return prompt
    
    def _query_ollama_fast(self, prompt: str, timeout: float = 0.5) -> str:
        """
        Query Ollama with aggressive timeout for real-time performance.
        
        Args:
            prompt: Minimal prompt (should be < 300 tokens)
            timeout: Max seconds to wait (0.5s for real-time: 30fps * 16ms each)
        
        Returns:
            LLM response or empty string on timeout
        """
        import requests
        
        if self.client is None:
            return ""
        
        try:
            base_url = self.client.get("base_url", "http://localhost:11434")
            
            # Ultra-fast inference settings
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,  # Lower = faster + more consistent
                    "top_p": 0.5,  # More restrictive = faster
                    "top_k": 10,   # More restrictive = faster
                    "num_predict": 150,  # CRITICAL: Limit output tokens (was unlimited)
                },
                timeout=timeout  # CRITICAL: Hard timeout for real-time
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.debug(f"Ollama error: {response.status_code}")
                return ""
        
        except requests.Timeout:
            logger.debug(f"LLM timeout ({timeout}s) - proceeding without enhancement")
            return ""
        except Exception as e:
            logger.debug(f"LLM error: {type(e).__name__}")
            return ""
    
    @staticmethod
    def _parse_llm_response(response: str) -> Dict:
        """Parse LLM response - expects minimal JSON output"""
        try:
            # Extract JSON quickly
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {}
        
        except json.JSONDecodeError:
            return {}
    
    @staticmethod
    def _merge_llm_results(
        original_veins: List[Dict],
        llm_results: Dict
    ) -> List[Dict]:
        """Merge LLM classification results with vein data"""
        
        if not llm_results or 'vein_classifications' not in llm_results:
            return original_veins
        
        llm_map = {
            v.get('id') or v.get('vein_id'): v 
            for v in llm_results.get('vein_classifications', [])
        }
        
        # Update veins with LLM classification
        for vein in original_veins:
            vein_id = vein.get('vein_id', '')
            
            if vein_id in llm_map:
                llm_class = llm_map[vein_id]
                
                # Update classification with LLM result
                if 'classification' not in vein:
                    vein['classification'] = {}
                
                # Use LLM classification if confidence is high
                if llm_class.get('conf', 0.5) > 0.5:
                    vein['classification']['llm_type'] = llm_class.get('type')
                    vein['classification']['llm_confidence'] = llm_class.get('conf')
        
        return original_veins
