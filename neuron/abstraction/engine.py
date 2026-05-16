import json
import logging
from typing import List
from neuron.models import AbstractionResult
from neuron.llm.provider import get_llm_provider

logger = logging.getLogger(__name__)

class AbstractionEngine:
    def __init__(self, provider_type: str = None):
        self.provider = get_llm_provider(provider_type)

    def extract(self, text: str, retries: int = 2) -> AbstractionResult:
        from neuron.config import settings
        prompt = f"{settings.abstraction_prompt}\n\nInput Text: '{text}'"
        
        for attempt in range(retries + 1):
            try:
                content = self.provider.completion(prompt, system_prompt="You are a semantic abstraction engine.")
                data = self._parse_json(content)
                data = self._normalize_fields(data)
                return AbstractionResult(**data)
            except Exception as e:
                logger.warning(f"Abstraction attempt {attempt + 1} failed: {e}")
                if attempt == retries:
                    # Return a safe fallback instead of crashing
                    logger.error(f"All abstraction attempts failed. Flagging fallback.")
                    return AbstractionResult(
                        label=text[:150],
                        confidence=0.5,
                        domain_tags=[],
                        entities=[],
                        metadata={"fallback": True}
                    )

    def batch_extract(self, texts: List[str], retries: int = 2) -> List[AbstractionResult]:
        """Extract metadata for a batch of texts using a single LLM call."""
        from neuron.config import settings
        
        # Format the batch of texts
        formatted_texts = "\n---\n".join([f"Chunk {i+1}: {text}" for i, text in enumerate(texts)])
        prompt = f"{settings.batch_abstraction_prompt}\n\nInput Chunks:\n{formatted_texts}"
        
        for attempt in range(retries + 1):
            try:
                content = self.provider.completion(prompt, system_prompt="You are a batched semantic abstraction engine.")
                results_data = self._parse_json_list(content)
                
                # Normalize and validate each result
                final_results = []
                for i, data in enumerate(results_data):
                    data = self._normalize_fields(data)
                    # Use provided chunk text if label is missing or too short
                    if not data.get("label"):
                        data["label"] = texts[i][:150]
                        data.setdefault("metadata", {})["fallback"] = True
                    final_results.append(AbstractionResult(**data))
                
                # Ensure we return the same number of results as input texts
                if len(final_results) != len(texts):
                    logger.warning(f"Batch size mismatch: expected {len(texts)}, got {len(final_results)}. Padding with fallbacks.")
                    while len(final_results) < len(texts):
                        idx = len(final_results)
                        final_results.append(AbstractionResult(label=texts[idx][:150], confidence=0.5, metadata={"fallback": True}))
                    final_results = final_results[:len(texts)]
                
                return final_results
            except Exception as e:
                logger.warning(f"Batch abstraction attempt {attempt + 1} failed: {e}")
                if attempt == retries:
                    logger.error(f"All batch abstraction attempts failed. Flagging all as fallbacks.")
                    return [AbstractionResult(label=t[:150], confidence=0.5, metadata={"fallback": True}) for t in texts]

    def _parse_json_list(self, text: str) -> List[dict]:
        """Extracts a JSON list from LLM response."""
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            json_str = text[start:end]
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        raise ValueError("No JSON list found in response")

    def _normalize_fields(self, data: dict) -> dict:
        """Normalize LLM output to match Pydantic Literal constraints."""
        # 1. Temporal Stability
        stability_options = ["stable", "volatile", "time-bound"]
        stability_map = {
            "constant": "stable", "permanent": "stable", 
            "temporary": "time-bound", "dynamic": "volatile"
        }
        
        if "temporal_stability" in data:
            val = str(data["temporal_stability"]).lower().strip()
            data["temporal_stability"] = stability_map.get(val, val)
            if data["temporal_stability"] not in stability_options:
                data["temporal_stability"] = "stable" # Default safe fallback
        else:
            data["temporal_stability"] = "stable"
        
        # 2. Abstraction Level
        level_options = ["specific", "pattern", "principle"]
        if "abstraction_level" in data:
            val = str(data["abstraction_level"]).lower().strip()
            if val not in level_options:
                # Attempt fuzzy match
                if "spec" in val: data["abstraction_level"] = "specific"
                elif "patt" in val: data["abstraction_level"] = "pattern"
                elif "princ" in val: data["abstraction_level"] = "principle"
                else: data["abstraction_level"] = "specific"
            else:
                data["abstraction_level"] = val
        else:
            data["abstraction_level"] = "specific"
        
        # 3. List fields (Robust string-to-list conversion)
        for field in ["entities", "domain_tags", "contradiction_candidates", "related_concepts"]:
            val = data.get(field, [])
            if isinstance(val, str):
                # Split by comma or semicolon and clean up
                import re
                data[field] = [s.strip() for s in re.split(r'[,;]', val) if s.strip()]
            elif not isinstance(val, list):
                data[field] = []
        
        # 4. Confidence Clamping
        try:
            conf = float(data.get("confidence", 0.5))
            data["confidence"] = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            data["confidence"] = 0.5
        
        return data

    def _parse_json(self, text: str) -> dict:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = text[start:end]
            return json.loads(json_str)
        raise ValueError("No JSON found in response")
