import json
import logging
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
                    logger.error(f"All abstraction attempts failed. Using raw text as fallback.")
                    return AbstractionResult(
                        label=text[:150],
                        confidence=0.5,
                        domain_tags=[],
                        entities=[],
                    )

    def _normalize_fields(self, data: dict) -> dict:
        """Normalize LLM output to match Pydantic Literal constraints."""
        stability_map = {"constant": "stable", "permanent": "stable", "temporary": "time-bound"}
        level_map = {}
        
        if "temporal_stability" in data:
            val = data["temporal_stability"].lower().strip()
            data["temporal_stability"] = stability_map.get(val, val)
        
        if "abstraction_level" in data:
            data["abstraction_level"] = data["abstraction_level"].lower().strip()
        
        # Ensure entities is always a list
        if "entities" not in data:
            data["entities"] = []
        
        return data

    def _parse_json(self, text: str) -> dict:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = text[start:end]
            return json.loads(json_str)
        raise ValueError("No JSON found in response")
