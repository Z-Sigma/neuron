import logging
import json
from typing import List
from neuron.models import RetrievalEvent
from neuron.config import settings

logger = logging.getLogger(__name__)

class ImplicitScorer:
    """
    Evaluates retrieval quality without user feedback.
    Uses LLM-based 'Judging' in the background.
    """
    def __init__(self, provider_type: str = None):
        from neuron.llm.provider import get_llm_provider
        self.provider = get_llm_provider(provider_type)
        self.model = settings.judge_llm_model

    def score_events(self, events: List[RetrievalEvent], use_llm: bool = True) -> List[float]:
        """
        Takes a list of events and returns a matching list of scores (0-1).
        Uses LLM-based 'Judging' if use_llm is True and context is available.
        """
        if not events: return []
        
        scores = []
        for event in events:
            # If use_llm is True and we have context, try LLM judge
            if use_llm and event.retrieved_context:
                try:
                    score = self.judge_with_llm(event.query, event.retrieved_context)
                    scores.append(score)
                    continue
                except Exception as e:
                    logger.warning(f"LLM Judging failed for event {event.id}: {e}. Falling back to heuristic.")
            
            # Heuristic-based fallback
            base_score = 0.5
            if event.nodes_found > 0:
                base_score += 0.2
            if event.nodes_found > 5:
                base_score += 0.1
            
            # Penalize empty results if query seems meaningful
            if event.nodes_found == 0 and len(event.query) > 10:
                base_score -= 0.3
            
            scores.append(max(0.0, min(1.0, base_score)))
            
        return scores

    def judge_with_llm(self, query: str, context: List[str]) -> float:
        """
        Uses an LLM to judge if the context is relevant to the query.
        """
        if not context:
            return 0.0
            
        prompt = f"""
        Query: {query}
        Retrieved Context: {context}
        
        On a scale of 0.0 to 1.0, how relevant and useful is this context for answering the query?
        Return ONLY a JSON object: {{"score": 0.X}}
        """
        
        content = self.provider.completion(prompt, system_prompt="You are an expert retrieval quality judge.")
        try:
            # Clean JSON if necessary (reuse logic or keep simple)
            import re
            cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', content, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            return float(data.get("score", 0.5))
        except Exception as e:
            logger.error(f"Error parsing LLM judge response: {e}")
            raise
