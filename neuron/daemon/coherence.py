from datetime import datetime, timezone
from typing import List
import json
import logging
import re
from neuron.models import Node, Edge
from neuron.graph.store_interface import GraphStore
from neuron.llm.provider import get_llm_provider

logger = logging.getLogger(__name__)

class CoherenceDaemon:
    def __init__(self, store: GraphStore, provider_type: str = None):
        self.store = store
        self.provider = get_llm_provider(provider_type)

    def run_cycle(self, user_id: str, always_use_llm: bool = False):
        print(f"Starting Coherence Cycle for user: {user_id}")
        self.resolve_conflicts(user_id, always_use_llm=always_use_llm)
        self.prune_stale_nodes(user_id)
        self.strengthen_beliefs(user_id)
        print(f"Coherence Cycle complete for user: {user_id}")

    def resolve_conflicts(self, user_id: str, always_use_llm: bool = False):
        pairs = self.store.get_contradiction_pairs(user_id)
        for node_a, node_b, edge_id in pairs:
            # SNR Heuristic: Signal = Confidence * Evidence
            signal_a = node_a.confidence * node_a.evidence_count
            signal_b = node_b.confidence * node_b.evidence_count
            
            # If one is significantly stronger (>20% diff), resolve immediately (unless always_use_llm is True)
            if not always_use_llm:
                if signal_a > signal_b * 1.2:
                    node_b.deprecated = True
                    self.store.update_node(node_b)
                    continue
                elif signal_b > signal_a * 1.2:
                    node_a.deprecated = True
                    self.store.update_node(node_a)
                    continue
            
            # Tie/Ambiguity or forced LLM: Resolve with LLM
            resolution_text = self._resolve_with_llm(node_a, node_b)
            try:
                cleaned_json = self._clean_json_response(resolution_text)
                res = json.loads(cleaned_json)
                action = res.get("action")
                
                if action == "merge":
                    merged_node = Node(
                        user_id=user_id,
                        label=res.get("label"),
                        confidence=(node_a.confidence + node_b.confidence) / 2,
                        evidence_count=node_a.evidence_count + node_b.evidence_count
                    )
                    self.store.add_node(merged_node)
                    node_a.deprecated = True
                    node_b.deprecated = True
                    self.store.update_node(node_a)
                    self.store.update_node(node_b)
                    logger.info(f"Merged nodes {node_a.id} and {node_b.id} into new node.")
                elif action == "discard_a":
                    node_a.deprecated = True
                    self.store.update_node(node_a)
                    logger.info(f"Discarded node {node_a.id} in favor of {node_b.id}")
                elif action == "discard_b":
                    node_b.deprecated = True
                    self.store.update_node(node_b)
                    logger.info(f"Discarded node {node_b.id} in favor of {node_a.id}")
            except json.JSONDecodeError as e:
                logger.error(f"Coherence failure: LLM returned invalid JSON for user {user_id}. Error: {e}")
                logger.debug(f"Raw LLM response: {resolution_text}")
            except Exception as e:
                logger.error(f"Unexpected error in coherence resolution: {e}")

    def prune_stale_nodes(self, user_id: str):
        stale_nodes = self.store.get_stale_nodes(user_id)
        for node in stale_nodes:
            node.deprecated = True
            self.store.update_node(node)

    def strengthen_beliefs(self, user_id: str):
        nodes = self.store.get_nodes_for_strengthening(user_id)
        for node in nodes:
            node.confidence = min(node.confidence * 1.1, 0.95)
            node.last_confirmed_at = datetime.now(timezone.utc)
            self.store.update_node(node)

    def _resolve_with_llm(self, node_a: Node, node_b: Node) -> str:
        prompt = f"""
Two beliefs in the memory graph are in conflict. Resolve them.
Belief A: "{node_a.label}" (Confidence: {node_a.confidence})
Belief B: "{node_b.label}" (Confidence: {node_b.confidence})

Output ONLY valid JSON:
{{
  "action": "merge | discard_a | discard_b",
  "label": "if merge, the new synthesized belief text",
  "reason": "short explanation"
}}
"""
        return self.provider.completion(prompt, system_prompt="You are a memory coherence daemon.")

    def _clean_json_response(self, text: str) -> str:
        """Removes markdown code blocks and extra whitespace from LLM JSON responses."""
        # Remove markdown code blocks like ```json ... ```
        text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
        return text.strip()
