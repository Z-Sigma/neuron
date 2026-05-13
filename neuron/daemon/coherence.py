from datetime import datetime
from typing import List
import json
from neuron.models import Node, Edge
from neuron.graph.store_interface import GraphStore
from neuron.llm.provider import get_llm_provider

class CoherenceDaemon:
    def __init__(self, store: GraphStore, provider_type: str = None):
        self.store = store
        self.provider = get_llm_provider(provider_type)

    def run_cycle(self, user_id: str):
        print(f"Starting Coherence Cycle for user: {user_id}")
        self.resolve_conflicts(user_id)
        self.prune_stale_nodes(user_id)
        self.strengthen_beliefs(user_id)
        print(f"Coherence Cycle complete for user: {user_id}")

    def resolve_conflicts(self, user_id: str):
        pairs = self.store.get_contradiction_pairs(user_id)
        for node_a, node_b, edge_id in pairs:
            resolution_text = self._resolve_with_llm(node_a, node_b)
            try:
                res = json.loads(resolution_text)
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
                elif action == "discard_a":
                    node_a.deprecated = True
                    self.store.update_node(node_a)
                elif action == "discard_b":
                    node_b.deprecated = True
                    self.store.update_node(node_b)
            except:
                pass

    def prune_stale_nodes(self, user_id: str):
        stale_nodes = self.store.get_stale_nodes(user_id)
        for node in stale_nodes:
            node.deprecated = True
            self.store.update_node(node)

    def strengthen_beliefs(self, user_id: str):
        nodes = self.store.get_nodes_for_strengthening(user_id)
        for node in nodes:
            node.confidence = min(node.confidence * 1.1, 0.95)
            node.last_confirmed_at = datetime.utcnow()
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
