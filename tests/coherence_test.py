import sys
from unittest.mock import MagicMock

# Mock sentence_transformers to avoid hang on import
sys.modules['sentence_transformers'] = MagicMock()

import unittest
from uuid import uuid4
from neuron.models import Node, Edge
from neuron.graph.in_memory_store import InMemoryGraphStore
from neuron.daemon.coherence import CoherenceDaemon

class MockProvider:
    def completion(self, prompt, system_prompt=None):
        return '{"action": "merge", "label": "merged belief"}'

class TestCoherenceSNR(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryGraphStore()
        self.daemon = CoherenceDaemon(self.store)
        self.daemon.provider = MockProvider()
        self.user_id = "test_user"

    def test_snr_resolution_strong_vs_weak(self):
        # Create a strong belief (high signal)
        strong_node = Node(
            id=uuid4(),
            user_id=self.user_id,
            label="The Earth is a sphere.",
            confidence=0.9,
            evidence_count=50  # Signal = 45
        )
        
        # Create a weak, contradictory belief (low signal)
        weak_node = Node(
            id=uuid4(),
            user_id=self.user_id,
            label="The Earth is flat.",
            confidence=0.4,
            evidence_count=1   # Signal = 0.4
        )
        
        self.store.add_node(strong_node)
        self.store.add_node(weak_node)
        
        # Create a contradiction edge
        edge = Edge(
            from_node_id=strong_node.id,
            to_node_id=weak_node.id,
            relation="contradicts",
            weight=1.0
        )
        self.store.add_edge(edge)
        
        # Mock the store's contradiction pair getter (if needed)
        # Note: In-memory store might need get_contradiction_pairs implemented
        # Let's ensure the daemon can see the conflict
        
        # Run resolution
        self.daemon.resolve_conflicts(self.user_id)
        
        # Verify results
        # The weak node should be deprecated
        updated_weak = self.store.get_node(weak_node.id)
        updated_strong = self.store.get_node(strong_node.id)
        
    def test_snr_resolution_reverse(self):
        # Weak vs Strong (B is stronger)
        weak_node = Node(id=uuid4(), user_id=self.user_id, label="A", confidence=0.5, evidence_count=1)
        strong_node = Node(id=uuid4(), user_id=self.user_id, label="B", confidence=0.9, evidence_count=10)
        self.store.add_node(weak_node)
        self.store.add_node(strong_node)
        self.store.add_edge(Edge(from_node_id=weak_node.id, to_node_id=strong_node.id, relation="contradicts"))
        
        self.daemon.resolve_conflicts(self.user_id)
        
        self.assertTrue(self.store.get_node(weak_node.id).deprecated)
        self.assertFalse(self.store.get_node(strong_node.id).deprecated)

    def test_llm_merge_resolution(self):
        # Similar signals - triggers LLM
        node_a = Node(id=uuid4(), user_id=self.user_id, label="Fact A", confidence=0.8, evidence_count=5)
        node_b = Node(id=uuid4(), user_id=self.user_id, label="Fact B", confidence=0.8, evidence_count=5)
        self.store.add_node(node_a)
        self.store.add_node(node_b)
        self.store.add_edge(Edge(from_node_id=node_a.id, to_node_id=node_b.id, relation="contradicts"))
        
        # Mock LLM returns a merge
        self.daemon.provider.completion = MagicMock(return_value='{"action": "merge", "label": "Merged Fact", "reason": "Consistent"}')
        
        self.daemon.resolve_conflicts(self.user_id)
        
        # Both originals should be deprecated
        self.assertTrue(self.store.get_node(node_a.id).deprecated)
        self.assertTrue(self.store.get_node(node_b.id).deprecated)
        
        # New merged node should exist
        all_nodes = self.store.list_nodes(self.user_id)
        merged_nodes = [n for n in all_nodes if n.label == "Merged Fact"]
        self.assertEqual(len(merged_nodes), 1)
        self.assertEqual(merged_nodes[0].evidence_count, 10)

    def test_strengthen_beliefs(self):
        # Node with high evidence should be strengthened
        node = Node(id=uuid4(), user_id=self.user_id, label="Truth", confidence=0.5, evidence_count=10)
        self.store.add_node(node)
        
        self.daemon.strengthen_beliefs(self.user_id)
        
        updated = self.store.get_node(node.id)
        self.assertGreater(updated.confidence, 0.5)
        self.assertEqual(updated.confidence, 0.55) # 0.5 * 1.1

    def test_prune_stale_nodes(self):
        # Mock store to return a stale node
        stale_node = Node(id=uuid4(), user_id=self.user_id, label="Old News", confidence=0.9)
        self.store.add_node(stale_node)
        self.store.get_stale_nodes = MagicMock(return_value=[stale_node])
        
        self.daemon.prune_stale_nodes(self.user_id)
        
        updated = self.store.get_node(stale_node.id)
        self.assertTrue(updated.deprecated)

    def test_llm_markdown_json_resolution(self):
        # Similar signals - LLM returns JSON wrapped in markdown
        node_a = Node(id=uuid4(), user_id=self.user_id, label="A", confidence=0.8, evidence_count=5)
        node_b = Node(id=uuid4(), user_id=self.user_id, label="B", confidence=0.8, evidence_count=5)
        self.store.add_node(node_a)
        self.store.add_node(node_b)
        self.store.add_edge(Edge(from_node_id=node_a.id, to_node_id=node_b.id, relation="contradicts"))
        
        # Mock LLM returns markdown-wrapped JSON
        markdown_json = '```json\n{"action": "discard_a", "reason": "B is better"}\n```'
        self.daemon.provider.completion = MagicMock(return_value=markdown_json)
        
        self.daemon.resolve_conflicts(self.user_id)
        
        # Should successfully parse despite the markdown
        self.assertTrue(self.store.get_node(node_a.id).deprecated)
        self.assertFalse(self.store.get_node(node_b.id).deprecated)

if __name__ == "__main__":
    unittest.main()
