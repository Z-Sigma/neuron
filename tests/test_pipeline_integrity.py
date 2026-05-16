"""
Verify each pipeline stage uses its defined implementation (not silent stubs).
Uses call spies on AbstractionEngine.provider.completion and Embedder.embed.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())

from neuron.graph.in_memory_store import InMemoryGraphStore
from neuron.memory import Memory
from neuron.models import AbstractionResult
from neuron.abstraction.engine import AbstractionEngine
from neuron.daemon.coherence import CoherenceDaemon
from neuron.models import Node, Edge
from uuid import uuid4


@pytest.fixture
def spied_memory():
    """In-memory store + tracked LLM calls."""
    store = InMemoryGraphStore()
    llm_calls = []

    class SpyProvider:
        def completion(self, prompt: str, system_prompt: str = "") -> str:
            llm_calls.append({"prompt": prompt, "system": system_prompt})
            # Return valid abstraction JSON (not raw text fallback shape)
            return (
                '{"label": "Paris is the capital of France", '
                '"confidence": 0.92, "domain_tags": ["geography"], '
                '"entities": ["Paris", "France"], '
                '"contradiction_candidates": [], '
                '"abstraction_level": "specific", "temporal_stability": "stable"}'
            )

    engine = AbstractionEngine()
    engine.provider = SpyProvider()

    def _mock_embed(text: str):
        import numpy as np
        vec = np.zeros(384)
        for i, ch in enumerate(text[:384]):
            vec[i] = (ord(ch) % 256) / 255.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = _mock_embed
    mock_embedder.embed_batch.side_effect = lambda texts: [_mock_embed(t) for t in texts]

    memory = Memory(store=store, embedder=mock_embedder, engine=engine)
    return memory, llm_calls, mock_embedder


def test_process_novel_calls_llm_abstraction(spied_memory):
    memory, llm_calls, embedder = spied_memory
    user = f"integrity_{uuid4().hex[:8]}"

    node = memory.process(
        "The capital of France is Paris, a major European city.",
        user,
        novelty_threshold=0.01,  # force novel path
    )

    assert node is not None
    assert len(llm_calls) == 1, "Novel process() must call LLM exactly once via engine.extract"
    assert "semantic abstraction" in llm_calls[0]["system"].lower()
    assert node.label == "Paris is the capital of France"
    assert node.label != "The capital of France is Paris, a major European city."[:150]
    embedder.embed.assert_called()


def test_process_replay_skips_llm_when_similar(spied_memory):
    """Re-ingesting the abstracted label should confirm, not re-call LLM."""
    memory, llm_calls, _ = spied_memory
    user = f"integrity_{uuid4().hex[:8]}"

    first = memory.process(
        "The capital of France is Paris, a major European city.",
        user,
        novelty_threshold=0.01,
    )
    assert first is not None
    llm_calls.clear()

    # Embed/query using the stored belief label (high similarity to itself)
    replay = memory.process(first.label, user, novelty_threshold=0.01)

    assert replay is None, "Near-duplicate of stored belief should not create a node"
    assert len(llm_calls) == 0, "Confirmation path must not call LLM"


def test_retrieve_never_calls_llm(spied_memory):
    memory, llm_calls, embedder = spied_memory
    user = f"integrity_{uuid4().hex[:8]}"

    memory.process("Einstein developed the theory of relativity.", user, novelty_threshold=0.01)
    llm_calls.clear()
    embedder.embed.reset_mock()

    result = memory.retrieve("Who developed relativity?", user)

    assert len(llm_calls) == 0, "Retrieval must not call chat LLM"
    assert embedder.embed.called, "Retrieval must embed the query"
    assert "direct_beliefs" in result


def test_batch_fast_skips_llm_uses_raw_labels(spied_memory):
    memory, llm_calls, _ = spied_memory
    user = f"integrity_{uuid4().hex[:8]}"

    texts = ["Raw chunk one.", "Raw chunk two."]
    result = memory.process_batch_fast(texts, user)

    assert len(llm_calls) == 0, "process_batch_fast intentionally skips LLM"
    nodes = memory.store.list_nodes(user)
    labels = {n.label for n in nodes}
    assert "Raw chunk one." in labels


def test_batch_deep_calls_llm(spied_memory):
    memory, llm_calls, _ = spied_memory
    user = f"integrity_{uuid4().hex[:8]}"

    memory.engine.provider.completion = MagicMock(
        return_value='[{"label": "Deep abstracted belief", "confidence": 0.85, '
        '"domain_tags": [], "entities": [], "abstraction_level": "specific", '
        '"temporal_stability": "stable"}]'
    )
    llm_calls.clear()

    memory.process_batch_deep(["Some deep batch text."], user, window_size=1)

    assert memory.engine.provider.completion.called, "process_batch_deep must call batch_extract LLM"


def test_coherence_tie_calls_llm():
    store = InMemoryGraphStore()
    llm_calls = []

    class SpyProvider:
        def completion(self, prompt: str, system_prompt: str = "") -> str:
            llm_calls.append(prompt)
            return '{"action": "discard_b", "reason": "test"}'

    daemon = CoherenceDaemon(store)
    daemon.provider = SpyProvider()
    user = f"integrity_{uuid4().hex[:8]}"

    a = Node(user_id=user, label="A", confidence=0.8, evidence_count=5)
    b = Node(user_id=user, label="B", confidence=0.8, evidence_count=5)
    store.add_node(a)
    store.add_node(b)
    store.add_edge(Edge(from_node_id=a.id, to_node_id=b.id, relation="contradicts", weight=1.0))

    daemon.resolve_conflicts(user)

    assert len(llm_calls) == 1, "Tied SNR conflicts must invoke coherence LLM"


def test_coherence_clear_winner_skips_llm():
    store = InMemoryGraphStore()
    llm_calls = []

    class SpyProvider:
        def completion(self, prompt: str, system_prompt: str = "") -> str:
            llm_calls.append(prompt)
            return '{"action": "merge", "label": "x"}'

    daemon = CoherenceDaemon(store)
    daemon.provider = SpyProvider()
    user = f"integrity_{uuid4().hex[:8]}"

    strong = Node(user_id=user, label="Strong", confidence=0.9, evidence_count=50)
    weak = Node(user_id=user, label="Weak", confidence=0.4, evidence_count=1)
    store.add_node(strong)
    store.add_node(weak)
    store.add_edge(Edge(from_node_id=strong.id, to_node_id=weak.id, relation="contradicts", weight=1.0))

    daemon.resolve_conflicts(user)

    assert len(llm_calls) == 0, "Clear SNR winner must not need LLM"


@pytest.mark.integration
def test_live_groq_abstraction_not_raw_fallback():
    """Real Groq call: label must differ from truncated input fallback."""
    from neuron.config import settings

    if not settings.groq_api_key:
        pytest.skip("GROQ_API_KEY not set")

    store = InMemoryGraphStore()
    memory = Memory(store=store)
    user = f"live_{uuid4().hex[:8]}"
    raw = (
        "In 1969, NASA's Apollo 11 mission landed astronauts Armstrong and Aldrin on the Moon."
    )

    node = memory.process(raw, user, novelty_threshold=0.01)
    assert node is not None
    assert node.label != raw[:150], "Should be LLM-distilled, not raw-text fallback"
    assert len(node.label) > 10
