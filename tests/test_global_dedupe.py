import sys
from unittest.mock import MagicMock

# Mock sentence_transformers to avoid hang on import
sys.modules['sentence_transformers'] = MagicMock()

import os
sys.path.append(os.getcwd())

from neuron.memory import Memory
from neuron.graph.in_memory_store import InMemoryGraphStore
from neuron.models import AbstractionResult

def test_global_deduplication_fast():
    print("\n--- Testing Global Deduplication: FAST ---")
    store = InMemoryGraphStore()
    
    # Mock Embedder to avoid loading SentenceTransformer
    mock_embedder = MagicMock()
    def mock_embed(t):
        import numpy as np
        # Create a semi-unique vector based on the string
        vec = np.zeros(384)
        for i, char in enumerate(t[:384]):
            vec[i] = ord(char)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        return vec.tolist()

    mock_embedder.embed.side_effect = mock_embed
    mock_embedder.embed_batch.side_effect = lambda ts: [mock_embedder.embed(t) for t in ts]
    
    memory = Memory(store=store, embedder=mock_embedder)
    user_id = "test_user"
    
    texts = [
        "The capital of France is Paris.",
        "Einstein developed the theory of relativity."
    ]
    
    # 1. First Ingestion (Global Dedupe OFF)
    print("Step 1: Ingesting first batch (Global Dedupe OFF)...")
    res1 = memory.process_batch_fast(texts, user_id, global_deduplication=False)
    print(f"Results: {res1}")
    
    assert res1['nodes_added'] == 2
    
    # 2. Second Ingestion of SAME texts (Global Dedupe ON)
    print("\nStep 2: Ingesting second batch (Global Dedupe ON)...")
    res2 = memory.process_batch_fast(texts, user_id, global_deduplication=True)
    print(f"Results: {res2}")
    
    assert res2['nodes_added'] == 0
    assert res2['duplicates_dropped'] == 2
    
    # Verify evidence increment
    nodes = store.list_nodes(user_id)
    assert len(nodes) == 2
    for node in nodes:
        assert node.evidence_count == 2
    
    print("SUCCESS: Global Deduplication Fast verified!")

def test_global_deduplication_deep():
    print("\n--- Testing Global Deduplication: DEEP ---")
    store = InMemoryGraphStore()
    
    # Mock Embedder
    mock_embedder = MagicMock()
    def mock_embed(t):
        import numpy as np
        vec = np.zeros(384)
        for i, char in enumerate(t[:384]):
            vec[i] = ord(char)
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        return vec.tolist()
        
    mock_embedder.embed.side_effect = mock_embed
    mock_embedder.embed_batch.side_effect = lambda ts: [mock_embedder.embed(t) for t in ts]
    
    memory = Memory(store=store, embedder=mock_embedder)
    user_id = "test_user_deep"
    
    # Mock Abstraction Engine to avoid LLM calls
    memory.engine.batch_extract = MagicMock(return_value=[
        AbstractionResult(
            label="Photosynthesis process",
            confidence=0.9,
            entities=["Sunlight", "Chlorophyll"],
            abstraction_level="pattern"
        )
    ])
    
    texts = ["Photosynthesis uses sunlight."]
    
    # 1. Ingest first time
    print("Step 1: Ingesting Deep batch (Global Dedupe OFF)...")
    res1 = memory.process_batch_deep(texts, user_id, global_deduplication=False)
    print(f"Results: {res1}")
    
    node = store.list_nodes(user_id)[0]
    assert "Sunlight" in node.entities
    assert node.evidence_count == 1
    
    # 2. Mock second call with a NEW entity discovered for the same concept
    memory.engine.batch_extract = MagicMock(return_value=[
        AbstractionResult(
            label="Photosynthesis process",
            confidence=0.9,
            entities=["Sunlight", "Carbon Dioxide"], # New entity added
            abstraction_level="pattern"
        )
    ])
    
    print("\nStep 2: Ingesting SAME Deep batch (Global Dedupe ON) with new entities...")
    res2 = memory.process_batch_deep(texts, user_id, global_deduplication=True)
    print(f"Results: {res2}")
    
    assert res2['nodes_added'] == 0
    assert res2['duplicates_dropped'] == 1
    
    # Verify evidence increment and entity merge
    updated_nodes = store.list_nodes(user_id)
    assert len(updated_nodes) == 1
    updated_node = updated_nodes[0]
    assert updated_node.evidence_count == 2
    assert "Sunlight" in updated_node.entities
    assert "Carbon Dioxide" in updated_node.entities
    
    print("SUCCESS: Global Deduplication Deep verified!")

if __name__ == "__main__":
    try:
        test_global_deduplication_fast()
        test_global_deduplication_deep()
        print("\nALL VERIFICATION TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
