import time
from neuron.memory import Memory

def test_fast_batch():
    print("Initializing NEURON Memory for Fast Batch Test...")
    memory = Memory()
    user_id = "fast_batch_tester"
    
    # We create a list of texts, intentionally including exact and near duplicates
    texts = [
        "The project code name is NEURON and it was founded in 2026.",
        "The project code name is NEURON and it was founded in 2026.", # Exact duplicate
        "NEURON is a cognitive memory framework.",
        "NEURON acts as a cognitive memory engine.", # Near duplicate
        "Graph databases like Neo4j are powerful.",
        "Neo4j is a great graph database.", # Near duplicate
        "Python is a programming language used for AI.",
        "Vector search is crucial for modern AI.",
        "Sentence transformers convert text to vectors.",
        "I love building autonomous agents."
    ]
    
    # Add a lot of random noise to test speed and k-NN scaling
    for i in range(500):
        texts.append(f"Synthetic filler chunk {i} about generic AI topics.")
        # Add a duplicate for every 10th filler chunk to ensure deduplication drops exactly 50
        if i % 10 == 0:
            texts.append(f"Synthetic filler chunk {i} about generic AI topics.")
    
    print(f"\n--- Running Fast Batch on {len(texts)} chunks ---")
    start_time = time.time()
    
    result = memory.process_batch_fast(
        texts=texts,
        user_id=user_id,
        deduplication_threshold=0.95,
        knn_edges=3
    )
    
    end_time = time.time()
    
    print(f"\n--- Fast Batch Results ---")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Nodes Added: {result['nodes_added']}")
    print(f"Edges Added: {result['edges_added']}")
    print(f"Duplicates Dropped: {result['duplicates_dropped']}")
    
    if result['duplicates_dropped'] > 0 and result['edges_added'] > 0:
        print("\nSTATUS: SUCCESS (Option 2.5 successfully dropped duplicates and built a semantic web!)")
    else:
        print("\nSTATUS: FAILURE (Something went wrong with deduplication or edge generation)")

if __name__ == "__main__":
    test_fast_batch()
