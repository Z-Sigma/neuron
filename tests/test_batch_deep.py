import time
from neuron.memory import Memory

def test_deep_batch():
    print("Initializing NEURON Memory for Deep Batch Test...")
    # Use InMemory store for testing to avoid DB issues
    from neuron.graph.in_memory_store import InMemoryGraphStore
    store = InMemoryGraphStore()
    memory = Memory(store=store)
    
    user_id = "deep_batch_tester"
    
    # We create a list of texts
    texts = [
        "The project code name is NEURON and it was founded in 2026 by Ajit.",
        "The project code name is NEURON and it was founded in 2026 by Ajit.", # Exact duplicate
        "NEURON is a cognitive memory framework designed for autonomous agents.",
        "SpaceX is a private aerospace manufacturer founded by Elon Musk.",
        "Starship is a fully reusable launch vehicle developed by SpaceX.",
        "Ajit is working on improving the NEURON framework.",
        "The Starship rocket aims to reach Mars.",
        "Python is the primary language for AI development at SpaceX.",
    ]
    
    print(f"\n--- Running Deep Batch on {len(texts)} chunks ---")
    start_time = time.time()
    
    # Using a small window size for testing
    result = memory.process_batch_deep(
        texts=texts,
        user_id=user_id,
        window_size=3,
        deduplication_threshold=0.95,
        knn_edges=2
    )
    
    end_time = time.time()
    
    print(f"\n--- Deep Batch Results ---")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Nodes Added: {result['nodes_added']}")
    print(f"Edges Added: {result['edges_added']}")
    print(f"Duplicates Dropped: {result['duplicates_dropped']}")
    
    # Verify nodes in store
    all_nodes = store.list_nodes(user_id)
    print(f"\nNodes in store: {len(all_nodes)}")
    for node in all_nodes:
        print(f" - [{node.label}] Entities: {node.entities}, Confidence: {node.confidence}")

    # Verify edges
    if result['nodes_added'] > 0:
        print("\nVerifying Entity Hubs...")
        # Check if SpaceX nodes are linked
        spacex_nodes = [n for n in all_nodes if "SpaceX" in n.entities]
        if len(spacex_nodes) >= 2:
            edges = store.get_edges(spacex_nodes[0].id)
            linked_ids = [e.to_node_id for e in edges] + [e.from_node_id for e in edges]
            if spacex_nodes[1].id in linked_ids:
                print("SUCCESS: SpaceX nodes are linked via Entity Hub logic!")
            else:
                print("WARNING: SpaceX nodes are NOT linked.")
        
    if result['duplicates_dropped'] == 1 and result['nodes_added'] == 7:
        print("\nSTATUS: SUCCESS (process_batch_deep worked correctly!)")
    else:
        print(f"\nSTATUS: PARTIAL (Added {result['nodes_added']}, expected 7. Dropped {result['duplicates_dropped']}, expected 1)")

if __name__ == "__main__":
    test_deep_batch()
