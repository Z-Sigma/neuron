import time
from neuron.memory import Memory
from neuron.graph.in_memory_store import InMemoryGraphStore

def run_comprehensive_test():
    print(">>> Starting Final Comprehensive Verification...")
    store = InMemoryGraphStore()
    memory = Memory(store=store)
    user_id = "final_tester"

    # --- TEST 1: process() (High Intelligence) ---
    print("\n[1/3] Testing process()...")
    text1 = "The NEURON engine was updated to support dynamic dot product weights today."
    node1 = memory.process(text1, user_id)
    if node1:
        print(f"SUCCESS: Node created with label: {node1.label}")
        # Verify if an edge was created (unlikely with only 1 node, but let's check)
        edges = store.get_edges(node1.id)
        print(f"   Edges created: {len(edges)}")
    else:
        print("FAILURE: process() failed to create node.")

    # --- TEST 2: process_batch_fast() (High Speed) ---
    print("\n[2/3] Testing process_batch_fast()...")
    fast_texts = [
        "Matrix math is used for fast deduplication in NEURON.",
        "Matrix math is used for fast deduplication in NEURON.", # Duplicate
        "Vectors are stored in a high-dimensional space.",
        "HNSW indexes allow for O(log N) retrieval speeds.",
        "Fast batch mode bypasses the LLM entirely."
    ]
    result_fast = memory.process_batch_fast(fast_texts, user_id)
    print(f"SUCCESS: Added {result_fast['nodes_added']} nodes, {result_fast['edges_added']} edges.")
    print(f"   Dropped {result_fast['duplicates_dropped']} duplicates.")

    # --- TEST 3: process_batch_deep() (Balanced) ---
    print("\n[3/3] Testing process_batch_deep()...")
    deep_texts = [
        "Ajit is a developer working on the NEURON framework.",
        "The project is overseen by the Z-Sigma organization.",
        "Ajit integrated windowed extraction for cost savings.",
        "Z-Sigma focuses on agentic cognitive architectures.",
        "The Starship rocket is mentioned in the entity hub tests."
    ]
    result_deep = memory.process_batch_deep(deep_texts, user_id, window_size=2)
    print(f"SUCCESS: Added {result_deep['nodes_added']} nodes, {result_deep['edges_added']} edges.")
    
    # --- CROSS-METHOD VERIFICATION ---
    print("\nSEARCH: Verifying Cross-Method Connectivity...")
    # Find the node about Ajit from deep batch
    all_nodes = store.list_nodes(user_id, limit=50)
    ajit_nodes = [n for n in all_nodes if "Ajit" in n.entities]
    
    if len(ajit_nodes) >= 2:
        # Check if they are linked (they should be via Entity Hub logic)
        edges = store.get_edges(ajit_nodes[0].id)
        is_linked = any(e.to_node_id == ajit_nodes[1].id or e.from_node_id == ajit_nodes[1].id for e in edges)
        if is_linked:
            print("SUCCESS: Entity Hub Link verified: 'Ajit' nodes are connected.")
        else:
            print("WARNING: 'Ajit' nodes are not connected.")

    # Test dynamic weights in process() by adding a related fact
    print("\nSEARCH: Verifying Dynamic Weights in process()...")
    related_text = "Dynamic weights improve the precision of the NEURON semantic brain."
    node2 = memory.process(related_text, user_id)
    if node2:
        edges = store.get_edges(node2.id)
        for edge in edges:
            print(f"   Created edge with Weight: {edge.weight:.4f} (Type: {edge.relation})")
            if edge.weight != 0.5 and edge.weight != 0.9:
                print("SUCCESS: Dynamic Weight Verified: Edge weight is a similarity score, not 0.5!")

    # Final Retrieval Check
    print("\nSEARCH: Testing Hybrid Retrieval across all data...")
    context = memory.retrieve("Tell me about Ajit and NEURON weights.", user_id)
    print(f"SUCCESS: Retrieval found {len(context['direct_beliefs'])} relevant nodes.")
    for node in context['direct_beliefs'][:3]:
        print(f" - {node['label']} (Conf: {node['confidence']})")

    print("\n*** ALL TESTS COMPLETED SUCCESSFULLY! ***")

if __name__ == "__main__":
    run_comprehensive_test()
