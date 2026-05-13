import time
import uuid
import random
import numpy as np
from neuron.memory import Memory
from neuron.models import Node, Edge
from neuron.graph.in_memory_store import InMemoryGraphStore

def run_scale_test(node_count=1000, edge_count=5000):
    print(f"--- STARTING NEURON SCALE TEST ({node_count} nodes, {edge_count} edges) ---")
    
    # Use default initialization (will pick up Postgres from .env)
    memory = Memory()
    store = memory.store
    user_id = "scale_tester_pg"
    
    # 1. Generate Synthetic Nodes
    print(f"Step 1: Generating {node_count} synthetic nodes...")
    start_time = time.time()
    nodes = []
    for i in range(node_count):
        embedding = np.random.randn(384).tolist()
        node = Node(
            user_id=user_id,
            label=f"Synthetic Belief Node #{i}: Specific detail about topic {i % 100}",
            confidence=random.uniform(0.5, 0.95),
            embedding=embedding
        )
        nodes.append(node)
    
    # BATCH ADD
    store.add_nodes_batch(nodes)
    
    gen_time = time.time() - start_time
    print(f"Created {node_count} nodes in {gen_time:.2f}s")

    # 2. Generate Synthetic Edges (The Hairball)
    print(f"Step 2: Creating {edge_count} random associative edges...")
    start_time = time.time()
    edges = []
    for _ in range(edge_count):
        node_a = random.choice(nodes)
        node_b = random.choice(nodes)
        if node_a.id != node_b.id:
            edges.append(Edge(
                from_node_id=node_a.id,
                to_node_id=node_b.id,
                relation="refines",
                weight=random.uniform(0.4, 0.9)
            ))
    
    # BATCH ADD
    store.add_edges_batch(edges)
    
    edge_time = time.time() - start_time
    print(f"Created {edge_count} edges in {edge_time:.2f}s")

    # 3. Stress Test Retrieval
    print("\nStep 3: Testing Retrieval Performance (3-Hop Graph Walk)...")
    # Query a random node's label
    test_node = random.choice(nodes)
    query = test_node.label
    
    start_time = time.time()
    context = memory.retrieve(query, user_id)
    retrieval_time = (time.time() - start_time) * 1000 # convert to ms
    
    print(f"--- RESULTS ---")
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {edge_count}")
    print(f"Retrieval Latency: {retrieval_time:.2f}ms")
    print(f"Direct Beliefs Found: {len(context['direct_beliefs'])}")
    print(f"Related Context Hops: {len(context['related_context'])}")
    
    if retrieval_time < 200:
        print("\nSTATUS: SUCCESS (Sub-200ms retrieval at scale)")
    else:
        print("\nSTATUS: WARNING (Latency over 200ms - Optimization suggested)")

if __name__ == "__main__":
    run_scale_test(node_count=2000, edge_count=10000)
