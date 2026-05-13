from neuron.memory import Memory
import time

def run_demo():
    print("Initializing NEURON Memory Layer...")
    # Initialize Memory (will fall back to InMemory if Postgres is down)
    memory = Memory()
    user_id = "correctness_tester_1"

    print("\n--- Phase 1: Storage & Abstraction ---")
    text1 = "I really hate having to re-explain the context of my projects every time we talk."
    print(f"Processing: {text1}")
    node1 = memory.process(text1, user_id)
    if node1:
        print(f"Stored Abstraction: {node1.label} (Confidence: {node1.confidence})")

    # Let's simulate a specific memory we might "forget"
    text2 = "My secret project code name is 'Stardust'."
    print(f"\nProcessing: {text2}")
    node2 = memory.process(text2, user_id)
    print(f"Stored Abstraction: {node2.label}")

    print("\n--- Phase 2: Forgetting (Soft Delete) ---")
    print("Simulating Synaptic Pruning: Archiving 'Stardust' memory...")
    node2.deprecated = True
    memory.store.update_node(node2)
    
    # Verify it's gone from active search
    context_before = memory.retrieve("What is my project code name?", user_id)
    if not any(b['label'] == node2.label for b in context_before['direct_beliefs']):
        print("Success: 'Stardust' is currently hidden from active memory.")

    print("\n--- Phase 3: Deep Recall & Reactivation ---")
    print("Querying for the forgotten secret...")
    # This should trigger Deep Recall in the Retriever
    context_after = memory.retrieve("Remind me about project Stardust", user_id)
    
    found = any("Stardust" in b['label'] for b in context_after['direct_beliefs'])
    if found:
        print("Success! NEURON triggered Deep Recall and reactivated the archived memory.")
        print(f"Retrieved Belief: {context_after['direct_beliefs'][0]['label']}")
    else:
        print("Deep Recall failed to find the archived node.")

    print("\n--- Final Context State ---")
    print("Active Beliefs in Context:")
    for b in context_after['direct_beliefs']:
        print(f"- {b['label']} (Conf: {b['confidence']})")

if __name__ == "__main__":
    run_demo()
