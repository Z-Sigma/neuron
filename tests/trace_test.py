import uuid
from neuron import Memory

def trace_cognition():
    memory = Memory()
    user_id = str(uuid.uuid4())
    
    print("\n[STEP 1: Learning Facts]")
    memory.process("The director of the movie 'Inception' is Christopher Nolan.", user_id)
    memory.process("Christopher Nolan was born in Westminster, London.", user_id)
    
    print("\n[STEP 2: Multi-hop Retrieval]")
    query = "Tell me about the person who directed Inception."
    result = memory.retrieve(query, user_id)
    
    print(f"\nQuery: {query}")
    print("\n[Direct Beliefs found at Hop 0]:")
    for b in result['direct_beliefs']:
        print(f" - {b['label']} (Conf: {b['confidence']})")
        
    print("\n[Related Context found via Graph Traversal (Hops 1-3)]:")
    for b in result['related_context']:
        print(f" - {b['label']} (Conf: {b['confidence']})")

    all_found = [b['label'] for b in result['direct_beliefs']] + [b['label'] for b in result['related_context']]
    if any("London" in label for label in all_found):
        print("\nSUCCESS: Multi-hop connection detected! The brain connected Inception to London via Nolan.")
    else:
        print("\nFAILED: No connection detected.")

if __name__ == "__main__":
    trace_cognition()
