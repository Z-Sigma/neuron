from neuron import Memory, settings

# 1. Setup the Brain (Automatically uses .env or defaults)
# settings.graph_store_type = "in_memory" # You can override defaults here
brain = Memory()

# 2. Add a new belief
user_id = "test_user_123"
brain.process(
    text="The project code name is NEURON and it was founded in 2026.",
    user_id=user_id
)

# 3. Retrieve context for a question
context = brain.retrieve(
    query="What is the name of the project?",
    user_id=user_id
)

print("\n--- DIRECT BELIEFS ---")
for node in context["direct_beliefs"]:
    print(f"- {node['label']} (Confidence: {node['confidence']})")

print("\n--- RELATED CONTEXT HOPS ---")
for node in context["related_context"]:
    print(f"- {node['label']} (Confidence: {node['confidence']})")
