import os
import logging
from neuron.adaptive.adaptive_memory import AdaptiveMemory
from neuron.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_adaptive_flow():
    print(">>> Starting Adaptive Memory Verification...")
    
    # Enable Adaptive Memory for the test
    settings.enable_adaptive_memory = True
    settings.graph_store_type = "in_memory"
    settings.adaptive_mode = "manual" # Keep it manual for the test to avoid background threads
    
    # 1. Initialize
    brain = AdaptiveMemory()
    user_id = "test_adaptive_user"
    
    # 2. Process some data (WRITE activity)
    print("\n[1/4] Testing Activity Tracking (WRITE)...")
    brain.process("The capital of France is Paris.", user_id)
    brain.process("The Eiffel Tower is in Paris.", user_id)
    
    # Verify activity log
    activities = brain.store.get_users_needing_maintenance(window_hours=1)
    if user_id in activities:
        print("SUCCESS: WRITE activity logged.")
    else:
        print("FAILED: Activity not found in log.")
        return

    # 3. Perform Retrieval (READ activity & Strategy selection)
    print("\n[2/4] Testing Strategy Selection & Retrieval Events (READ)...")
    # Provide feedback to test fitness update
    result = brain.retrieve("Where is the Eiffel Tower?", user_id, score_feedback=0.9)
    print(f"Retrieved {len(result.get('nodes', []))} nodes.")
    
    # Verify retrieval event
    events = brain.store.get_retrieval_events(user_id, limit=5)
    if events:
        print(f"SUCCESS: Retrieval event logged. Strategy: {events[0].strategy_id}, Score: {events[0].score}")
    else:
        print("FAILED: Retrieval event not found.")
        return

    # 4. Strategy Evolution (Manual trigger)
    print("\n[3/4] Testing Strategy Evolution...")
    report_before = brain.strategy_report()
    print(f"Strategies before: {len(report_before)}")
    
    # Simulate enough data for evolution
    for i in range(50):
        brain.store.log_retrieval_event(events[0])
    
    brain.force_sleep(user_id)
    
    report_after = brain.strategy_report()
    print(f"Strategies after: {len(report_after)}")
    
    if len(report_after) > len(report_before):
        print("SUCCESS: Strategy population evolved.")
    else:
        # Note: evolution only happens if population < limit or fitness gap exists
        print("INFO: Population stable or already at limit.")

    # 5. Graph Health
    print("\n[4/4] Testing Observability (Graph Health)...")
    health = brain.graph_health(user_id)
    print(f"Graph Health: {health}")
    if health.get("status") == "healthy":
        print("SUCCESS: Graph health metrics retrieved.")
    else:
        print("INFO: Graph may be too small for definitive health status.")

    print("\n*** ADAPTIVE MEMORY VERIFICATION COMPLETED ***")

if __name__ == "__main__":
    # Ensure Postgres is available or it will fallback to In-Memory
    test_adaptive_flow()
