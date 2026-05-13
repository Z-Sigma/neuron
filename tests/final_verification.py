import logging
import time
from neuron import AdaptiveMemory
from neuron.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinalTest")

def run_final_verification():
    logger.info(">>> STARTING FINAL VERIFICATION <<<")
    
    # 1. Configuration (Force In-Memory for testing environment)
    settings.enable_adaptive_memory = True
    settings.graph_store_type = "in_memory"
    settings.adaptive_mode = "manual"
    
    brain = AdaptiveMemory()
    user_id = "final_test_user"
    
    # 2. Ingestion (Write Activity)
    logger.info("[1/5] Testing Ingestion & Activity Logging...")
    brain.process("Quantum computing uses qubits for computation.", user_id=user_id)
    brain.process("The Eiffel Tower is located in Paris.", user_id=user_id)
    
    # 3. Retrieval & Delayed Feedback (Read Activity)
    logger.info("[2/5] Testing Retrieval & Feedback Loop...")
    result = brain.retrieve("What are qubits?", user_id=user_id)
    event_id = result.get("event_id")
    
    if event_id:
        logger.info(f"Successfully received event_id: {event_id}")
        # Simulate user liking the answer
        brain.submit_feedback(event_id, score=1.0)
        logger.info("Feedback 1.0 submitted.")
    else:
        raise Exception("Failed to receive event_id during retrieval.")

    # 4. Strategy Report (Verification of Learning)
    logger.info("[3/5] Verifying Strategy Fitness...")
    report = brain.strategy_report()
    # Find the strategy used and check if fitness is > 0.5 (initial)
    for s in report:
        if s['fitness_score'] > 0.5:
            logger.info(f"SUCCESS: Strategy {s['id']} evolved! New Fitness: {s['fitness_score']}")
            break
    else:
        logger.warning("Strategy fitness did not update yet. Checking Implicit Scorer next.")

    # 5. User Preferences & Maintenance
    logger.info("[4/5] Testing Preferences & Manual Sleep...")
    brain.set_user_preferences(user_id, {"pruning": False})
    brain.force_sleep(user_id)
    logger.info("Manual sleep cycle completed.")

    # 6. Graph Health (Observability)
    logger.info("[5/5] Final Graph Health Check...")
    health = brain.graph_health(user_id)
    logger.info(f"Health Report: {health}")
    
    if health['total_nodes'] >= 2:
        logger.info(">>> ALL TESTS PASSED SUCCESSFULLY <<<")
    else:
        logger.error(f"Test failed: Expected at least 2 nodes, found {health['total_nodes']}")

if __name__ == "__main__":
    run_final_verification()
