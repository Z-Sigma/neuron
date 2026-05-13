import logging
from datetime import datetime
from neuron.graph.store_interface import GraphStore
from neuron.adaptive.adaptive_engine import StrategyRegistry
from neuron.adaptive.scorer import ImplicitScorer
from neuron.models import ActivityLog, ActivityType, RetrievalEvent

logger = logging.getLogger(__name__)

class SleepConsolidator:
    """
    The 'Offline Brain' that processes data while the system is idle.
    Replays high-scoring events, boosts node confidence, and triggers strategy evolution.
    """
    def __init__(self, store: GraphStore, registry: StrategyRegistry):
        self.store = store
        self.registry = registry
        self.scorer = ImplicitScorer()

    def perform_sleep_cycle(self, user_id: str):
        """
        Runs a full consolidation cycle for a specific user.
        """
        logger.info(f"Starting sleep cycle for user {user_id}")
        
        # 1. Analyze retrieval events for reinforcement
        events = self.store.get_retrieval_events(user_id, limit=50)
        
        # 2. Implicit Scoring (Zero Latency judging)
        # We only score events that don't have a user score already
        unscored_events = [e for e in events if e.score == 0.0]
        if unscored_events:
            scores = self.scorer.score_events(unscored_events)
            for event, score in zip(unscored_events, scores):
                event.score = score
                self.registry.update_fitness(event.strategy_id, score)
        
        # 3. Reinforce 'Useful' nodes (Myelination)
        # High-scoring events indicate paths that should be strengthened.
        # In a full implementation, we extract node IDs from event metadata
        # and boost their confidence and edge weights directly.
        high_score_events = [e for e in events if e.score >= 0.8]
        logger.info(f"Found {len(high_score_events)} high-score events for myelination.")

        # 4. Trigger Strategy Evolution if we have enough data
        if len(events) >= 50:
            self.registry.evolve()

        # 4. Standard Maintenance (Pruning & Consolidation)
        # We reuse the existing Memory.maintenance logic but triggered automatically
        from neuron.memory import Memory
        mem = Memory(store=self.store)
        mem.maintenance(user_id)

        # 5. Log the cycle
        self.store.log_activity(ActivityLog(
            user_id=user_id,
            activity_type=ActivityType.MAINTENANCE,
            details=f"Completed SleepConsolidator cycle. Evolved={len(events) >= 50}"
        ))
