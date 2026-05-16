import logging
from typing import List, Dict, Optional
from neuron.memory import Memory
from neuron.models import ActivityLog, ActivityType, RetrievalEvent
from neuron.adaptive.adaptive_engine import StrategyRegistry
from neuron.adaptive.consolidator import SleepConsolidator
from neuron.config import settings

logger = logging.getLogger(__name__)

class AdaptiveMemory(Memory):
    """
    Advanced Memory engine with adaptive strategies and automated maintenance.
    API-identical to raw Memory, but smarter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = StrategyRegistry(self.store)
        self.consolidator = SleepConsolidator(self.store, self.registry)
        self.user_prefs: Dict[str, Dict] = {} # In-memory cache for demo
        
        # Phase 3: Start background scheduler if auto-mode
        if settings.enable_adaptive_memory and settings.adaptive_mode == "auto":
            from neuron.daemon.scheduler import SleepScheduler
            self.scheduler = SleepScheduler(self)
            self.scheduler.start()

    def process(self, text: str, user_id: str):
        # 1. Select Strategy (Adaptive Ingestion)
        novelty_threshold = None
        if settings.enable_adaptive_memory:
            strategy = self.registry.select_strategy(user_id)
            novelty_threshold = strategy.novelty_threshold

        # 2. Process with adaptive threshold
        node = super().process(text, user_id, novelty_threshold=novelty_threshold)
        
        if settings.enable_adaptive_memory and node:
            self.store.log_activity(ActivityLog(
                user_id=user_id,
                activity_type=ActivityType.WRITE,
                details=f"Stored node: {node.id}"
            ))
        return node

    def process_batch(self, texts: list, user_id: str, global_deduplication: bool = False):
        """Alias for process_batch_fast — high-speed ingestion."""
        return self.process_batch_fast(texts, user_id, global_deduplication=global_deduplication)

    def search_archive(self, query: str, user_id: str) -> list:
        """
        Deep Recall — searches deprecated (archived) nodes only.
        """
        embedding = self.embedder.embed(query)
        return self.store.search_deprecated_nodes(embedding, user_id, k=5)

    def retrieve(self, query: str, user_id: str, score_feedback: Optional[float] = None) -> Dict:
        # 1. Strategy Selection (Adaptive)
        strategy_params = {}
        strategy = None
        if settings.enable_adaptive_memory:
            strategy = self.registry.select_strategy(user_id)
            strategy_params = {
                "k_seeds": strategy.k_seeds,
                "traversal_depth": strategy.traversal_depth,
                "min_edge_weight": strategy.min_edge_weight
            }

        # 2. Perform Retrieval
        result = super().retrieve(query, user_id, **strategy_params)

        # 3. Log Event
        if settings.enable_adaptive_memory and strategy:
            event = RetrievalEvent(
                user_id=user_id,
                query=query,
                strategy_id=strategy.id,
                nodes_found=len(result.get("direct_beliefs", []))
                + len(result.get("related_context", [])),
                retrieved_context=[b["label"] for b in result.get("direct_beliefs", [])]
                + [c["label"] for c in result.get("related_context", [])],
                score=score_feedback if score_feedback is not None else 0.5
            )
            self.store.log_retrieval_event(event)
            
            result["event_id"] = str(event.id)
            
            if score_feedback is not None:
                self.registry.update_fitness(strategy.id, score_feedback, user_id)

        return result

    def submit_feedback(self, event_id: str, score: float):
        """
        Submits delayed feedback for a specific retrieval event.
        This is the core of the human-in-the-loop learning.
        """
        # 1. Direct Lookup by ID (Optimized)
        from uuid import UUID
        event = self.store.get_retrieval_event(UUID(event_id))
        
        if event:
            self.registry.update_fitness(event.strategy_id, score, event.user_id)
            logger.info(f"Feedback submitted for event {event_id}: {score}")
        else:
            logger.warning(f"Could not find event {event_id} for feedback submission.")

    def set_user_preferences(self, user_id: str, prefs: Dict):
        """Sets maintenance and retrieval preferences for a user."""
        if user_id not in self.user_prefs:
            self.user_prefs[user_id] = {}
        self.user_prefs[user_id].update(prefs)
        logger.info(f"Updated preferences for user {user_id}: {self.user_prefs[user_id]}")

    def get_user_preferences(self, user_id: str) -> Dict:
        """Returns the user's current preferences, with defaults."""
        return self.user_prefs.get(user_id, {
            "pruning": True,
            "consolidation": True,
            "min_confidence": settings.min_confidence
        })

    def force_sleep(self, user_id: str):
        """Manually trigger a consolidation cycle, respecting user preferences."""
        prefs = self.get_user_preferences(user_id)
        if not prefs.get("pruning") and not prefs.get("consolidation"):
            logger.info(f"Maintenance skipped for {user_id} based on preferences.")
            return
            
        self.consolidator.perform_sleep_cycle(user_id)

    def strategy_report(self, user_id: Optional[str] = None) -> List[Dict]:
        """Returns the current state of evolved strategies."""
        strategies = self.store.get_strategies(user_id)
        return [s.model_dump() for s in strategies]

    def graph_health(self, user_id: str) -> Dict:
        """Returns diagnostic metrics for the user's graph."""
        nodes = self.store.list_nodes(user_id, limit=1000)
        total = len(nodes)
        if total == 0: return {"status": "empty"}
        
        high_conf = len([n for n in nodes if n.confidence >= 0.8])
        deprecated = len([n for n in nodes if n.deprecated])
        
        return {
            "total_nodes": total,
            "confidence_ratio": high_conf / total,
            "prune_ratio": deprecated / total,
            "status": "healthy" if high_conf / total > 0.5 else "noisy"
        }
