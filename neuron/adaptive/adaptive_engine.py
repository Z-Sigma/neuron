import random
import logging
from typing import List, Optional
from neuron.models import RetrievalStrategy, RetrievalEvent
from neuron.graph.store_interface import GraphStore
from neuron.config import settings

logger = logging.getLogger(__name__)

class StrategyRegistry:
    """
    Manages the population of retrieval strategies and handles selection/evolution.
    Implements ε-greedy selection and mutation-based evolution (ShinkaEvolve style).
    """
    def __init__(self, store: GraphStore):
        self.store = store
        self.population_size = settings.strategy_population_size
        self.exploration_rate = settings.exploration_rate

    def select_strategy(self, user_id: str) -> RetrievalStrategy:
        """
        Selects a strategy using ε-greedy selection for a specific user.
        """
        self._ensure_user_population(user_id)
        strategies = [s for s in self.store.get_strategies(user_id) if s.user_id == user_id]
        
        if not strategies:
            return RetrievalStrategy(id="default", user_id=user_id, k_seeds=5, traversal_depth=3, novelty_threshold=0.8)

        if random.random() < self.exploration_rate:
            return random.choice(strategies)
        else:
            return max(strategies, key=lambda s: s.fitness_score)

    def _ensure_user_population(self, user_id: str):
        """Ensures a user has a starting population of strategies."""
        all_strats = self.store.get_strategies(user_id)
        user_owned = [s for s in all_strats if s.user_id == user_id]
        
        if not user_owned:
            logger.info(f"Seeding strategy population for new user: {user_id}")
            system_defaults = [s for s in all_strats if s.user_id is None]
            if not system_defaults:
                system_defaults = [RetrievalStrategy(id="sys_default", k_seeds=settings.k_seeds, traversal_depth=settings.traversal_depth)]
            
            for template in system_defaults:
                child = template.model_copy(update={
                    "id": f"u_{user_id[:6]}_{template.id}",
                    "user_id": user_id,
                    "fitness_score": 0.5,
                    "generations_survived": 0
                })
                self.store.add_strategy(child)

    def update_fitness(self, strategy_id: str, score: float, user_id: str):
        """Updates the rolling fitness score for a strategy."""
        strategies = self.store.get_strategies(user_id)
        strategy = next((s for s in strategies if s.id == strategy_id and s.user_id == user_id), None)
        if strategy:
            alpha = 0.3
            strategy.fitness_score = (1 - alpha) * strategy.fitness_score + alpha * score
            self.store.update_strategy(strategy)

    def evolve(self, user_id: str):
        """Performs evolution cycle for a specific user."""
        strategies = [s for s in self.store.get_strategies(user_id) if s.user_id == user_id]
        if len(strategies) < 2:
            return

        strategies.sort(key=lambda s: s.fitness_score, reverse=True)
        
        best = strategies[0]
        worst = strategies[-1]

        if best.fitness_score > worst.fitness_score + 0.05:
            logger.info(f"Evolving strategies for {user_id}: Replacing {worst.id} with child of {best.id}")
            
            child_id = f"mutant_{user_id[:4]}_{best.id}_{best.generations_survived + 1}"
            child = RetrievalStrategy(
                id=child_id,
                user_id=user_id,
                k_seeds=max(1, best.k_seeds + random.randint(-1, 1)),
                traversal_depth=max(1, best.traversal_depth + random.randint(-1, 1)),
                min_edge_weight=max(0.1, min(1.0, best.min_edge_weight + random.uniform(-0.1, 0.1))),
                novelty_threshold=max(0.1, min(1.0, best.novelty_threshold + random.uniform(-0.1, 0.1))),
                parent_id=best.id,
                generations_survived=0
            )
            
            best.generations_survived += 1
            self.store.update_strategy(best)
            self.store.add_strategy(child)
            
            # Prune population if it exceeds the limit
            if len(strategies) >= self.population_size:
                logger.info(f"Pruning population for {user_id}: Removing least-fit strategy {worst.id}")
                self.store.delete_strategy(worst.id)
