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

    def select_strategy(self) -> RetrievalStrategy:
        """
        Selects a strategy using ε-greedy selection.
        80% chance to pick the best performing, 20% to explore.
        """
        strategies = self.store.get_strategies()
        if not strategies:
            # Fallback to default if somehow empty
            return RetrievalStrategy(id="default", k_seeds=5, traversal_depth=3)

        if random.random() < self.exploration_rate:
            # Explore: Random pick
            return random.choice(strategies)
        else:
            # Exploit: Best fitness
            return max(strategies, key=lambda s: s.fitness_score)

    def update_fitness(self, strategy_id: str, score: float):
        """Updates the rolling fitness score for a strategy."""
        strategies = self.store.get_strategies()
        strategy = next((s for s in strategies if s.id == strategy_id), None)
        if strategy:
            # Simple moving average for fitness
            alpha = 0.3
            strategy.fitness_score = (1 - alpha) * strategy.fitness_score + alpha * score
            self.store.update_strategy(strategy)

    def evolve(self):
        """
        Performs evolution cycle.
        Replaces the weakest strategy with a mutated child of the best strategy.
        """
        strategies = self.store.get_strategies()
        if len(strategies) < 2:
            return

        # Sort by fitness
        strategies.sort(key=lambda s: s.fitness_score, reverse=True)
        
        best = strategies[0]
        worst = strategies[-1]

        # Only evolve if we have enough data and a clear loser
        if best.fitness_score > worst.fitness_score + 0.1:
            logger.info(f"Evolving strategies: Replacing {worst.id} with child of {best.id}")
            
            # Create mutated child
            child_id = f"mutant_{best.id}_{best.generations_survived + 1}"
            child = RetrievalStrategy(
                id=child_id,
                k_seeds=max(1, best.k_seeds + random.randint(-1, 1)),
                traversal_depth=max(1, best.traversal_depth + random.randint(-1, 1)),
                min_edge_weight=max(0.1, min(1.0, best.min_edge_weight + random.uniform(-0.1, 0.1))),
                parent_id=best.id,
                generations_survived=0
            )
            
            # Update best's age
            best.generations_survived += 1
            self.store.update_strategy(best)
            
            # Replace worst in DB
            # Note: In a real implementation we might delete the worst, but here we just add the new one
            # and maintenance will prune strategies if population > limit.
            self.store.add_strategy(child)
            # self.store.delete_strategy(worst.id) # Interface doesn't have this yet, assume population management
