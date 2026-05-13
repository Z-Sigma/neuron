import numpy as np
from typing import List, Optional
from neuron.models import SurpriseResult, Node
from neuron.filter.embedder import Embedder
from neuron.graph.store_interface import GraphStore
from neuron.config import settings

class SurpriseFilter:
    def __init__(self, store: GraphStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.threshold = settings.base_novelty_threshold

    def evaluate(self, text: str, user_id: str) -> SurpriseResult:
        # 1. Embed incoming text
        embedding = self.embedder.embed(text)
        
        # 2. Query for K nearest existing nodes
        nearest_nodes = self.store.search_nearest_nodes(embedding, user_id, k=settings.k_seeds)
        
        if not nearest_nodes:
            return SurpriseResult(is_novel=True, novelty_score=1.0)

        # 3. Compute cosine similarity (dot product since embeddings are normalized)
        similarities = []
        for node in nearest_nodes:
            if node.embedding:
                sim = np.dot(embedding, node.embedding)
                similarities.append((node, sim))
        
        if not similarities:
            return SurpriseResult(is_novel=True, novelty_score=1.0)

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        max_sim = similarities[0][1]
        novelty_score = 1.0 - max_sim

        # 4. Decision logic
        is_novel = novelty_score >= self.threshold
        
        confirmation_targets = []
        contradiction_candidates = []
        
        # Logic for confirmation and contradiction
        for node, sim in similarities:
            if settings.confirmation_range_start <= sim <= settings.confirmation_range_end:
                confirmation_targets.append(node.id)
            
            # If similarity is high but not perfect, it might be a contradiction
            if sim > 0.6:
                contradiction_candidates.append(node.id)

        return SurpriseResult(
            is_novel=is_novel,
            novelty_score=novelty_score,
            nearest_node_ids=[n[0].id for n in similarities],
            contradiction_candidates=contradiction_candidates,
            confirmation_targets=confirmation_targets
        )
