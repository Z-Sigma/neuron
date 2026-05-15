from typing import List, Set, Dict, Optional
from uuid import UUID
import logging
import numpy as np
from neuron.models import Node, Edge
from neuron.filter.embedder import Embedder
from neuron.graph.store_interface import GraphStore
from neuron.config import settings

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, store: GraphStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, user_id: str, **kwargs) -> Dict:
        # 1. Embed query
        query_embedding = self.embedder.embed(query)
        
        # Strategy overrides (kwargs take precedence over global settings)
        k_seeds = kwargs.get("k_seeds", settings.k_seeds)
        traversal_depth = kwargs.get("traversal_depth", settings.traversal_depth)
        min_edge_weight = kwargs.get("min_edge_weight", settings.min_edge_weight)
        max_context_nodes = kwargs.get("max_context_nodes", settings.max_context_nodes)
        min_confidence = kwargs.get("min_confidence", settings.min_confidence)

        # 2. Find seed nodes
        seeds = self.store.search_nearest_nodes(
            query_embedding, 
            user_id, 
            k=k_seeds
        )

        # 2b. Deep Recall Trigger: 
        # If no seeds or if the top seed is too weak, search archives
        RECALL_THRESHOLD = 0.6
        needs_deep_recall = not seeds
        
        if seeds:
            # Calculate similarity of the top seed
            top_node = seeds[0]
            if top_node.embedding:
                similarity = np.dot(query_embedding, top_node.embedding)
                if similarity < RECALL_THRESHOLD:
                    needs_deep_recall = True
                    logger.debug(f"Top active match is weak ({similarity:.2f}). Checking archives...")

        if needs_deep_recall:
            archived_seeds = self.store.search_deprecated_nodes(
                query_embedding,
                user_id,
                k=1
            )
            if archived_seeds:
                archived = archived_seeds[0]
                # Only reactivate if the archived match is actually strong
                archived_sim = np.dot(query_embedding, archived.embedding) if archived.embedding else 0
                if archived_sim > RECALL_THRESHOLD:
                    logger.info(f"Deep Recall found strong match in archives ({archived_sim:.2f}): '{archived.label}'")
                    archived.deprecated = False
                    self.store.update_node(archived)
                    # Prioritize the reactivated node
                    seeds = [archived] + [s for s in seeds if s.id != archived.id]
        
        # 3. Graph Traversal (Context Discovery)
        all_nodes: List[Node] = list(seeds)
        all_edges = []
        seen_node_ids = {n.id for n in seeds}
        
        # Unified Traversal (Now mandatory for all stores)
        related_nodes, discovered_edges = self.store.traverse_graph(
            start_node_ids=list(seen_node_ids),
            depth=traversal_depth,
            user_id=user_id
        )
        
        for node in related_nodes:
            if node.id not in seen_node_ids:
                seen_node_ids.add(node.id)
                all_nodes.append(node)
                if len(all_nodes) >= max_context_nodes:
                    break
        
        # 3. Neural Path Strengthening (Myelination)
        # We myelinate all discovered edges exactly once
        for edge in discovered_edges:
            if edge.weight >= min_edge_weight:
                all_edges.append(edge)
                self._myelinate_edge(edge)
        
        # Bulk save myelinated paths
        if all_edges:
            self.store.update_edges_batch(all_edges)
        
        # 5. Filter by confidence and format
        direct_beliefs = [
            {"label": n.label, "confidence": n.confidence, "evidence": n.evidence_count}
            for n in seeds if n.confidence >= min_confidence
        ]
        
        unresolved_tensions = []
        for edge in all_edges:
            if edge.relation == "contradicts" or edge.relation == "unresolved_tension":
                node_a = next((n for n in all_nodes if n.id == edge.from_node_id), None)
                node_b = next((n for n in all_nodes if n.id == edge.to_node_id), None)
                if node_a and node_b:
                    unresolved_tensions.append({
                        "belief_a": node_a.label,
                        "belief_b": node_b.label,
                        "relation": edge.relation
                    })

        return {
            "direct_beliefs": direct_beliefs,
            "unresolved_tensions": unresolved_tensions,
            "related_context": sorted([
                {"label": n.label, "confidence": n.confidence, "evidence": n.evidence_count}
                for n in all_nodes if n.id not in {s.id for s in seeds}
            ], key=lambda x: (x['confidence'], x['evidence']), reverse=True)
        }

    def _myelinate_edge(self, edge: Edge):
        """
        Inspired by brain myelination: Strengthen active neural paths.
        Increases weight and evidence count of a relationship when used.
        """
        edge.weight = min(edge.weight * 1.05, 1.0)
        edge.evidence_count += 1
        # We no longer call store.update_edge(edge) here to avoid N+1 writes.
        # The edges are updated in-memory and will be saved in batch later.
