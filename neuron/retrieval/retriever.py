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

    def retrieve(self, query: str, user_id: str) -> Dict:
        # 1. Embed query
        query_embedding = self.embedder.embed(query)
        
        # 2. Find seed nodes
        seeds = self.store.search_nearest_nodes(
            query_embedding, 
            user_id, 
            k=settings.k_seeds
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
        seen_node_ids: Set[UUID] = {n.id for n in seeds}
        all_nodes: List[Node] = list(seeds)
        all_edges: List[Edge] = []
        
        # Optimization: Use native single-trip traversal if the store supports it (Fast on Cloud)
        if hasattr(self.store, 'traverse_graph'):
            related_nodes = self.store.traverse_graph(
                start_node_ids=list(seen_node_ids),
                depth=settings.traversal_depth,
                user_id=user_id
            )
            for node in related_nodes:
                if node.id not in seen_node_ids:
                    seen_node_ids.add(node.id)
                    all_nodes.append(node)
                    if len(all_nodes) >= settings.max_context_nodes:
                        break
        else:
            # Fallback to iterative layer-by-layer traversal (Slow on Cloud)
            current_layer = list(seeds)
            for hop in range(settings.traversal_depth):
                if len(all_nodes) >= settings.max_context_nodes:
                    break
                    
                layer_ids = [n.id for n in current_layer]
                edges_map = self.store.get_edges_batch(layer_ids)
                
                next_layer_candidates = set()
                for node_id, edges in edges_map.items():
                    for edge in edges:
                        if edge.weight < settings.min_edge_weight:
                            continue
                        all_edges.append(edge)
                        self._myelinate_edge(edge)
                        target_id = edge.to_node_id if edge.from_node_id == node_id else edge.from_node_id
                        if target_id not in seen_node_ids:
                            next_layer_candidates.add(target_id)
                
                if not next_layer_candidates: break
                
                next_layer = self.store.get_nodes_batch(list(next_layer_candidates))
                filtered_layer = []
                for target_node in next_layer:
                    if not target_node.deprecated:
                        seen_node_ids.add(target_node.id)
                        all_nodes.append(target_node)
                        filtered_layer.append(target_node)
                        if len(all_nodes) >= settings.max_context_nodes: break
                
                if not filtered_layer: break
                current_layer = filtered_layer
        
        # 4. Finalize Myelination (Batch Update to Store)
        if all_edges:
            self.store.update_edges_batch(all_edges)
        
        # 5. Filter by confidence and format
        direct_beliefs = [
            {"label": n.label, "confidence": n.confidence, "evidence": n.evidence_count}
            for n in seeds if n.confidence >= settings.min_confidence
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
            "related_context": [
                {"label": n.label, "confidence": n.confidence}
                for n in all_nodes if n.id not in {s.id for s in seeds}
            ]
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
