from typing import List, Optional, Dict
from uuid import UUID
import numpy as np
from neuron.models import Node, Edge, ActivityLog, RetrievalEvent, RetrievalStrategy
from neuron.graph.store_interface import GraphStore
from neuron.config import settings

class InMemoryGraphStore(GraphStore):
    def __init__(self):
        self.nodes: Dict[UUID, Node] = {}
        self.edges: List[Edge] = []
        self.activity_log: List[ActivityLog] = []
        self.retrieval_events: List[RetrievalEvent] = []
        self.strategies: Dict[str, RetrievalStrategy] = {}
        self._ensure_default_strategy()

    def _ensure_default_strategy(self):
        if not self.strategies:
            default = RetrievalStrategy(
                id="default_v1",
                k_seeds=settings.k_seeds,
                traversal_depth=settings.traversal_depth,
                min_edge_weight=settings.min_edge_weight
            )
            self.add_strategy(default)

    def setup(self) -> None:
        pass

    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        for node in nodes:
            self.nodes[node.id] = node

    def get_node(self, node_id: UUID) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_nodes_batch(self, node_ids: List[UUID]) -> List[Node]:
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        results = []
        user_nodes = [n for n in self.nodes.values() if n.user_id == user_id and not n.deprecated]
        
        if not user_nodes:
            return []

        # Simple cosine similarity search
        similarities = []
        for node in user_nodes:
            if node.embedding:
                # Assuming embeddings are normalized
                sim = np.dot(embedding, node.embedding)
                similarities.append((node, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similarities[:k]]

    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        results = []
        user_nodes = [n for n in self.nodes.values() if n.user_id == user_id and n.deprecated]
        
        if not user_nodes:
            return []

        similarities = []
        for node in user_nodes:
            if node.embedding:
                sim = np.dot(embedding, node.embedding)
                similarities.append((node, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similarities[:k]]

    def add_edge(self, edge: Edge) -> Edge:
        self.edges.append(edge)
        return edge

    def add_edges_batch(self, edges: List[Edge]) -> None:
        self.edges.extend(edges)

    def get_edges(self, node_id: UUID) -> List[Edge]:
        return [e for e in self.edges if e.from_node_id == node_id or e.to_node_id == node_id]

    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        ids_set = set(node_ids)
        result = {nid: [] for nid in node_ids}
        for e in self.edges:
            if e.from_node_id in ids_set:
                result[e.from_node_id].append(e)
            if e.to_node_id in ids_set:
                result[e.to_node_id].append(e)
        return result

    def update_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node
        
    # Maintenance stubs for in-memory
    def get_stale_nodes(self, user_id: str, days: int = 30, conf_threshold: float = 0.35) -> List[Node]:
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            n for n in self.nodes.values()
            if n.user_id == user_id
            and n.last_confirmed_at < cutoff
            and n.confidence < conf_threshold
            and not n.deprecated
        ]
        
    def get_contradiction_pairs(self, user_id: str) -> List[tuple]:
        results = []
        for edge in self.edges:
            if edge.relation == "contradicts":
                node_a = self.nodes.get(edge.from_node_id)
                node_b = self.nodes.get(edge.to_node_id)
                if node_a and node_b and node_a.user_id == user_id:
                    results.append((node_a, node_b, edge.id))
        return results
        
    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        return [
            n for n in self.nodes.values()
            if n.user_id == user_id
            and n.evidence_count >= min_evidence
            and n.confidence < 0.95
            and not n.deprecated
        ]

    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        user_nodes = [n for n in self.nodes.values() if n.user_id == user_id]
        return user_nodes[:limit]

    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        return [
            n for n in self.nodes.values() 
            if n.user_id == user_id 
            and not n.deprecated 
            and entity in n.entities
        ]

    def update_edge(self, edge: Edge) -> Edge:
        for i, e in enumerate(self.edges):
            if e.id == edge.id:
                self.edges[i] = edge
                return edge
        return edge

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        for edge in edges:
            self.update_edge(edge)
        return edges

    # --- Adaptive Memory Implementation ---

    def log_activity(self, activity: ActivityLog) -> None:
        self.activity_log.append(activity)

    def log_retrieval_event(self, event: RetrievalEvent) -> None:
        self.retrieval_events.append(event)

    def add_strategy(self, strategy: RetrievalStrategy) -> None:
        self.strategies[strategy.id] = strategy

    def get_strategies(self, user_id: Optional[str] = None) -> List[RetrievalStrategy]:
        if user_id:
            return [s for s in self.strategies.values() if s.user_id == user_id or s.user_id is None]
        return [s for s in self.strategies.values() if s.user_id is None]

    def update_strategy(self, strategy: RetrievalStrategy) -> None:
        if strategy.id in self.strategies:
            self.strategies[strategy.id] = strategy

    def get_users_needing_maintenance(self, window_hours: int = 24) -> List[str]:
        # Simple implementation for in-memory
        users = set()
        for activity in self.activity_log:
            users.add(activity.user_id)
        for event in self.retrieval_events:
            users.add(event.user_id)
        return list(users)

    def get_retrieval_events(self, user_id: Optional[str], limit: int = 100) -> List[RetrievalEvent]:
        if user_id:
            user_events = [e for e in self.retrieval_events if e.user_id == user_id]
        else:
            user_events = self.retrieval_events
        return sorted(user_events, key=lambda x: x.created_at, reverse=True)[:limit]

    def get_retrieval_event(self, event_id: UUID) -> Optional[RetrievalEvent]:
        for e in self.retrieval_events:
            if e.id == event_id:
                return e
        return None
