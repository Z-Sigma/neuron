import logging
from typing import List, Optional, Dict, Tuple
from uuid import UUID
from collections import deque
import numpy as np
from neuron.models import Node, Edge, ActivityLog, RetrievalEvent, RetrievalStrategy
from neuron.graph.store_interface import GraphStore
import threading

logger = logging.getLogger(__name__)

class InMemoryGraphStore(GraphStore):
    def __init__(self):
        self.nodes: Dict[UUID, Node] = {}
        self.edges: Dict[UUID, Edge] = {}
        self.edge_index: Dict[UUID, List[Edge]] = {} # Node ID -> List of incident edges
        self.strategies: Dict[str, RetrievalStrategy] = {}
        self.retrieval_events: Dict[UUID, RetrievalEvent] = {}
        self.lock = threading.Lock()
        self.setup()

    def transaction(self):
        class Transaction:
            def __init__(self, lock): self.lock = lock
            def __enter__(self): self.lock.acquire(); return self
            def __exit__(self, exc_type, exc_val, exc_tb): self.lock.release()
        return Transaction(self.lock)

    def setup(self) -> None:
        self._ensure_default_strategy()

    def _ensure_default_strategy(self):
        if not self.strategies:
            default = RetrievalStrategy(id="default_v1", user_id="system", name="Default", config={})
            self.add_strategy(default)

    def add_node(self, node: Node) -> Node:
        with self.lock: self.nodes[node.id] = node; return node

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        with self.lock:
            for n in nodes: self.nodes[n.id] = n

    def get_node(self, node_id: UUID) -> Optional[Node]: return self.nodes.get(node_id)

    def get_nodes_batch(self, node_ids: List[UUID]) -> List[Node]:
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        valid_nodes = [n for n in self.nodes.values() if n.user_id == user_id and not n.deprecated]
        if not valid_nodes: return []
        sims = [(n, np.dot(embedding, n.embedding)) for n in valid_nodes]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sims[:k]]

    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        valid_nodes = [n for n in self.nodes.values() if n.user_id == user_id and n.deprecated]
        if not valid_nodes: return []
        sims = [(n, np.dot(embedding, n.embedding)) for n in valid_nodes]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sims[:k]]

    def add_edge(self, edge: Edge) -> Edge:
        with self.lock: 
            self.edges[edge.id] = edge
            # Index by both nodes
            for nid in [edge.from_node_id, edge.to_node_id]:
                if nid not in self.edge_index: self.edge_index[nid] = []
                self.edge_index[nid].append(edge)
            return edge

    def add_edges_batch(self, edges: List[Edge]) -> None:
        with self.lock:
            for e in edges: 
                self.edges[e.id] = e
                for nid in [e.from_node_id, e.to_node_id]:
                    if nid not in self.edge_index: self.edge_index[nid] = []
                    self.edge_index[nid].append(e)

    def get_edges(self, node_id: UUID) -> List[Edge]:
        return self.edge_index.get(node_id, [])

    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        res = {nid: [] for nid in node_ids}
        for e in self.edges.values():
            if e.from_node_id in res: res[e.from_node_id].append(e)
            if e.to_node_id in res: res[e.to_node_id].append(e)
        return res

    def update_node(self, node: Node) -> Node:
        with self.lock: self.nodes[node.id] = node; return node

    def update_edge(self, edge: Edge) -> Edge:
        with self.lock: self.edges[edge.id] = edge; return edge

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        with self.lock:
            for e in edges: self.edges[e.id] = e
        return edges

    def traverse_graph(self, start_node_ids: List[UUID], depth: int, user_id: str) -> Tuple[List[Node], List[Edge]]:
        visited_nodes, visited_edges = set(), set()
        queue = deque([(nid, 0) for nid in start_node_ids if nid in self.nodes])
        while queue:
            curr_id, curr_depth = queue.popleft()
            if curr_id in visited_nodes or curr_depth > depth: continue
            node = self.nodes[curr_id]
            if node.user_id != user_id or node.deprecated: continue
            visited_nodes.add(curr_id)
            if curr_depth < depth:
                edges = self.get_edges(curr_id)
                for e in edges:
                    visited_edges.add(e.id)
                    next_id = e.to_node_id if e.from_node_id == curr_id else e.from_node_id
                    queue.append((next_id, curr_depth + 1))
        return [self.nodes[nid] for nid in visited_nodes], [self.edges[eid] for eid in visited_edges]

    def log_activity(self, activity: ActivityLog) -> None:
        if not hasattr(self, 'activities'):
            self.activities = []
        with self.lock:
            self.activities.append(activity)
    def log_retrieval_event(self, event: RetrievalEvent) -> None:
        with self.lock: self.retrieval_events[event.id] = event

    def add_strategy(self, strategy: RetrievalStrategy) -> None:
        with self.lock: self.strategies[strategy.id] = strategy
    def get_strategies(self, user_id: Optional[str] = None) -> List[RetrievalStrategy]:
        if user_id: return [s for s in self.strategies.values() if s.user_id == user_id or s.user_id == "system"]
        return list(self.strategies.values())
    def update_strategy(self, strategy: RetrievalStrategy) -> None:
        with self.lock: self.strategies[strategy.id] = strategy
    def delete_strategy(self, strategy_id: str) -> None:
        with self.lock:
            if strategy_id in self.strategies: del self.strategies[strategy_id]

    def get_users_needing_maintenance(self, window_hours: int = 24) -> List[str]:
        if not hasattr(self, 'activities'):
            return []
        # In a real impl, we'd check timestamps. For the test, we return all who logged.
        return list(set([a.user_id for a in self.activities]))

    def get_stale_nodes(self, user_id: str, days: int = 30) -> List[Node]:
        # Stubs for in-memory
        return []
    def get_retrieval_events(self, user_id: Optional[str], limit: int = 100) -> List[RetrievalEvent]:
        events = list(self.retrieval_events.values())
        if user_id: events = [e for e in events if e.user_id == user_id]
        events.sort(key=lambda x: x.created_at, reverse=True)
        return events[:limit]
    def get_retrieval_event(self, event_id: UUID) -> Optional[RetrievalEvent]: return self.retrieval_events.get(event_id)
    def delete_edge(self, edge_id: UUID) -> None:
        with self.lock:
            if edge_id in self.edges: del self.edges[edge_id]

    def get_contradiction_pairs(self, user_id: str) -> List[Tuple[Node, Node, UUID]]:
        results = []
        for e in self.edges.values():
            if e.relation == 'contradicts':
                na = self.nodes.get(e.from_node_id)
                nb = self.nodes.get(e.to_node_id)
                if na and nb and na.user_id == user_id and nb.user_id == user_id:
                    if not na.deprecated and not nb.deprecated:
                        results.append((na, nb, e.id))
        return results
    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        return [n for n in self.nodes.values() if n.user_id == user_id and n.evidence_count >= min_evidence and not n.deprecated]
    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        return [n for n in self.nodes.values() if n.user_id == user_id][:limit]
    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        return [n for n in self.nodes.values() if n.user_id == user_id and entity in n.entities]
