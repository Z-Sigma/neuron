from typing import List, Optional, Dict
from uuid import UUID
import numpy as np
from neuron.models import Node, Edge
from neuron.graph.store_interface import GraphStore

class InMemoryGraphStore(GraphStore):
    def __init__(self):
        self.nodes: Dict[UUID, Node] = {}
        self.edges: List[Edge] = []

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
        return []
        
    def get_contradiction_pairs(self, user_id: str) -> List[tuple]:
        return []
        
    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        return []

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
