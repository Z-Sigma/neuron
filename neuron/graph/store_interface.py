from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from uuid import UUID
from neuron.models import Node, Edge, ActivityLog, RetrievalEvent, RetrievalStrategy

class GraphStore(ABC):
    @abstractmethod
    def setup(self) -> None:
        """Initialize the database schema, indexes, and constraints."""
        pass

    @abstractmethod
    def transaction(self):
        """Context manager for atomicity."""
        pass

    @abstractmethod
    def add_node(self, node: Node) -> Node:
        pass

    @abstractmethod
    def add_nodes_batch(self, nodes: List[Node]) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: UUID) -> Optional[Node]:
        pass

    @abstractmethod
    def get_nodes_batch(self, node_ids: List[UUID]) -> List[Node]:
        pass

    @abstractmethod
    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        pass

    @abstractmethod
    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        pass

    @abstractmethod
    def add_edge(self, edge: Edge) -> Edge:
        pass

    @abstractmethod
    def add_edges_batch(self, edges: List[Edge]) -> None:
        pass

    @abstractmethod
    def get_edges(self, node_id: UUID) -> List[Edge]:
        pass

    @abstractmethod
    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        pass

    @abstractmethod
    def update_node(self, node: Node) -> Node:
        pass

    @abstractmethod
    def update_edge(self, edge: Edge) -> Edge:
        pass

    @abstractmethod
    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        pass

    @abstractmethod
    def delete_edge(self, edge_id: UUID) -> None:
        pass

    @abstractmethod
    def log_activity(self, activity: ActivityLog) -> None:
        pass

    @abstractmethod
    def log_retrieval_event(self, event: RetrievalEvent) -> None:
        pass

    @abstractmethod
    def add_strategy(self, strategy: RetrievalStrategy) -> None:
        pass

    @abstractmethod
    def get_strategies(self, user_id: Optional[str] = None) -> List[RetrievalStrategy]:
        pass

    @abstractmethod
    def update_strategy(self, strategy: RetrievalStrategy) -> None:
        pass

    @abstractmethod
    def delete_strategy(self, strategy_id: str) -> None:
        pass

    @abstractmethod
    def get_users_needing_maintenance(self, window_hours: int = 24) -> List[str]:
        pass

    @abstractmethod
    def get_retrieval_events(self, user_id: Optional[str], limit: int = 100) -> List[RetrievalEvent]:
        pass

    @abstractmethod
    def get_retrieval_event(self, event_id: UUID) -> Optional[RetrievalEvent]:
        pass

    @abstractmethod
    def get_contradiction_pairs(self, user_id: str) -> List[Tuple[Node, Node, UUID]]:
        pass

    @abstractmethod
    def get_stale_nodes(self, user_id: str, days: int = 30) -> List[Node]:
        pass

    @abstractmethod
    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        pass

    @abstractmethod
    def traverse_graph(self, start_node_ids: List[UUID], depth: int, user_id: str) -> Tuple[List[Node], List[Edge]]:
        """Optional optimized traversal. Returns nodes and edges."""
        pass

    @abstractmethod
    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        pass

    @abstractmethod
    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        pass
