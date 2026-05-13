from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from uuid import UUID
from neuron.models import Node, Edge

class GraphStore(ABC):
    @abstractmethod
    def setup(self) -> None:
        """Initialize the database schema, indexes, and constraints."""
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
    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        pass
        
    @abstractmethod
    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        pass

    @abstractmethod
    def update_edge(self, edge: Edge) -> Edge:
        pass

    @abstractmethod
    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        pass
