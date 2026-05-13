import psycopg2
from psycopg2.extras import execute_values
from typing import List, Optional, Dict
from uuid import UUID
import json
import logging
from neuron.models import Node, Edge
from neuron.graph.store_interface import GraphStore
from neuron.config import settings

logger = logging.getLogger(__name__)

class PostgresGraphStore(GraphStore):
    def __init__(self, connection_url: str = settings.database_url):
        self.conn = psycopg2.connect(connection_url)
        self.conn.autocommit = True
        self.setup()

    def setup(self) -> None:
        with self.conn.cursor() as cur:
            # 1. Extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # 2. Nodes Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence FLOAT NOT NULL,
                    evidence_count INTEGER DEFAULT 1,
                    contradiction_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_confirmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_contradicted_at TIMESTAMP,
                    domain_tags TEXT[],
                    entities TEXT[],
                    temporal_stability TEXT,
                    abstraction_level TEXT,
                    embedding vector({settings.embedding_dimension}),
                    deprecated BOOLEAN DEFAULT FALSE
                )
            """.format(settings=settings))
            
            # 3. Edges Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id UUID PRIMARY KEY,
                    from_node_id UUID REFERENCES nodes(id),
                    to_node_id UUID REFERENCES nodes(id),
                    relation TEXT NOT NULL,
                    weight FLOAT NOT NULL,
                    evidence_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 4. Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_user_id ON nodes(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_deprecated ON nodes(deprecated)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_embedding ON nodes USING hnsw (embedding vector_cosine_ops)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_nodes ON edges(from_node_id, to_node_id)")

    def add_node(self, node: Node) -> Node:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO nodes (
                    id, user_id, label, confidence, evidence_count, contradiction_count,
                    created_at, last_confirmed_at, domain_tags, entities,
                    temporal_stability, abstraction_level, embedding, deprecated
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(node.id), node.user_id, node.label, node.confidence,
                node.evidence_count, node.contradiction_count, node.created_at,
                node.last_confirmed_at, node.domain_tags, node.entities,
                node.temporal_stability, node.abstraction_level, 
                node.embedding, node.deprecated
            ))
        return node

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        if not nodes:
            return
        query = """
            INSERT INTO nodes (
                id, user_id, label, confidence, evidence_count, contradiction_count,
                created_at, last_confirmed_at, domain_tags, entities,
                temporal_stability, abstraction_level, embedding, deprecated
            ) VALUES %s
        """
        data = [(
            str(node.id), node.user_id, node.label, node.confidence,
            node.evidence_count, node.contradiction_count, node.created_at,
            node.last_confirmed_at, node.domain_tags, node.entities,
            node.temporal_stability, node.abstraction_level, 
            node.embedding, node.deprecated
        ) for node in nodes]
        with self.conn.cursor() as cur:
            execute_values(cur, query, data)

    def get_node(self, node_id: UUID) -> Optional[Node]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM nodes WHERE id = %s", (str(node_id),))
            row = cur.fetchone()
            if row:
                return self._row_to_node(row)
        return None

    def get_nodes_batch(self, node_ids: List[UUID]) -> List[Node]:
        if not node_ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM nodes WHERE id IN %s", (tuple(str(nid) for nid in node_ids),))
            rows = cur.fetchall()
            return [self._row_to_node(row) for row in rows]

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM nodes 
                WHERE user_id = %s AND deprecated = FALSE
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
            """, (user_id, embedding, k))
            rows = cur.fetchall()
            return [self._row_to_node(row) for row in rows]

    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM nodes 
                WHERE user_id = %s AND deprecated = TRUE
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
            """, (user_id, embedding, k))
            rows = cur.fetchall()
            return [self._row_to_node(row) for row in rows]

    def add_edge(self, edge: Edge) -> Edge:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO edges (
                    id, from_node_id, to_node_id, relation, weight,
                    evidence_count, created_at, last_updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(edge.id), str(edge.from_node_id), str(edge.to_node_id),
                edge.relation, edge.weight, edge.evidence_count,
                edge.created_at, edge.last_updated_at
            ))
        return edge

    def add_edges_batch(self, edges: List[Edge]) -> None:
        if not edges:
            return
        query = """
            INSERT INTO edges (
                id, from_node_id, to_node_id, relation, weight,
                evidence_count, created_at, last_updated_at
            ) VALUES %s
        """
        data = [(
            str(edge.id), str(edge.from_node_id), str(edge.to_node_id),
            edge.relation, edge.weight, edge.evidence_count,
            edge.created_at, edge.last_updated_at
        ) for edge in edges]
        with self.conn.cursor() as cur:
            execute_values(cur, query, data)

    def get_edges(self, node_id: UUID) -> List[Edge]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM edges WHERE from_node_id = %s OR to_node_id = %s", (str(node_id), str(node_id)))
            rows = cur.fetchall()
            return [self._row_to_edge(row) for row in rows]

    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        if not node_ids:
            return {}
        ids_str = tuple(str(nid) for nid in node_ids)
        result = {nid: [] for nid in node_ids}
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM edges WHERE from_node_id IN %s OR to_node_id IN %s", (ids_str, ids_str))
            rows = cur.fetchall()
            for row in rows:
                edge = self._row_to_edge(row)
                if edge.from_node_id in result:
                    result[edge.from_node_id].append(edge)
                if edge.to_node_id in result:
                    result[edge.to_node_id].append(edge)
        return result

    def update_node(self, node: Node) -> Node:
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE nodes SET
                    label = %s, confidence = %s, evidence_count = %s,
                    contradiction_count = %s, last_confirmed_at = %s,
                    last_contradicted_at = %s, domain_tags = %s, entities = %s,
                    temporal_stability = %s, abstraction_level = %s,
                    deprecated = %s
                WHERE id = %s
            """, (
                node.label, node.confidence, node.evidence_count,
                node.contradiction_count, node.last_confirmed_at,
                node.last_contradicted_at, node.domain_tags, node.entities,
                node.temporal_stability, node.abstraction_level,
                node.deprecated, str(node.id)
            ))
        return node

    def get_stale_nodes(self, user_id: str, days: int = 30, conf_threshold: float = 0.35) -> List[Node]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM nodes 
                WHERE user_id = %s 
                AND last_confirmed_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND confidence < %s
                AND deprecated = FALSE
            """, (user_id, days, conf_threshold))
            rows = cur.fetchall()
            return [self._row_to_node(row) for row in rows]

    def get_contradiction_pairs(self, user_id: str) -> List[tuple]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT n1.id, n1.label, n1.confidence, n1.evidence_count,
                       n2.id, n2.label, n2.confidence, n2.evidence_count,
                       e.id as edge_id
                FROM nodes n1
                JOIN edges e ON n1.id = e.from_node_id
                JOIN nodes n2 ON e.to_node_id = n2.id
                WHERE n1.user_id = %s AND e.relation = 'contradicts'
                AND n1.deprecated = FALSE AND n2.deprecated = FALSE
            """, (user_id,))
            rows = cur.fetchall()
            results = []
            for row in rows:
                node_a = Node(id=UUID(row[0]), user_id=user_id, label=row[1], confidence=row[2], evidence_count=row[3])
                node_b = Node(id=UUID(row[4]), user_id=user_id, label=row[5], confidence=row[6], evidence_count=row[7])
                results.append((node_a, node_b, UUID(row[8])))
            return results

    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM nodes 
                WHERE user_id = %s 
                AND evidence_count >= %s
                AND confidence < 0.95
                AND deprecated = FALSE
            """, (user_id, min_evidence))
            rows = cur.fetchall()
            return [self._row_to_node(row) for row in rows]

    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM nodes 
                WHERE user_id = %s 
                AND deprecated = FALSE
                LIMIT %s
            """, (user_id, limit))
            rows = cur.fetchall()
            return [self._row_to_node(row) for row in rows]

    def update_edge(self, edge: Edge) -> Edge:
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE edges SET
                    weight = %s, evidence_count = %s, last_updated_at = %s
                WHERE id = %s
            """, (edge.weight, edge.evidence_count, edge.last_updated_at, str(edge.id)))
        return edge

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        if not edges:
            return []
        data = [(edge.weight, edge.evidence_count, edge.last_updated_at, str(edge.id)) for edge in edges]
        with self.conn.cursor() as cur:
            execute_values(cur, """
                UPDATE edges AS e SET
                    weight = data.weight,
                    evidence_count = data.evidence_count,
                    last_updated_at = data.last_updated_at
                FROM (VALUES %s) AS data(weight, evidence_count, last_updated_at, id)
                WHERE e.id = data.id::uuid
            """, data)
        return edges

    def _row_to_node(self, row) -> Node:
        # Column order matches schema.sql:
        # id, user_id, label, confidence, evidence_count, contradiction_count,
        # created_at, last_confirmed_at, last_contradicted_at, domain_tags,
        # entities, temporal_stability, abstraction_level, embedding, deprecated
        return Node(
            id=UUID(row[0]),
            user_id=row[1],
            label=row[2],
            confidence=row[3],
            evidence_count=row[4],
            contradiction_count=row[5],
            created_at=row[6],
            last_confirmed_at=row[7],
            last_contradicted_at=row[8],
            domain_tags=row[9] or [],
            entities=row[10] or [],
            temporal_stability=row[11],
            abstraction_level=row[12],
            embedding=json.loads(row[13]) if isinstance(row[13], str) else row[13],
            deprecated=row[14]
        )

    def _row_to_edge(self, row) -> Edge:
        return Edge(
            id=UUID(row[0]),
            from_node_id=UUID(row[1]),
            to_node_id=UUID(row[2]),
            relation=row[3],
            weight=row[4],
            evidence_count=row[5],
            created_at=row[6],
            last_updated_at=row[7]
        )
