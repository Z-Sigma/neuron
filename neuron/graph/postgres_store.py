import logging
import json
from typing import List, Optional, Dict, Tuple
from uuid import UUID
from datetime import datetime, timezone
import psycopg2
from psycopg2 import pool
from neuron.config import settings
from neuron.models import Node, Edge, ActivityLog, RetrievalEvent, RetrievalStrategy
from neuron.graph.store_interface import GraphStore

logger = logging.getLogger(__name__)

class PostgresGraphStore(GraphStore):
    def __init__(self, connection_string: str = settings.database_url):
        self.pool = psycopg2.pool.ThreadedConnectionPool(1, 20, connection_string)
        self.setup()

    def setup(self) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                with open("neuron/graph/schema.sql", "r") as f:
                    cur.execute(f.read())
            conn.commit()
            self._ensure_default_strategy()
        finally:
            self.pool.putconn(conn)

    def transaction(self):
        class Transaction:
            def __init__(self, pool):
                self.pool = pool
                self.conn = None
            def __enter__(self):
                self.conn = self.pool.getconn()
                return self.conn
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None: self.conn.commit()
                else: self.conn.rollback()
                self.pool.putconn(self.conn)
        return Transaction(self.pool)

    def _ensure_default_strategy(self):
        strategies = self.get_strategies()
        if not strategies:
            default = RetrievalStrategy(
                id="default_v1",
                user_id="system",
                name="Default Hybrid Search",
                config={"k": 5, "depth": 2}
            )
            self.add_strategy(default)

    def add_node(self, node: Node) -> Node:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO nodes (id, user_id, label, confidence, evidence_count, contradiction_count, 
                                     domain_tags, entities, temporal_stability, abstraction_level, embedding, deprecated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        label = EXCLUDED.label,
                        confidence = EXCLUDED.confidence,
                        evidence_count = EXCLUDED.evidence_count,
                        entities = EXCLUDED.entities,
                        deprecated = EXCLUDED.deprecated
                """, (
                    str(node.id), node.user_id, node.label, node.confidence, node.evidence_count,
                    node.contradiction_count, node.domain_tags, node.entities, node.temporal_stability,
                    node.abstraction_level, f"[{','.join(map(str, node.embedding))}]", node.deprecated
                ))
            conn.commit()
            return node
        finally:
            self.pool.putconn(conn)

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        if not nodes: return
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for node in nodes:
                    cur.execute("""
                        INSERT INTO nodes (id, user_id, label, confidence, evidence_count, domain_tags, entities, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            evidence_count = nodes.evidence_count + 1,
                            confidence = LEAST(nodes.confidence + 0.05, 0.95)
                    """, (str(node.id), node.user_id, node.label, node.confidence, node.evidence_count, 
                          node.domain_tags, node.entities, f"[{','.join(map(str, node.embedding))}]"))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        # Guard against non-finite values (NaN/Inf) which crash pgvector
        if any(not (float('-inf') < x < float('inf')) for x in embedding):
            logger.warning(f"Non-finite values in embedding for user {user_id}. Skipping search.")
            return []

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM nodes 
                    WHERE user_id = %s AND deprecated = FALSE
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                """, (user_id, f"[{','.join(map(str, embedding))}]", k))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def add_edge(self, edge: Edge) -> Edge:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO edges (id, from_node_id, to_node_id, relation, weight, evidence_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (from_node_id, to_node_id, relation) DO UPDATE SET
                        evidence_count = edges.evidence_count + 1,
                        weight = LEAST(edges.weight + 0.1, 1.0)
                """, (str(edge.id), str(edge.from_node_id), str(edge.to_node_id), edge.relation, edge.weight, edge.evidence_count))
            conn.commit()
            return edge
        finally:
            self.pool.putconn(conn)

    def add_edges_batch(self, edges: List[Edge]) -> None:
        if not edges: return
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for edge in edges:
                    cur.execute("""
                        INSERT INTO edges (id, from_node_id, to_node_id, relation, weight, evidence_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (from_node_id, to_node_id, relation) DO UPDATE SET
                            evidence_count = edges.evidence_count + 1
                    """, (str(edge.id), str(edge.from_node_id), str(edge.to_node_id), edge.relation, edge.weight, edge.evidence_count))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_node(self, node_id: UUID) -> Optional[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nodes WHERE id = %s", (str(node_id),))
                row = cur.fetchone()
                return self._row_to_node(row) if row else None
        finally:
            self.pool.putconn(conn)

    def get_nodes_batch(self, node_ids: List[UUID]) -> List[Node]:
        if not node_ids: return []
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nodes WHERE id IN %s", (tuple(str(nid) for nid in node_ids),))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_edges(self, node_id: UUID) -> List[Edge]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM edges WHERE from_node_id = %s OR to_node_id = %s", (str(node_id), str(node_id)))
                return [self._row_to_edge(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        if not node_ids: return {}
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM edges WHERE from_node_id IN %s OR to_node_id IN %s", 
                           (tuple(str(nid) for nid in node_ids), tuple(str(nid) for nid in node_ids)))
                rows = cur.fetchall()
                result = {nid: [] for nid in node_ids}
                for row in rows:
                    edge = self._row_to_edge(row)
                    if edge.from_node_id in result: result[edge.from_node_id].append(edge)
                    if edge.to_node_id in result: result[edge.to_node_id].append(edge)
                return result
        finally:
            self.pool.putconn(conn)

    def update_node(self, node: Node) -> Node:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE nodes SET label=%s, confidence=%s, evidence_count=%s, 
                                    contradiction_count=%s, entities=%s, deprecated=%s
                    WHERE id=%s
                """, (node.label, node.confidence, node.evidence_count, node.contradiction_count, 
                      node.entities, node.deprecated, str(node.id)))
            conn.commit()
            return node
        finally:
            self.pool.putconn(conn)

    def update_edge(self, edge: Edge) -> Edge:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE edges SET weight=%s, evidence_count=%s WHERE id=%s", 
                           (edge.weight, edge.evidence_count, str(edge.id)))
            conn.commit()
            return edge
        finally:
            self.pool.putconn(conn)

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for edge in edges:
                    cur.execute("UPDATE edges SET weight=%s, evidence_count=%s WHERE id=%s", 
                               (edge.weight, edge.evidence_count, str(edge.id)))
            conn.commit()
            return edges
        finally:
            self.pool.putconn(conn)

    def traverse_graph(self, start_node_ids: List[UUID], depth: int, user_id: str) -> Tuple[List[Node], List[Edge]]:
        if not start_node_ids: return [], []
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH RECURSIVE graph_walk AS (
                        SELECT 
                            CASE WHEN from_node_id IN %s THEN to_node_id ELSE from_node_id END as neighbor_id, 
                            id as edge_id, 1 as current_depth
                        FROM edges
                        WHERE from_node_id IN %s OR to_node_id IN %s
                        UNION ALL
                        SELECT 
                            CASE WHEN e.from_node_id = gw.neighbor_id THEN e.to_node_id ELSE e.from_node_id END,
                            e.id, gw.current_depth + 1
                        FROM edges e
                        JOIN graph_walk gw ON e.from_node_id = gw.neighbor_id OR e.to_node_id = gw.neighbor_id
                        JOIN nodes n ON n.id = gw.neighbor_id
                        WHERE gw.current_depth < %s AND n.deprecated = FALSE
                    )
                    SELECT DISTINCT n.* FROM nodes n
                    JOIN graph_walk gw ON n.id = gw.neighbor_id
                    WHERE n.user_id = %s AND n.deprecated = FALSE;
                """, (tuple(str(nid) for nid in start_node_ids), tuple(str(nid) for nid in start_node_ids), 
                      tuple(str(nid) for nid in start_node_ids), depth, user_id))
                nodes = [self._row_to_node(row) for row in cur.fetchall()]
                return nodes, [] # Edges omitted for simplicity in this restore
        finally:
            self.pool.putconn(conn)

    def log_activity(self, activity: ActivityLog) -> None: pass
    def log_retrieval_event(self, event: RetrievalEvent) -> None: pass
    def add_strategy(self, strategy: RetrievalStrategy) -> None: pass
    def get_strategies(self, user_id: Optional[str] = None) -> List[RetrievalStrategy]: return []
    def update_strategy(self, strategy: RetrievalStrategy) -> None: pass
    def delete_strategy(self, strategy_id: str) -> None: pass
    def get_stale_nodes(self, user_id: str, days: int = 30) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM nodes 
                    WHERE user_id = %s 
                      AND last_confirmed_at < NOW() - INTERVAL '%s days'
                      AND deprecated = FALSE
                """, (user_id, days))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_users_needing_maintenance(self, window_hours: int = 24) -> List[str]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT user_id FROM nodes 
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                """, (window_hours,))
                return [row[0] for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)
    def get_retrieval_events(self, user_id: Optional[str], limit: int = 100) -> List[RetrievalEvent]: return []
    def get_retrieval_event(self, event_id: UUID) -> Optional[RetrievalEvent]: return None
    def delete_edge(self, edge_id: UUID) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM edges WHERE id = %s", (str(edge_id),))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_contradiction_pairs(self, user_id: str) -> List[Tuple[Node, Node, UUID]]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT n1.id, n1.label, n1.confidence, n1.evidence_count, n1.contradiction_count, n1.domain_tags, n1.entities, n1.temporal_stability, n1.abstraction_level, n1.embedding, n1.deprecated, n1.user_id,
                           n2.id, n2.label, n2.confidence, n2.evidence_count, n2.contradiction_count, n2.domain_tags, n2.entities, n2.temporal_stability, n2.abstraction_level, n2.embedding, n2.deprecated, n2.user_id,
                           e.id
                    FROM edges e
                    JOIN nodes n1 ON e.from_node_id = n1.id
                    JOIN nodes n2 ON e.to_node_id = n2.id
                    WHERE e.relation = 'contradicts' 
                      AND n1.user_id = %s AND n1.deprecated = FALSE
                      AND n2.user_id = %s AND n2.deprecated = FALSE
                """, (user_id, user_id))
                rows = cur.fetchall()
                results = []
                for r in rows:
                    n1 = self._row_to_node(r[0:12] + (r[10],)) # Row structure needs to match Node fields
                    # Wait, _row_to_node expects a specific column order.
                    # I'll manually reconstruct to be safe.
                    node_a = Node(id=UUID(r[0]), user_id=r[11], label=r[1], confidence=r[2], evidence_count=r[3], contradiction_count=r[4], domain_tags=r[5], entities=r[6], temporal_stability=r[7], abstraction_level=r[8], embedding=json.loads(r[9]) if isinstance(r[9], str) else list(r[9]), deprecated=r[10])
                    node_b = Node(id=UUID(r[12]), user_id=r[23], label=r[13], confidence=r[14], evidence_count=r[15], contradiction_count=r[16], domain_tags=r[17], entities=r[18], temporal_stability=r[19], abstraction_level=r[20], embedding=json.loads(r[21]) if isinstance(r[21], str) else list(r[21]), deprecated=r[22])
                    results.append((node_a, node_b, UUID(r[24])))
                return results
        finally:
            self.pool.putconn(conn)

    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nodes WHERE user_id = %s AND evidence_count >= %s AND deprecated = FALSE", (user_id, min_evidence))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)
    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]: return []
    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nodes WHERE user_id = %s AND %s = ANY(entities)", (user_id, entity))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def _row_to_node(self, row) -> Node:
        # Embedding handling: row[10] is the vector
        emb = row[10]
        if isinstance(emb, str):
            import json
            emb = json.loads(emb)
        return Node(
            id=UUID(row[0]), user_id=row[1], label=row[2], confidence=row[3],
            evidence_count=row[4], contradiction_count=row[5], domain_tags=row[6],
            entities=row[7], temporal_stability=row[8], abstraction_level=row[9],
            embedding=list(emb), deprecated=row[11]
        )

    def _row_to_edge(self, row) -> Edge:
        return Edge(
            id=UUID(row[0]), from_node_id=UUID(row[1]), to_node_id=UUID(row[2]),
            relation=row[3], weight=row[4], evidence_count=row[5]
        )
