import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from uuid import UUID
from datetime import datetime, timezone
import psycopg2
from psycopg2 import pool
from neuron.config import settings
from neuron.models import Node, Edge, ActivityLog, RetrievalEvent, RetrievalStrategy
from neuron.graph.store_interface import GraphStore

logger = logging.getLogger(__name__)
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class PostgresGraphStore(GraphStore):
    def __init__(self, connection_string: str = settings.database_url):
        self.pool = psycopg2.pool.ThreadedConnectionPool(1, 20, connection_string)
        self.setup()

    def setup(self) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_PATH.read_text(encoding="utf-8"))
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
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
                self.pool.putconn(self.conn)
        return Transaction(self.pool)

    def _ensure_default_strategy(self):
        if not self.get_strategies():
            default = RetrievalStrategy(
                id="default_v1",
                user_id="system",
                k_seeds=settings.k_seeds,
                traversal_depth=settings.traversal_depth,
                min_edge_weight=settings.min_edge_weight,
                novelty_threshold=0.8,
            )
            self.add_strategy(default)

    def _vector_literal(self, embedding: Optional[List[float]]) -> str:
        if not embedding:
            return "[]"
        return f"[{','.join(map(str, embedding))}]"

    def add_node(self, node: Node) -> Node:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO nodes (id, user_id, label, confidence, evidence_count, contradiction_count,
                                     domain_tags, entities, temporal_stability, abstraction_level, embedding, deprecated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        label = EXCLUDED.label,
                        confidence = EXCLUDED.confidence,
                        evidence_count = EXCLUDED.evidence_count,
                        entities = EXCLUDED.entities,
                        deprecated = EXCLUDED.deprecated,
                        last_confirmed_at = CURRENT_TIMESTAMP
                """, (
                    str(node.id), node.user_id, node.label, node.confidence, node.evidence_count,
                    node.contradiction_count, node.domain_tags, node.entities, node.temporal_stability,
                    node.abstraction_level, self._vector_literal(node.embedding), node.deprecated
                ))
            conn.commit()
            return node
        finally:
            self.pool.putconn(conn)

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        if not nodes:
            return
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for node in nodes:
                    cur.execute("""
                        INSERT INTO nodes (id, user_id, label, confidence, evidence_count, domain_tags, entities, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (id) DO UPDATE SET
                            evidence_count = nodes.evidence_count + 1,
                            confidence = LEAST(nodes.confidence + 0.05, 0.95),
                            last_confirmed_at = CURRENT_TIMESTAMP
                    """, (
                        str(node.id), node.user_id, node.label, node.confidence, node.evidence_count,
                        node.domain_tags, node.entities, self._vector_literal(node.embedding)
                    ))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        return self._vector_search(embedding, user_id, k, deprecated=False)

    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        return self._vector_search(embedding, user_id, k, deprecated=True)

    def _vector_search(self, embedding: List[float], user_id: str, k: int, deprecated: bool) -> List[Node]:
        if any(not (float("-inf") < x < float("inf")) for x in embedding):
            logger.warning("Non-finite values in embedding for user %s. Skipping search.", user_id)
            return []

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM nodes
                    WHERE user_id = %s AND deprecated = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (user_id, deprecated, self._vector_literal(embedding), k))
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
                """, (
                    str(edge.id), str(edge.from_node_id), str(edge.to_node_id),
                    edge.relation, edge.weight, edge.evidence_count
                ))
            conn.commit()
            return edge
        finally:
            self.pool.putconn(conn)

    def add_edges_batch(self, edges: List[Edge]) -> None:
        if not edges:
            return
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for edge in edges:
                    cur.execute("""
                        INSERT INTO edges (id, from_node_id, to_node_id, relation, weight, evidence_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (from_node_id, to_node_id, relation) DO UPDATE SET
                            evidence_count = edges.evidence_count + 1
                    """, (
                        str(edge.id), str(edge.from_node_id), str(edge.to_node_id),
                        edge.relation, edge.weight, edge.evidence_count
                    ))
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
        if not node_ids:
            return []
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
                cur.execute(
                    "SELECT * FROM edges WHERE from_node_id = %s OR to_node_id = %s",
                    (str(node_id), str(node_id)),
                )
                return [self._row_to_edge(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        if not node_ids:
            return {}
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                ids = tuple(str(nid) for nid in node_ids)
                cur.execute(
                    "SELECT * FROM edges WHERE from_node_id IN %s OR to_node_id IN %s",
                    (ids, ids),
                )
                result = {nid: [] for nid in node_ids}
                for row in cur.fetchall():
                    edge = self._row_to_edge(row)
                    if edge.from_node_id in result:
                        result[edge.from_node_id].append(edge)
                    if edge.to_node_id in result:
                        result[edge.to_node_id].append(edge)
                return result
        finally:
            self.pool.putconn(conn)

    def update_node(self, node: Node) -> Node:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE nodes SET label=%s, confidence=%s, evidence_count=%s,
                                    contradiction_count=%s, entities=%s, deprecated=%s,
                                    last_confirmed_at=%s
                    WHERE id=%s
                """, (
                    node.label, node.confidence, node.evidence_count, node.contradiction_count,
                    node.entities, node.deprecated, node.last_confirmed_at, str(node.id),
                ))
            conn.commit()
            return node
        finally:
            self.pool.putconn(conn)

    def update_edge(self, edge: Edge) -> Edge:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE edges SET weight=%s, evidence_count=%s WHERE id=%s",
                    (edge.weight, edge.evidence_count, str(edge.id)),
                )
            conn.commit()
            return edge
        finally:
            self.pool.putconn(conn)

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for edge in edges:
                    cur.execute(
                        "UPDATE edges SET weight=%s, evidence_count=%s WHERE id=%s",
                        (edge.weight, edge.evidence_count, str(edge.id)),
                    )
            conn.commit()
            return edges
        finally:
            self.pool.putconn(conn)

    def traverse_graph(self, start_node_ids: List[UUID], depth: int, user_id: str) -> Tuple[List[Node], List[Edge]]:
        if not start_node_ids:
            return [], []

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                start_ids = tuple(str(nid) for nid in start_node_ids)
                cur.execute("""
                    WITH RECURSIVE graph_walk AS (
                        SELECT
                            CASE WHEN from_node_id IN %s THEN to_node_id ELSE from_node_id END AS neighbor_id,
                            id AS edge_id,
                            1 AS current_depth
                        FROM edges
                        WHERE from_node_id IN %s OR to_node_id IN %s
                        UNION ALL
                        SELECT
                            CASE WHEN e.from_node_id = gw.neighbor_id THEN e.to_node_id ELSE e.from_node_id END,
                            e.id,
                            gw.current_depth + 1
                        FROM edges e
                        JOIN graph_walk gw ON e.from_node_id = gw.neighbor_id OR e.to_node_id = gw.neighbor_id
                        JOIN nodes n ON n.id = gw.neighbor_id
                        WHERE gw.current_depth < %s AND n.deprecated = FALSE
                    )
                    SELECT DISTINCT n.* FROM nodes n
                    JOIN graph_walk gw ON n.id = gw.neighbor_id
                    WHERE n.user_id = %s AND n.deprecated = FALSE
                """, (start_ids, start_ids, start_ids, depth, user_id))
                nodes = [self._row_to_node(row) for row in cur.fetchall()]

                node_ids = tuple({str(n.id) for n in nodes} | {str(nid) for nid in start_node_ids})
                if not node_ids:
                    return nodes, []
                cur.execute(
                    "SELECT * FROM edges WHERE from_node_id IN %s OR to_node_id IN %s",
                    (node_ids, node_ids),
                )
                edges = [self._row_to_edge(row) for row in cur.fetchall()]
                return nodes, edges
        finally:
            self.pool.putconn(conn)

    def log_activity(self, activity: ActivityLog) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO activity_log (id, user_id, activity_type, details)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    str(activity.id), activity.user_id, activity.activity_type.value, activity.details,
                ))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def log_retrieval_event(self, event: RetrievalEvent) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO retrieval_events (id, user_id, query, strategy_id, nodes_found, score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET score = EXCLUDED.score
                """, (
                    str(event.id), event.user_id, event.query, event.strategy_id,
                    event.nodes_found, event.score,
                ))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def add_strategy(self, strategy: RetrievalStrategy) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO retrieval_strategies
                        (id, user_id, k_seeds, traversal_depth, min_edge_weight, novelty_threshold,
                         fitness_score, generations_survived, parent_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        k_seeds = EXCLUDED.k_seeds,
                        traversal_depth = EXCLUDED.traversal_depth,
                        min_edge_weight = EXCLUDED.min_edge_weight,
                        novelty_threshold = EXCLUDED.novelty_threshold,
                        fitness_score = EXCLUDED.fitness_score,
                        generations_survived = EXCLUDED.generations_survived,
                        parent_id = EXCLUDED.parent_id
                """, (
                    strategy.id, strategy.user_id, strategy.k_seeds, strategy.traversal_depth,
                    strategy.min_edge_weight, strategy.novelty_threshold, strategy.fitness_score,
                    strategy.generations_survived, strategy.parent_id,
                ))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_strategies(self, user_id: Optional[str] = None) -> List[RetrievalStrategy]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                if user_id:
                    cur.execute("""
                        SELECT * FROM retrieval_strategies
                        WHERE user_id = %s OR user_id = 'system' OR user_id IS NULL
                    """, (user_id,))
                else:
                    cur.execute("SELECT * FROM retrieval_strategies")
                return [self._row_to_strategy(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def update_strategy(self, strategy: RetrievalStrategy) -> None:
        self.add_strategy(strategy)

    def delete_strategy(self, strategy_id: str) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM retrieval_strategies WHERE id = %s", (strategy_id,))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_stale_nodes(self, user_id: str, days: int = 30) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM nodes
                    WHERE user_id = %s
                      AND last_confirmed_at < NOW() - (%s * interval '1 day')
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
                    SELECT DISTINCT user_id FROM activity_log
                    WHERE created_at > NOW() - (%s * interval '1 hour')
                """, (window_hours,))
                users = [row[0] for row in cur.fetchall()]
                if users:
                    return users
                cur.execute("""
                    SELECT DISTINCT user_id FROM nodes
                    WHERE created_at > NOW() - (%s * interval '1 hour')
                """, (window_hours,))
                return [row[0] for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_retrieval_events(self, user_id: Optional[str], limit: int = 100) -> List[RetrievalEvent]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                if user_id:
                    cur.execute("""
                        SELECT * FROM retrieval_events
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, limit))
                else:
                    cur.execute("""
                        SELECT * FROM retrieval_events
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                return [self._row_to_retrieval_event(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_retrieval_event(self, event_id: UUID) -> Optional[RetrievalEvent]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM retrieval_events WHERE id = %s", (str(event_id),))
                row = cur.fetchone()
                return self._row_to_retrieval_event(row) if row else None
        finally:
            self.pool.putconn(conn)

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
                    SELECT e.id, e.from_node_id, e.to_node_id
                    FROM edges e
                    JOIN nodes n1 ON e.from_node_id = n1.id
                    JOIN nodes n2 ON e.to_node_id = n2.id
                    WHERE e.relation = 'contradicts'
                      AND n1.user_id = %s AND n1.deprecated = FALSE
                      AND n2.user_id = %s AND n2.deprecated = FALSE
                """, (user_id, user_id))
                results = []
                for edge_id, from_id, to_id in cur.fetchall():
                    node_a = self.get_node(UUID(from_id))
                    node_b = self.get_node(UUID(to_id))
                    if node_a and node_b:
                        results.append((node_a, node_b, UUID(edge_id)))
                return results
        finally:
            self.pool.putconn(conn)

    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM nodes WHERE user_id = %s AND evidence_count >= %s AND deprecated = FALSE",
                    (user_id, min_evidence),
                )
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM nodes WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                    (user_id, limit),
                )
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM nodes WHERE user_id = %s AND %s = ANY(entities)",
                    (user_id, entity),
                )
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def _parse_embedding(self, emb) -> List[float]:
        if emb is None:
            return []
        if isinstance(emb, str):
            return json.loads(emb)
        return list(emb)

    def _row_to_node(self, row) -> Node:
        return Node(
            id=UUID(row[0]),
            user_id=row[1],
            label=row[2],
            confidence=row[3],
            evidence_count=row[4],
            contradiction_count=row[5],
            created_at=row[6] or datetime.now(timezone.utc),
            last_confirmed_at=row[7] or datetime.now(timezone.utc),
            last_contradicted_at=row[8],
            domain_tags=row[9] or [],
            entities=row[10] or [],
            temporal_stability=row[11] or "stable",
            abstraction_level=row[12] or "specific",
            embedding=self._parse_embedding(row[13]),
            deprecated=bool(row[14]),
        )

    def _row_to_edge(self, row) -> Edge:
        return Edge(
            id=UUID(row[0]),
            from_node_id=UUID(row[1]),
            to_node_id=UUID(row[2]),
            relation=row[3],
            weight=row[4],
            evidence_count=row[5],
        )

    def _row_to_strategy(self, row) -> RetrievalStrategy:
        return RetrievalStrategy(
            id=row[0],
            user_id=row[1],
            k_seeds=row[2],
            traversal_depth=row[3],
            min_edge_weight=row[4],
            novelty_threshold=row[5] if row[5] is not None else 0.8,
            fitness_score=row[6] if row[6] is not None else 0.0,
            generations_survived=row[7] if row[7] is not None else 0,
            parent_id=row[8],
        )

    def _row_to_retrieval_event(self, row) -> RetrievalEvent:
        return RetrievalEvent(
            id=UUID(row[0]),
            user_id=row[1],
            query=row[2],
            strategy_id=row[3],
            nodes_found=row[4],
            score=row[5] if row[5] is not None else 0.0,
            created_at=row[6] or datetime.now(timezone.utc),
        )
