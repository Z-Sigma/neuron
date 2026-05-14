import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from typing import List, Optional, Dict
from uuid import UUID
import json
import logging
from neuron.models import Node, Edge, ActivityLog, RetrievalEvent, RetrievalStrategy
from neuron.graph.store_interface import GraphStore
from neuron.config import settings

logger = logging.getLogger(__name__)

class PostgresGraphStore(GraphStore):
    def __init__(self, connection_url: str = settings.database_url):
        self.pool = pool.SimpleConnectionPool(1, 20, connection_url)
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

    def _ensure_default_strategy(self):
        strategies = self.get_strategies()
        if not strategies:
            default = RetrievalStrategy(
                id="default_v1",
                k_seeds=settings.k_seeds,
                traversal_depth=settings.traversal_depth,
                min_edge_weight=settings.min_edge_weight
            )
            self.add_strategy(default)

    def add_node(self, node: Node) -> Node:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
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
            conn.commit()
        finally:
            self.pool.putconn(conn)
        return node

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        if not nodes: return
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
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
                execute_values(cur, query, data)
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

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM nodes 
                    WHERE user_id = %s AND deprecated = FALSE
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                """, (user_id, embedding, k))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM nodes 
                    WHERE user_id = %s AND deprecated = TRUE
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                """, (user_id, embedding, k))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def add_edge(self, edge: Edge) -> Edge:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
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
            conn.commit()
        finally:
            self.pool.putconn(conn)
        return edge

    def add_edges_batch(self, edges: List[Edge]) -> None:
        if not edges: return
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
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
                execute_values(cur, query, data)
            conn.commit()
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
        ids_str = tuple(str(nid) for nid in node_ids)
        result = {nid: [] for nid in node_ids}
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM edges WHERE from_node_id IN %s OR to_node_id IN %s", (ids_str, ids_str))
                for row in cur.fetchall():
                    edge = self._row_to_edge(row)
                    if edge.from_node_id in result: result[edge.from_node_id].append(edge)
                    if edge.to_node_id in result: result[edge.to_node_id].append(edge)
        finally:
            self.pool.putconn(conn)
        return result

    def update_node(self, node: Node) -> Node:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
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
            conn.commit()
        finally:
            self.pool.putconn(conn)
        return node

    def get_stale_nodes(self, user_id: str, days: int = 30, conf_threshold: float = 0.35) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM nodes 
                    WHERE user_id = %s 
                    AND last_confirmed_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * %s
                    AND confidence < %s
                    AND deprecated = FALSE
                """, (user_id, days, conf_threshold))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nodes WHERE user_id = %s AND deprecated = FALSE LIMIT %s", (user_id, limit))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nodes WHERE user_id = %s AND %s = ANY(entities) AND deprecated = FALSE", (user_id, entity))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def update_edge(self, edge: Edge) -> Edge:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE edges SET weight = %s, evidence_count = %s, last_updated_at = %s WHERE id = %s",
                    (edge.weight, edge.evidence_count, edge.last_updated_at, str(edge.id)))
            conn.commit()
        finally:
            self.pool.putconn(conn)
        return edge

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        if not edges: return []
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                data = [(edge.weight, edge.evidence_count, edge.last_updated_at, str(edge.id)) for edge in edges]
                execute_values(cur, """
                    UPDATE edges AS e SET
                        weight = data.weight,
                        evidence_count = data.evidence_count,
                        last_updated_at = data.last_updated_at
                    FROM (VALUES %s) AS data(weight, evidence_count, last_updated_at, id)
                    WHERE e.id = data.id::uuid
                """, data)
            conn.commit()
        finally:
            self.pool.putconn(conn)
        return edges

    # --- Adaptive Memory Extensions ---

    def log_activity(self, activity: ActivityLog) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO activity_log (id, user_id, activity_type, details, created_at) VALUES (%s, %s, %s, %s, %s)",
                    (str(activity.id), activity.user_id, activity.activity_type.value, activity.details, activity.created_at))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def log_retrieval_event(self, event: RetrievalEvent) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO retrieval_events (id, user_id, query, strategy_id, nodes_found, score, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (str(event.id), event.user_id, event.query, event.strategy_id, event.nodes_found, event.score, event.created_at))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def add_strategy(self, strategy: RetrievalStrategy) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO retrieval_strategies (id, k_seeds, traversal_depth, min_edge_weight, fitness_score, generations_survived, parent_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (strategy.id, strategy.k_seeds, strategy.traversal_depth, strategy.min_edge_weight, strategy.fitness_score, strategy.generations_survived, strategy.parent_id))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_strategies(self) -> List[RetrievalStrategy]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, k_seeds, traversal_depth, min_edge_weight, fitness_score, generations_survived, parent_id FROM retrieval_strategies")
                return [RetrievalStrategy(id=r[0], k_seeds=r[1], traversal_depth=r[2], min_edge_weight=r[3], fitness_score=r[4], generations_survived=r[5], parent_id=r[6]) for r in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def update_strategy(self, strategy: RetrievalStrategy) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE retrieval_strategies SET fitness_score = %s, generations_survived = %s WHERE id = %s",
                    (strategy.fitness_score, strategy.generations_survived, strategy.id))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_users_needing_maintenance(self, window_hours: int = 24) -> List[str]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""SELECT DISTINCT user_id FROM activity_log WHERE created_at >= NOW() - INTERVAL '%s hours'
                               UNION SELECT DISTINCT user_id FROM retrieval_events WHERE created_at >= NOW() - INTERVAL '%s hours'""",
                    (window_hours, window_hours))
                return [row[0] for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_retrieval_events(self, user_id: Optional[str], limit: int = 100) -> List[RetrievalEvent]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                if user_id:
                    cur.execute("SELECT id, user_id, query, strategy_id, nodes_found, score, created_at FROM retrieval_events WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                        (user_id, limit))
                else:
                    cur.execute("SELECT id, user_id, query, strategy_id, nodes_found, score, created_at FROM retrieval_events ORDER BY created_at DESC LIMIT %s",
                        (limit,))
                return [RetrievalEvent(id=r[0], user_id=r[1], query=r[2], strategy_id=r[3], nodes_found=r[4], score=r[5], created_at=r[6]) for r in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_contradiction_pairs(self, user_id: str) -> List[tuple]:
        # Placeholder for daemon-specific query
        return []

    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nodes WHERE user_id = %s AND evidence_count >= %s AND confidence < 0.95 AND deprecated = FALSE", (user_id, min_evidence))
                return [self._row_to_node(row) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def _row_to_node(self, row) -> Node:
        return Node(
            id=UUID(row[0]), user_id=row[1], label=row[2], confidence=row[3],
            evidence_count=row[4], contradiction_count=row[5], created_at=row[6],
            last_confirmed_at=row[7], last_contradicted_at=row[8], domain_tags=row[9] or [],
            entities=row[10] or [], temporal_stability=row[11], abstraction_level=row[12],
            embedding=json.loads(row[13]) if isinstance(row[13], str) else row[13], deprecated=row[14]
        )

    def _row_to_edge(self, row) -> Edge:
        return Edge(
            id=UUID(row[0]), from_node_id=UUID(row[1]), to_node_id=UUID(row[2]),
            relation=row[3], weight=row[4], evidence_count=row[5],
            created_at=row[6], last_updated_at=row[7]
        )
