import logging
from typing import List, Optional, Dict, Tuple
from uuid import UUID
from datetime import datetime, timezone
from neo4j import GraphDatabase
from neuron.config import settings
from neuron.models import Node, Edge, ActivityLog, RetrievalEvent, RetrievalStrategy
from neuron.graph.store_interface import GraphStore

logger = logging.getLogger(__name__)


def _neo4j_props(data: dict) -> dict:
    """Serialize Pydantic model fields for the Neo4j driver."""
    out = {}
    for key, value in data.items():
        if isinstance(value, UUID):
            out[key] = str(value)
        elif isinstance(value, datetime):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str = settings.neo4j_uri, user: str = settings.neo4j_user, password: str = settings.neo4j_password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.setup()

    def transaction(self):
        class Neo4jTransaction:
            def __init__(self, driver):
                self.driver = driver
                self.session = None
                self.tx = None
            def __enter__(self):
                self.session = self.driver.session()
                self.tx = self.session.begin_transaction()
                return self.tx
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None: self.tx.commit()
                else: self.tx.rollback()
                self.session.close()
        return Neo4jTransaction(self.driver)

    def setup(self) -> None:
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Belief) REQUIRE n.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Belief) ON (n.user_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Belief) ON (n.deprecated)")
            try:
                session.run(
                    "CREATE VECTOR INDEX belief_embeddings IF NOT EXISTS "
                    "FOR (n:Belief) ON (n.embedding) "
                    "OPTIONS {indexConfig: {"
                    f"`vector.dimensions`: {settings.embedding_dimension}, "
                    "`vector.similarity_function`: 'cosine'"
                    "}}"
                )
            except Exception as exc:
                logger.debug("Vector index setup skipped: %s", exc)
            self._ensure_default_strategy()

    def _ensure_default_strategy(self) -> None:
        if not any(s.id == "default_v1" for s in self.get_strategies()):
            default = RetrievalStrategy(
                id="default_v1",
                user_id="system",
                k_seeds=settings.k_seeds,
                traversal_depth=settings.traversal_depth,
                min_edge_weight=settings.min_edge_weight,
                novelty_threshold=0.8,
            )
            self.add_strategy(default)

    def add_node(self, node: Node) -> Node:
        with self.driver.session() as session:
            props = _neo4j_props(node.model_dump(exclude={'id', 'created_at', 'last_confirmed_at'}))
            session.run("MERGE (n:Belief {id: $id}) SET n += $props, n.created_at = datetime($created_at), n.last_confirmed_at = datetime($last_confirmed_at)", id=str(node.id), props=props, created_at=node.created_at.isoformat(), last_confirmed_at=node.last_confirmed_at.isoformat())
            return node

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        if not nodes: return
        processed = []
        for n in nodes:
            d = n.model_dump(); d['id'] = str(d['id']); d['created_at'] = d['created_at'].isoformat(); d['last_confirmed_at'] = d['last_confirmed_at'].isoformat()
            processed.append(d)
        with self.driver.session() as session:
            session.run("UNWIND $nodes as n_data MERGE (n:Belief {id: n_data.id}) SET n += n_data, n.created_at = datetime(n_data.created_at), n.last_confirmed_at = datetime(n_data.last_confirmed_at)", nodes=processed)

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        return self._vector_search(embedding, user_id, k, deprecated=False)

    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        return self._vector_search(embedding, user_id, k, deprecated=True)

    def _vector_search(
        self, embedding: List[float], user_id: str, k: int, deprecated: bool
    ) -> List[Node]:
        """Per-user vector search (scoped like Postgres), not global top-k + filter."""
        with self.driver.session() as session:
            res = session.run(
                """
                MATCH (n:Belief)
                WHERE n.user_id = $uid AND n.deprecated = $deprecated AND n.embedding IS NOT NULL
                WITH n, vector.similarity.cosine(n.embedding, $emb) AS score
                ORDER BY score DESC
                LIMIT $k
                RETURN n
                """,
                uid=user_id,
                deprecated=deprecated,
                emb=embedding,
                k=k,
            )
            return [self._row_to_node(record["n"]) for record in res]

    def add_edge(self, edge: Edge) -> Edge:
        with self.driver.session() as session:
            props = _neo4j_props(edge.model_dump(exclude={'from_node_id', 'to_node_id', 'relation'}))
            session.run("MATCH (a:Belief {id: $fid}), (b:Belief {id: $tid}) MERGE (a)-[r:ASSOCIATION {relation: $rel}]->(b) SET r += $props", fid=str(edge.from_node_id), tid=str(edge.to_node_id), rel=edge.relation, props=props)
            return edge

    def add_edges_batch(self, edges: List[Edge]) -> None:
        if not edges: return
        processed = []
        for e in edges:
            d = e.model_dump(); d['id'] = str(d['id']); d['from_node_id'] = str(d['from_node_id']); d['to_node_id'] = str(d['to_node_id']); d['created_at'] = d['created_at'].isoformat(); d['last_updated_at'] = d['last_updated_at'].isoformat()
            processed.append(d)
        with self.driver.session() as session:
            session.run("UNWIND $edges as e_data MATCH (a:Belief {id: e_data.from_node_id}), (b:Belief {id: e_data.to_node_id}) MERGE (a)-[r:ASSOCIATION {relation: e_data.relation}]->(b) SET r += e_data", edges=processed)

    def get_node(self, node_id: UUID) -> Optional[Node]:
        with self.driver.session() as session:
            res = session.run("MATCH (n:Belief {id: $id}) RETURN n", id=str(node_id)).single()
            return self._row_to_node(res['n']) if res else None

    def get_nodes_batch(self, node_ids: List[UUID]) -> List[Node]:
        with self.driver.session() as session:
            res = session.run("MATCH (n:Belief) WHERE n.id IN $ids RETURN n", ids=[str(nid) for nid in node_ids])
            return [self._row_to_node(record['n']) for record in res]

    def get_edges(self, node_id: UUID) -> List[Edge]:
        with self.driver.session() as session:
            res = session.run("MATCH (n:Belief {id: $id})-[r:ASSOCIATION]-(o) RETURN r", id=str(node_id))
            return [self._row_to_edge(record['r']) for record in res]

    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        with self.driver.session() as session:
            res = session.run("MATCH (n:Belief)-[r:ASSOCIATION]-(o) WHERE n.id IN $ids RETURN n.id as nid, r", ids=[str(nid) for nid in node_ids])
            out = {nid: [] for nid in node_ids}
            for rec in res: out[UUID(rec['nid'])].append(self._row_to_edge(rec['r']))
            return out

    def update_node(self, node: Node) -> Node:
        with self.driver.session() as session:
            props = _neo4j_props(node.model_dump(exclude={'id', 'created_at'}))
            session.run("MATCH (n:Belief {id: $id}) SET n += $props", id=str(node.id), props=props)
            return node

    def update_edge(self, edge: Edge) -> Edge:
        with self.driver.session() as session:
            session.run("MATCH ()-[r:ASSOCIATION {id: $id}]->() SET r.weight = $w, r.evidence_count = $e", id=str(edge.id), w=edge.weight, e=edge.evidence_count)
            return edge

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        for e in edges: self.update_edge(e)
        return edges

    def traverse_graph(self, start_node_ids: List[UUID], depth: int, user_id: str) -> Tuple[List[Node], List[Edge]]:
        try:
            with self.driver.session() as session:
                res = session.run(
                    "MATCH (s:Belief) WHERE s.id IN $ids "
                    "CALL apoc.path.subgraphAll(s, {maxLevel: $d, relationshipFilter: 'ASSOCIATION', labelFilter: '+Belief'}) "
                    "YIELD nodes, relationships RETURN nodes, relationships",
                    ids=[str(nid) for nid in start_node_ids],
                    d=depth,
                ).single()
                if not res:
                    return [], []
                nodes = [
                    self._row_to_node(n) for n in res['nodes']
                    if n.get('user_id') == user_id and not n.get('deprecated')
                ]
                edges = [self._row_to_edge(r) for r in res['relationships']]
                return nodes, edges
        except Exception as exc:
            logger.debug("APOC traversal unavailable, using BFS fallback: %s", exc)
            return self._traverse_bfs(start_node_ids, depth, user_id)

    def _traverse_bfs(self, start_node_ids: List[UUID], depth: int, user_id: str) -> Tuple[List[Node], List[Edge]]:
        from collections import deque
        visited_nodes: Dict[UUID, Node] = {}
        visited_edges: Dict[UUID, Edge] = {}
        queue = deque([(nid, 0) for nid in start_node_ids])
        while queue:
            curr_id, curr_depth = queue.popleft()
            if curr_id in visited_nodes or curr_depth > depth:
                continue
            node = self.get_node(curr_id)
            if not node or node.user_id != user_id or node.deprecated:
                continue
            visited_nodes[curr_id] = node
            if curr_depth < depth:
                for edge in self.get_edges(curr_id):
                    visited_edges[edge.id] = edge
                    next_id = edge.to_node_id if edge.from_node_id == curr_id else edge.from_node_id
                    queue.append((next_id, curr_depth + 1))
        return list(visited_nodes.values()), list(visited_edges.values())

    def log_activity(self, activity: ActivityLog) -> None:
        with self.driver.session() as session:
            session.run(
                "CREATE (a:ActivityLog {id: $id, user_id: $uid, activity_type: $type, details: $details, created_at: datetime()})",
                id=str(activity.id),
                uid=activity.user_id,
                type=activity.activity_type.value,
                details=activity.details,
            )
    def log_retrieval_event(self, event: RetrievalEvent) -> None:
        with self.driver.session() as session:
            session.run("CREATE (e:RetrievalEvent {id: $id, user_id: $uid, query: $q, nodes_found: $nf, score: $s, created_at: datetime()}) WITH e MATCH (s:RetrievalStrategy {id: $sid}) MERGE (e)-[:USED_STRATEGY]->(s)", id=str(event.id), uid=event.user_id, q=event.query, nf=event.nodes_found, s=event.score, sid=event.strategy_id)

    def add_strategy(self, strategy: RetrievalStrategy) -> None:
        with self.driver.session() as session:
            session.run("MERGE (s:RetrievalStrategy {id: $id}) SET s += $props", id=strategy.id, props=strategy.model_dump())

    def get_strategies(self, user_id: Optional[str] = None) -> List[RetrievalStrategy]:
        with self.driver.session() as session:
            q = "MATCH (s:RetrievalStrategy) "
            if user_id: q += "WHERE s.user_id = $uid OR s.user_id = 'system' "
            q += "RETURN s"
            res = session.run(q, uid=user_id)
            return [RetrievalStrategy(**dict(record['s'])) for record in res]

    def update_strategy(self, strategy: RetrievalStrategy) -> None: self.add_strategy(strategy)
    def delete_strategy(self, strategy_id: str) -> None:
        with self.driver.session() as session: session.run("MATCH (s:RetrievalStrategy {id: $id}) DETACH DELETE s", id=strategy_id)

    def get_users_needing_maintenance(self, window_hours: int = 24) -> List[str]:
        with self.driver.session() as session:
            res = session.run("""
                MATCH (a:ActivityLog)
                WHERE a.created_at > datetime() - duration({hours: $h})
                RETURN DISTINCT a.user_id AS uid
            """, h=window_hours)
            users = [record['uid'] for record in res]
            if users:
                return users
            res = session.run("""
                MATCH (n:Belief)
                WHERE n.created_at > datetime() - duration({hours: $h})
                RETURN DISTINCT n.user_id AS uid
            """, h=window_hours)
            return [record['uid'] for record in res]

    def get_stale_nodes(self, user_id: str, days: int = 30) -> List[Node]:
        with self.driver.session() as session:
            res = session.run("""
                MATCH (n:Belief)
                WHERE n.user_id = $uid AND n.deprecated = FALSE
                  AND n.last_confirmed_at < datetime() - duration({days: $days})
                RETURN n
            """, uid=user_id, days=days)
            return [self._row_to_node(record['n']) for record in res]

    def get_retrieval_events(self, user_id: Optional[str], limit: int = 100) -> List[RetrievalEvent]:
        with self.driver.session() as session:
            if user_id:
                res = session.run("""
                    MATCH (e:RetrievalEvent {user_id: $uid})
                    OPTIONAL MATCH (e)-[:USED_STRATEGY]->(s:RetrievalStrategy)
                    RETURN e, s.id AS strategy_id
                    ORDER BY e.created_at DESC
                    LIMIT $limit
                """, uid=user_id, limit=limit)
            else:
                res = session.run("""
                    MATCH (e:RetrievalEvent)
                    OPTIONAL MATCH (e)-[:USED_STRATEGY]->(s:RetrievalStrategy)
                    RETURN e, s.id AS strategy_id
                    ORDER BY e.created_at DESC
                    LIMIT $limit
                """, limit=limit)
            events = []
            for record in res:
                e = record['e']
                events.append(RetrievalEvent(
                    id=UUID(e['id']),
                    user_id=e['user_id'],
                    query=e['query'],
                    strategy_id=record['strategy_id'] or e.get('strategy_id', ''),
                    nodes_found=e['nodes_found'],
                    score=e.get('score', 0.0),
                ))
            return events

    def get_retrieval_event(self, event_id: UUID) -> Optional[RetrievalEvent]:
        with self.driver.session() as session:
            res = session.run("""
                MATCH (e:RetrievalEvent {id: $id})
                OPTIONAL MATCH (e)-[:USED_STRATEGY]->(s:RetrievalStrategy)
                RETURN e, s.id as strategy_id
            """, id=str(event_id)).single()
            if not res: return None
            e = res['e']
            return RetrievalEvent(
                id=UUID(e['id']), user_id=e['user_id'], query=e['query'], 
                strategy_id=res['strategy_id'], nodes_found=e['nodes_found'], score=e['score']
            )

    def delete_edge(self, edge_id: UUID) -> None:
        with self.driver.session() as session:
            session.run("MATCH ()-[r:ASSOCIATION {id: $id}]-() DELETE r", id=str(edge_id))

    def get_contradiction_pairs(self, user_id: str) -> List[Tuple[Node, Node, UUID]]:
        with self.driver.session() as session:
            res = session.run("""
                MATCH (a:Belief)-[r:ASSOCIATION {relation: 'contradicts'}]->(b:Belief)
                WHERE a.user_id = $uid AND a.deprecated = FALSE
                  AND b.user_id = $uid AND b.deprecated = FALSE
                RETURN a, b, r.id as eid
            """, uid=user_id)
            return [(self._row_to_node(rec['a']), self._row_to_node(rec['b']), UUID(rec['eid'])) for rec in res]
    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        with self.driver.session() as session:
            res = session.run("MATCH (n:Belief) WHERE n.user_id = $uid AND n.evidence_count >= $me AND n.deprecated = FALSE RETURN n", uid=user_id, me=min_evidence)
            return [self._row_to_node(record['n']) for record in res]

    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        with self.driver.session() as session:
            res = session.run("MATCH (n:Belief) WHERE n.user_id = $uid RETURN n LIMIT $l", uid=user_id, l=limit)
            return [self._row_to_node(record['n']) for record in res]

    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        with self.driver.session() as session:
            res = session.run("MATCH (n:Belief) WHERE n.user_id = $uid AND $ent IN n.entities RETURN n", uid=user_id, ent=entity)
            return [self._row_to_node(record['n']) for record in res]

    def _row_to_node(self, n) -> Node:
        created = n.get('created_at')
        confirmed = n.get('last_confirmed_at')
        if hasattr(created, 'to_native'):
            created = created.to_native()
        if hasattr(confirmed, 'to_native'):
            confirmed = confirmed.to_native()
        embedding = n.get('embedding')
        if embedding is None:
            emb_list = []
        else:
            emb_list = list(embedding)
        return Node(
            id=UUID(n['id']),
            user_id=n['user_id'],
            label=n['label'],
            confidence=n['confidence'],
            evidence_count=n.get('evidence_count', 1),
            contradiction_count=n.get('contradiction_count', 0),
            created_at=created or datetime.now(timezone.utc),
            last_confirmed_at=confirmed or datetime.now(timezone.utc),
            domain_tags=list(n.get('domain_tags', [])),
            entities=list(n.get('entities', [])),
            temporal_stability=n.get('temporal_stability', 'stable'),
            abstraction_level=n.get('abstraction_level', 'specific'),
            embedding=emb_list,
            deprecated=n.get('deprecated', False),
        )

    def _row_to_edge(self, r) -> Edge:
        return Edge(id=UUID(r['id']), from_node_id=UUID(r.start_node['id']), to_node_id=UUID(r.end_node['id']), relation=r['relation'], weight=r.get('weight', 0.5), evidence_count=r.get('evidence_count', 1))
