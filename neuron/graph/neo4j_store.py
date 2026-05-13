from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime
import logging
from neo4j import GraphDatabase
from neuron.models import Node, Edge
from neuron.graph.store_interface import GraphStore
from neuron.config import settings

logger = logging.getLogger(__name__)

class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str = settings.neo4j_uri, user: str = settings.neo4j_user, password: str = settings.neo4j_password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.setup()

    def setup(self) -> None:
        with self.driver.session() as session:
            # 1. Unique Constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Belief) REQUIRE n.id IS UNIQUE")
            
            # 2. Metadata Indexes
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Belief) ON (n.user_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Belief) ON (n.deprecated)")
            
            # 3. Vector Index (Neo4j 5.x syntax)
            try:
                session.run("""
                    CREATE VECTOR INDEX belief_embeddings IF NOT EXISTS
                    FOR (n:Belief) ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {settings.embedding_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """.format(settings=settings))
            except Exception as e:
                logger.warning(f"Could not create Neo4j vector index: {e}. Falling back to manual search.")

    def close(self):
        self.driver.close()

    def add_node(self, node: Node) -> Node:
        with self.driver.session() as session:
            session.execute_write(self._create_node, node)
        return node

    def add_nodes_batch(self, nodes: List[Node]) -> None:
        if not nodes: return
        query = """
        UNWIND $nodes as node_data
        CREATE (n:Belief {
            id: node_data.id,
            user_id: node_data.user_id,
            label: node_data.label,
            confidence: node_data.confidence,
            embedding: node_data.embedding,
            deprecated: node_data.deprecated,
            created_at: datetime(node_data.created_at)
        })
        """
        data = [n.dict() for n in nodes]
        for d in data:
            d['id'] = str(d['id'])
            d['created_at'] = d['created_at'].isoformat()
        with self.driver.session() as session:
            session.run(query, nodes=data)

    def get_node(self, node_id: UUID) -> Optional[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._find_node, str(node_id))

    def get_nodes_batch(self, node_ids: List[UUID]) -> List[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._find_nodes_batch, [str(nid) for nid in node_ids])

    def search_nearest_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._vector_search, embedding, user_id, k, False)

    def search_deprecated_nodes(self, embedding: List[float], user_id: str, k: int = 5) -> List[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._vector_search, embedding, user_id, k, True)

    def add_edge(self, edge: Edge) -> Edge:
        with self.driver.session() as session:
            session.execute_write(self._create_edge, edge)
        return edge

    def add_edges_batch(self, edges: List[Edge]) -> None:
        if not edges: return
        query = """
        UNWIND $edges as edge_data
        MATCH (a:Belief {id: edge_data.from_node_id})
        MATCH (b:Belief {id: edge_data.to_node_id})
        CREATE (a)-[r:RELATED {
            id: edge_data.id,
            relation: edge_data.relation,
            weight: edge_data.weight,
            created_at: datetime(edge_data.created_at)
        }]->(b)
        """
        data = [e.dict() for e in edges]
        for d in data:
            d['id'] = str(d['id'])
            d['from_node_id'] = str(d['from_node_id'])
            d['to_node_id'] = str(d['to_node_id'])
            d['created_at'] = d['created_at'].isoformat()
        with self.driver.session() as session:
            session.run(query, edges=data)

    def get_edges(self, node_id: UUID) -> List[Edge]:
        with self.driver.session() as session:
            return session.execute_read(self._find_edges, str(node_id))

    def get_edges_batch(self, node_ids: List[UUID]) -> Dict[UUID, List[Edge]]:
        with self.driver.session() as session:
            return session.execute_read(self._find_edges_batch, [str(nid) for nid in node_ids])

    def update_node(self, node: Node) -> Node:
        with self.driver.session() as session:
            session.execute_write(self._update_node_tx, node)
        return node

    def update_edge(self, edge: Edge) -> Edge:
        with self.driver.session() as session:
            session.execute_write(self._update_edge_tx, edge)
        return edge

    def update_edges_batch(self, edges: List[Edge]) -> List[Edge]:
        with self.driver.session() as session:
            session.execute_write(self._update_edges_batch_tx, edges)
        return edges

    def traverse_graph(self, start_node_ids: List[UUID], depth: int, user_id: str) -> List[Node]:
        """Performs a multi-hop graph walk in a SINGLE database round-trip."""
        if not start_node_ids: return []
        query = f"""
        MATCH (start:Belief)
        WHERE start.id IN $start_ids AND start.user_id = $user_id
        MATCH (start)-[r:RELATED*1..{depth}]->(neighbor:Belief)
        WHERE neighbor.user_id = $user_id AND neighbor.deprecated = false
        RETURN DISTINCT neighbor
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query, start_ids=[str(nid) for nid in start_node_ids], user_id=user_id)
            return [self._record_to_node(record["neighbor"]) for record in result]

    def list_nodes(self, user_id: str, limit: int = 100) -> List[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._list_nodes_tx, user_id, limit)

    def get_nodes_by_entity(self, entity: str, user_id: str) -> List[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._get_nodes_by_entity_tx, entity, user_id)

    def get_stale_nodes(self, user_id: str, days: int = 30, conf_threshold: float = 0.35) -> List[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._get_stale_tx, user_id, days, conf_threshold)

    def get_contradiction_pairs(self, user_id: str) -> List[tuple]:
        with self.driver.session() as session:
            return session.execute_read(self._get_contradictions_tx, user_id)

    def get_nodes_for_strengthening(self, user_id: str, min_evidence: int = 5) -> List[Node]:
        with self.driver.session() as session:
            return session.execute_read(self._get_strengthening_tx, user_id, min_evidence)

    # --- Transaction Functions ---

    @staticmethod
    def _create_node(tx, node: Node):
        tx.run("""
            CREATE (n:Belief {
                id: $id, user_id: $user_id, label: $label, 
                confidence: $confidence, evidence_count: $evidence_count,
                contradiction_count: $contradiction_count,
                temporal_stability: $temporal_stability, 
                abstraction_level: $abstraction_level,
                domain_tags: $domain_tags, entities: $entities,
                deprecated: $deprecated, embedding: $embedding,
                created_at: $created_at, last_confirmed_at: $last_confirmed_at
            })
        """, id=str(node.id), user_id=node.user_id, label=node.label,
            confidence=node.confidence, evidence_count=node.evidence_count,
            contradiction_count=node.contradiction_count,
            temporal_stability=node.temporal_stability,
            abstraction_level=node.abstraction_level,
            domain_tags=node.domain_tags, entities=node.entities,
            deprecated=node.deprecated, embedding=node.embedding,
            created_at=str(node.created_at), last_confirmed_at=str(node.last_confirmed_at))

    @staticmethod
    def _find_node(tx, node_id: str):
        result = tx.run("MATCH (n:Belief {id: $id}) RETURN n", id=node_id)
        record = result.single()
        if record:
            return Neo4jGraphStore._record_to_node(record['n'])
        return None

    @staticmethod
    def _find_nodes_batch(tx, node_ids: List[str]):
        result = tx.run("MATCH (n:Belief) WHERE n.id IN $ids RETURN n", ids=node_ids)
        return [Neo4jGraphStore._record_to_node(record['n']) for record in result]

    @staticmethod
    def _vector_search(tx, embedding, user_id, k, deprecated):
        result = tx.run("""
            MATCH (n:Belief {user_id: $user_id, deprecated: $deprecated})
            WHERE n.embedding IS NOT NULL
            RETURN n, vector.similarity.cosine(n.embedding, $embedding) AS score
            ORDER BY score DESC LIMIT $k
        """, user_id=user_id, embedding=embedding, k=k, deprecated=deprecated)
        return [Neo4jGraphStore._record_to_node(record['n']) for record in result]

    @staticmethod
    def _create_edge(tx, edge: Edge):
        tx.run("""
            MATCH (a:Belief {id: $from_id}), (b:Belief {id: $to_id})
            CREATE (a)-[r:ASSOCIATION {
                id: $id, relation: $relation, weight: $weight,
                evidence_count: $evidence_count, 
                created_at: $created_at, last_updated_at: $last_updated_at
            }]->(b)
        """, from_id=str(edge.from_node_id), to_id=str(edge.to_node_id),
            id=str(edge.id), relation=edge.relation, weight=edge.weight,
            evidence_count=edge.evidence_count,
            created_at=str(edge.created_at), last_updated_at=str(edge.last_updated_at))

    @staticmethod
    def _find_edges(tx, node_id: str):
        result = tx.run("""
            MATCH (a:Belief)-[r:ASSOCIATION]-(b:Belief)
            WHERE a.id = $id
            RETURN r, startNode(r).id as from_id, endNode(r).id as to_id
        """, id=node_id)
        edges = []
        for record in result:
            r = dict(record['r'])
            edges.append(Edge(
                id=UUID(r['id']),
                from_node_id=UUID(record['from_id']),
                to_node_id=UUID(record['to_id']),
                relation=r.get('relation', 'refines'),
                weight=r.get('weight', 0.5),
                evidence_count=r.get('evidence_count', 1),
            ))
        return edges

    @staticmethod
    def _find_edges_batch(tx, node_ids: List[str]):
        result = tx.run("""
            MATCH (a:Belief)-[r:ASSOCIATION]-(b:Belief)
            WHERE a.id IN $ids
            RETURN r, startNode(r).id as from_id, endNode(r).id as to_id
        """, ids=node_ids)
        
        edges_map = {UUID(nid): [] for nid in node_ids}
        for record in result:
            r = dict(record['r'])
            edge = Edge(
                id=UUID(r['id']),
                from_node_id=UUID(record['from_id']),
                to_node_id=UUID(record['to_id']),
                relation=r.get('relation', 'refines'),
                weight=r.get('weight', 0.5),
                evidence_count=r.get('evidence_count', 1),
            )
            # Add to the correct bucket(s)
            from_id = UUID(record['from_id'])
            to_id = UUID(record['to_id'])
            if from_id in edges_map:
                edges_map[from_id].append(edge)
            if to_id in edges_map:
                edges_map[to_id].append(edge)
        return edges_map

    @staticmethod
    def _update_node_tx(tx, node: Node):
        tx.run("""
            MATCH (n:Belief {id: $id})
            SET n.label = $label, n.confidence = $confidence,
                n.evidence_count = $evidence_count, n.deprecated = $deprecated,
                n.domain_tags = $domain_tags, n.entities = $entities,
                n.temporal_stability = $temporal_stability,
                n.abstraction_level = $abstraction_level,
                n.last_confirmed_at = $last_confirmed_at
        """, id=str(node.id), label=node.label, confidence=node.confidence,
            evidence_count=node.evidence_count, deprecated=node.deprecated,
            domain_tags=node.domain_tags, entities=node.entities,
            temporal_stability=node.temporal_stability,
            abstraction_level=node.abstraction_level,
            last_confirmed_at=str(node.last_confirmed_at))

    @staticmethod
    def _update_edge_tx(tx, edge: Edge):
        tx.run("""
            MATCH ()-[r:ASSOCIATION {id: $id}]->()
            SET r.weight = $weight, r.evidence_count = $evidence_count,
                r.last_updated_at = $last_updated_at
        """, id=str(edge.id), weight=edge.weight,
            evidence_count=edge.evidence_count,
            last_updated_at=str(edge.last_updated_at))

    @staticmethod
    def _update_edges_batch_tx(tx, edges: List[Edge]):
        data = [
            {
                'id': str(e.id),
                'weight': e.weight,
                'evidence_count': e.evidence_count,
                'last_updated_at': str(e.last_updated_at)
            } for e in edges
        ]
        tx.run("""
            UNWIND $batch AS data
            MATCH ()-[r:ASSOCIATION {id: data.id}]->()
            SET r.weight = data.weight, 
                r.evidence_count = data.evidence_count,
                r.last_updated_at = data.last_updated_at
        """, batch=data)

    @staticmethod
    def _list_nodes_tx(tx, user_id, limit):
        result = tx.run("""
            MATCH (n:Belief {user_id: $user_id, deprecated: false})
            RETURN n LIMIT $limit
        """, user_id=user_id, limit=limit)
        return [Neo4jGraphStore._record_to_node(record['n']) for record in result]

    @staticmethod
    def _get_nodes_by_entity_tx(tx, entity, user_id):
        result = tx.run("""
            MATCH (n:Belief {user_id: $user_id, deprecated: false})
            WHERE $entity IN n.entities
            RETURN n
        """, user_id=user_id, entity=entity)
        return [Neo4jGraphStore._record_to_node(record['n']) for record in result]

    @staticmethod
    def _get_stale_tx(tx, user_id, days, conf_threshold):
        result = tx.run("""
            MATCH (n:Belief {user_id: $user_id, deprecated: false})
            WHERE n.confidence < $threshold
            RETURN n
        """, user_id=user_id, threshold=conf_threshold)
        return [Neo4jGraphStore._record_to_node(record['n']) for record in result]

    @staticmethod
    def _get_contradictions_tx(tx, user_id):
        result = tx.run("""
            MATCH (a:Belief {user_id: $user_id, deprecated: false})
                  -[r:ASSOCIATION {relation: 'contradicts'}]->
                  (b:Belief {deprecated: false})
            RETURN a, b, r.id as edge_id
        """, user_id=user_id)
        results = []
        for record in result:
            node_a = Neo4jGraphStore._record_to_node(record['a'])
            node_b = Neo4jGraphStore._record_to_node(record['b'])
            results.append((node_a, node_b, UUID(record['edge_id'])))
        return results

    @staticmethod
    def _get_strengthening_tx(tx, user_id, min_evidence):
        result = tx.run("""
            MATCH (n:Belief {user_id: $user_id, deprecated: false})
            WHERE n.evidence_count >= $min_ev AND n.confidence < 0.95
            RETURN n
        """, user_id=user_id, min_ev=min_evidence)
        return [Neo4jGraphStore._record_to_node(record['n']) for record in result]

    @staticmethod
    def _record_to_node(data) -> Node:
        """Convert a Neo4j node record to a Pydantic Node."""
        d = dict(data)
        return Node(
            id=UUID(d['id']),
            user_id=d.get('user_id', ''),
            label=d.get('label', ''),
            confidence=d.get('confidence', 0.5),
            evidence_count=d.get('evidence_count', 1),
            contradiction_count=d.get('contradiction_count', 0),
            domain_tags=d.get('domain_tags', []),
            entities=d.get('entities', []),
            temporal_stability=d.get('temporal_stability', 'stable'),
            abstraction_level=d.get('abstraction_level', 'specific'),
            embedding=d.get('embedding'),
            deprecated=d.get('deprecated', False),
        )
