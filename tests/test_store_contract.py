"""Cross-backend contract tests for GraphStore implementations."""
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

import numpy as np
import pytest
from uuid import uuid4

from neuron.models import Node, Edge, ActivityLog, ActivityType, RetrievalEvent, RetrievalStrategy
from neuron.graph.in_memory_store import InMemoryGraphStore


def _make_store():
    return InMemoryGraphStore()


def test_instantiation_and_vector_search():
    store = _make_store()
    user = "contract_user"
    emb = [1.0] + [0.0] * 383
    node = Node(user_id=user, label="Alpha fact", embedding=emb, confidence=0.8)
    store.add_node(node)
    hits = store.search_nearest_nodes(emb, user, k=3)
    assert len(hits) == 1
    assert hits[0].id == node.id


def test_deprecated_search_and_reactivation_path():
    store = _make_store()
    user = "archive_user"
    emb = [0.0, 1.0] + [0.0] * 382
    archived = Node(user_id=user, label="Old belief", embedding=emb, deprecated=True)
    store.add_node(archived)
    hits = store.search_deprecated_nodes(emb, user, k=1)
    assert len(hits) == 1
    assert hits[0].deprecated is True


def test_contradiction_pairs_and_delete_edge():
    store = _make_store()
    user = "conflict_user"
    a = Node(user_id=user, label="A", confidence=0.9, evidence_count=10)
    b = Node(user_id=user, label="B", confidence=0.4, evidence_count=1)
    store.add_node(a)
    store.add_node(b)
    edge = Edge(from_node_id=a.id, to_node_id=b.id, relation="contradicts", weight=1.0)
    store.add_edge(edge)
    pairs = store.get_contradiction_pairs(user)
    assert len(pairs) == 1
    store.delete_edge(edge.id)
    assert store.get_contradiction_pairs(user) == []


def test_adaptive_hooks_round_trip():
    store = _make_store()
    user = "adaptive_user"
    strategy = RetrievalStrategy(
        id="sys_test",
        user_id="system",
        k_seeds=3,
        traversal_depth=2,
        min_edge_weight=0.5,
    )
    store.add_strategy(strategy)
    strategies = store.get_strategies(user)
    assert any(s.id == "sys_test" for s in strategies)

    store.log_activity(ActivityLog(user_id=user, activity_type=ActivityType.WRITE))
    assert user in store.get_users_needing_maintenance(window_hours=24)

    event = RetrievalEvent(
        user_id=user,
        query="test",
        strategy_id="sys_test",
        nodes_found=2,
        score=0.9,
    )
    store.log_retrieval_event(event)
    loaded = store.get_retrieval_event(event.id)
    assert loaded is not None
    assert loaded.strategy_id == "sys_test"
    events = store.get_retrieval_events(user, limit=5)
    assert len(events) >= 1


def test_traverse_and_list_nodes():
    store = _make_store()
    user = "graph_user"
    n1 = Node(user_id=user, label="One", embedding=[1.0] + [0.0] * 383)
    n2 = Node(user_id=user, label="Two", embedding=[0.9, 0.1] + [0.0] * 382)
    store.add_node(n1)
    store.add_node(n2)
    store.add_edge(Edge(from_node_id=n1.id, to_node_id=n2.id, relation="refines", weight=0.9))
    nodes, edges = store.traverse_graph([n1.id], depth=2, user_id=user)
    assert len(nodes) >= 1
    assert len(edges) >= 1
    listed = store.list_nodes(user, limit=10)
    assert len(listed) == 2


def test_postgres_store_is_concrete():
    from neuron.graph.postgres_store import PostgresGraphStore
    assert "search_deprecated_nodes" not in PostgresGraphStore.__abstractmethods__


def test_neo4j_store_is_concrete():
    from neuron.graph.neo4j_store import Neo4jGraphStore
    assert "search_deprecated_nodes" not in Neo4jGraphStore.__abstractmethods__


def _neo4j_store_or_skip():
    from neuron.config import settings
    from neuron.graph.neo4j_store import Neo4jGraphStore
    try:
        return Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
    except Exception as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_make_store, id="in_memory"),
        pytest.param(_neo4j_store_or_skip, id="neo4j"),
    ],
)
def test_store_contract(factory):
    store = factory()
    user = f"contract_{uuid4().hex[:12]}"
    emb = [1.0] + [0.0] * 383
    node = Node(user_id=user, label="Alpha fact", embedding=emb, confidence=0.8)
    store.add_node(node)
    hits = store.search_nearest_nodes(emb, user, k=3)
    assert len(hits) >= 1
    assert hits[0].id == node.id

    archived = Node(user_id=user, label="Old belief", embedding=[0.0, 1.0] + [0.0] * 382, deprecated=True)
    store.add_node(archived)
    dep_hits = store.search_deprecated_nodes([0.0, 1.0] + [0.0] * 382, user, k=1)
    assert dep_hits and dep_hits[0].deprecated is True

    a = Node(user_id=user, label="A", confidence=0.9, evidence_count=10)
    b = Node(user_id=user, label="B", confidence=0.4, evidence_count=1)
    store.add_node(a)
    store.add_node(b)
    edge = Edge(from_node_id=a.id, to_node_id=b.id, relation="contradicts", weight=1.0)
    store.add_edge(edge)
    assert store.get_contradiction_pairs(user)
    store.delete_edge(edge.id)

    strategy = RetrievalStrategy(
        id=f"sys_test_{user[:12]}",
        user_id="system",
        k_seeds=3,
        traversal_depth=2,
        min_edge_weight=0.5,
    )
    store.add_strategy(strategy)
    assert any(s.id == strategy.id for s in store.get_strategies(user))
    store.log_activity(ActivityLog(user_id=user, activity_type=ActivityType.WRITE))
    assert user in store.get_users_needing_maintenance(window_hours=24)
    event = RetrievalEvent(
        user_id=user,
        query="test",
        strategy_id=strategy.id,
        nodes_found=2,
        score=0.9,
    )
    store.log_retrieval_event(event)
    loaded = store.get_retrieval_event(event.id)
    assert loaded is not None
    assert loaded.strategy_id == strategy.id
    assert store.get_retrieval_events(user, limit=5)

    n1 = Node(user_id=user, label="One", embedding=emb)
    n2 = Node(user_id=user, label="Two", embedding=[0.9, 0.1] + [0.0] * 382)
    store.add_node(n1)
    store.add_node(n2)
    store.add_edge(Edge(from_node_id=n1.id, to_node_id=n2.id, relation="refines", weight=0.9))
    nodes, edges = store.traverse_graph([n1.id], depth=2, user_id=user)
    assert len(nodes) >= 1
    assert len(edges) >= 1
    assert len(store.list_nodes(user, limit=10)) >= 2
