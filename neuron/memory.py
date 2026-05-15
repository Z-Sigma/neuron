from typing import List, Dict, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone
import logging

from neuron.config import settings
from neuron.models import Node, Edge, SurpriseResult, AbstractionResult
from neuron.filter.embedder import Embedder
from neuron.filter.surprise_filter import SurpriseFilter
from neuron.abstraction.engine import AbstractionEngine
from neuron.graph.store_interface import GraphStore
from neuron.graph.postgres_store import PostgresGraphStore
from neuron.graph.in_memory_store import InMemoryGraphStore
from neuron.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

class Memory:
    def __init__(
        self, 
        store: Optional[GraphStore] = None,
        embedder: Optional[Embedder] = None,
        engine: Optional[AbstractionEngine] = None
    ):
        if store:
            self.store = store
        else:
            self.store = self._init_store()
            
        self.embedder = embedder or Embedder()
        self.engine = engine or AbstractionEngine()
        
        self.filter = SurpriseFilter(self.store, self.embedder)
        self.retriever = Retriever(self.store, self.embedder)

    def maintenance(self, user_id: str, stale_days: int = 30, min_confidence: float = 0.35, consolidate: bool = True):
        """
        Optimizes the graph by pruning stale data and consolidating clusters.
        
        Args:
            user_id: User context.
            stale_days: Number of days since last confirm before pruning (if confidence low).
            min_confidence: Confidence floor below which stale nodes are pruned.
            consolidate: Whether to trigger LLM-based cluster merging.
            
        Returns:
            A dictionary with counts of pruned and consolidated nodes.
        """
        logger.info(f"Starting maintenance for user {user_id}...")
        pruned_count = 0
        consolidated_count = 0
        
        # 1. Conflict Resolution (Core logic)
        contradictions = self.store.get_contradiction_pairs(user_id)
        for node_a, node_b, edge_id in contradictions:
            if node_a.evidence_count >= node_b.evidence_count:
                node_b.deprecated = True
                self.store.update_node(node_b)
            else:
                node_a.deprecated = True
                self.store.update_node(node_a)
            self.store.delete_edge(edge_id)
            pruned_count += 1
            
        # 2. Stale Pruning
        stale_nodes = self.store.get_stale_nodes(user_id, days=stale_days)
        for node in stale_nodes:
            if node.confidence < min_confidence:
                node.deprecated = True
                self.store.update_node(node)
                pruned_count += 1
                
        # 3. Consolidation
        if consolidate:
            candidates = self.store.get_nodes_for_strengthening(user_id)
            if len(candidates) >= 3:
                # Process larger batches for scalability (top 50)
                consolidated_count = self._consolidate_cluster(candidates[:50], user_id)
                
        return {
            "pruned": pruned_count,
            "consolidated": consolidated_count
        }

    def _consolidate_cluster(self, nodes: List[Node], user_id: str) -> int:
        texts = [n.label for n in nodes]
        combined_text = "\n".join(texts)
        master_abstraction = self.engine.extract(f"Synthesize these related memories into one core truth:\n{combined_text}")
        
        with self.store.transaction():
            master_node = Node(
                user_id=user_id,
                label=master_abstraction.label,
                confidence=master_abstraction.confidence,
                evidence_count=sum([n.evidence_count for n in nodes]),
                contradiction_count=sum([n.contradiction_count for n in nodes]),
                domain_tags=list(set(sum([n.domain_tags for n in nodes], []))),
                entities=list(set(sum([n.entities for n in nodes], []))),
                abstraction_level="pattern",
                embedding=self.embedder.embed(master_abstraction.label)
            )
            self.store.add_node(master_node)
            
            all_old_edges = self.store.get_edges_batch([n.id for n in nodes])
            new_edges = []
            seen_targets = set()
            node_ids = [n.id for n in nodes]
            
            for node_id, edges in all_old_edges.items():
                for edge in edges:
                    other_id = edge.to_node_id if edge.from_node_id == node_id else edge.from_node_id
                    if other_id not in node_ids and other_id not in seen_targets:
                        new_edges.append(Edge(
                            from_node_id=master_node.id,
                            to_node_id=other_id,
                            relation=edge.relation,
                            weight=edge.weight,
                            evidence_count=edge.evidence_count
                        ))
                        seen_targets.add(other_id)
            
            if new_edges:
                self.store.add_edges_batch(new_edges)
                
            for node in nodes:
                node.deprecated = True
                self.store.update_node(node)
                
        return len(nodes)

    def _init_store(self) -> GraphStore:
        store_type = settings.graph_store_type.lower()
        if store_type == "postgres": return PostgresGraphStore()
        elif store_type == "neo4j":
            from neuron.graph.neo4j_store import Neo4jGraphStore
            return Neo4jGraphStore(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
        else: return InMemoryGraphStore()

    def process(self, text: str, user_id: str, novelty_threshold: Optional[float] = None) -> Optional[Node]:
        try:
            result: SurpriseResult = self.filter.evaluate(text, user_id, threshold=novelty_threshold)
            for target_id in result.confirmation_targets:
                node = self.store.get_node(target_id)
                if node:
                    node.evidence_count += 1
                    node.confidence = min(node.confidence + 0.05, 0.95)
                    self.store.update_node(node)
            if result.is_novel:
                abstraction: AbstractionResult = self.engine.extract(text)
                new_node = Node(
                    user_id=user_id,
                    label=abstraction.label,
                    confidence=abstraction.confidence,
                    domain_tags=abstraction.domain_tags,
                    temporal_stability=abstraction.temporal_stability,
                    abstraction_level=abstraction.abstraction_level,
                    embedding=self.embedder.embed(abstraction.label)
                )
                self.store.add_node(new_node)
                self._link_node(new_node, abstraction, user_id)
                return new_node
            return None
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return None

    def retrieve(self, query: str, user_id: str, **kwargs) -> Dict:
        return self.retriever.retrieve(query, user_id, **kwargs)

    def add_manual(self, text: str, user_id: str, confidence: float = 0.8) -> Node:
        abstraction = self.engine.extract(text)
        new_node = Node(
            user_id=user_id, label=abstraction.label, confidence=max(confidence, abstraction.confidence),
            domain_tags=abstraction.domain_tags, entities=abstraction.entities,
            temporal_stability=abstraction.temporal_stability, abstraction_level=abstraction.abstraction_level,
            embedding=self.embedder.embed(abstraction.label)
        )
        self.store.add_node(new_node)
        self._link_node(new_node, abstraction, user_id)
        return new_node

    def _link_node(self, node: Node, abstraction: AbstractionResult, user_id: str):
        if abstraction.contradiction_candidates:
            for cand in abstraction.contradiction_candidates:
                candidates = self.store.search_nearest_nodes(node.embedding, user_id, k=3)
                for match in candidates:
                    if match.id != node.id and (cand.lower() in match.label.lower() or match.label.lower() in cand.lower()):
                        self.store.add_edge(Edge(
                            from_node_id=node.id, to_node_id=match.id, relation="contradicts", weight=1.0
                        ))
                        match.contradiction_count += 1
                        self.store.update_node(match)
        if abstraction.entities:
            entity_embeddings = self.embedder.embed_batch(abstraction.entities)
            for i, entity in enumerate(abstraction.entities):
                candidates = self.store.search_nearest_nodes(entity_embeddings[i], user_id, k=5)
                for match in candidates:
                    if match.id != node.id and entity.lower() in match.label.lower():
                        self.store.add_edge(Edge(
                            from_node_id=node.id, to_node_id=match.id, relation="refines", weight=0.9
                        ))

    def process_batch_fast(
        self, 
        texts: List[str], 
        user_id: str, 
        deduplication_threshold: float = 0.95,
        knn_edges: int = 3,
        global_deduplication: bool = False
    ) -> Dict:
        """
        Fast Batch Ingestion (Variant 1.5).
        Groups texts by semantic similarity, creates nodes, and establishes local context edges.
        
        Args:
            texts: List of raw text chunks.
            user_id: Unique identifier for the user.
            deduplication_threshold: Cosine similarity for dropping redundant chunks.
            knn_edges: Number of nearest-neighbor semantic edges per node.
            global_deduplication: If True, checks database for existing similar nodes to increment evidence.
            
        Returns:
            A summary of added nodes, edges, and dropped duplicates.
        """
        if not texts:
            return {"nodes_added": 0, "edges_added": 0, "duplicates_dropped": 0}
            
        logger.info(f"Fast-processing batch of {len(texts)} chunks for user {user_id}...")
        
        # 1. Batch Embed
        import numpy as np
        embeddings = self.embedder.embed_batch(texts)
        emb_matrix = np.array(embeddings)
        
        # 2. Local De-duplication
        # Normalize for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_matrix = emb_matrix / norms
        sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        
        keep_indices = []
        dropped_count = 0
        for i in range(len(texts)):
            is_duplicate = False
            for j in keep_indices:
                if sim_matrix[i, j] >= deduplication_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)
            else:
                dropped_count += 1
                
        # 3. Create Nodes (with Global Deduplication check)
        unique_nodes = []
        node_mapping = {} # index_in_keep_indices -> Node (either new or existing)
        
        for i, idx in enumerate(keep_indices):
            emb = embeddings[idx]
            label = texts[idx]
            
            node = Node(
                user_id=user_id,
                label=label,
                confidence=0.7,
                embedding=emb
            )
            
            if global_deduplication:
                matches = self.store.search_nearest_nodes(emb, user_id, k=1)
                if matches:
                    best_match = matches[0]
                    # Compute dot product (embeddings are normalized by search_nearest_nodes usually, 
                    # but we'll do it manually here for safety)
                    sim = float(np.dot(emb, best_match.embedding))
                    if sim >= deduplication_threshold:
                        # EVIDENCE INCREMENT
                        best_match.evidence_count += 1
                        best_match.last_confirmed_at = datetime.now(timezone.utc)
                        best_match.confidence = min(best_match.confidence + 0.05, 0.95)
                        self.store.update_node(best_match)
                        node_mapping[i] = best_match
                        dropped_count += 1
                        continue
            
            unique_nodes.append(node)
            node_mapping[i] = node
            
        # 4. Local Edge Construction (K-NN within batch)
        edges = []
        if len(keep_indices) > 1:
            for i, idx_a in enumerate(keep_indices):
                node_a = node_mapping[i]
                neighbors = []
                for j, idx_b in enumerate(keep_indices):
                    if i != j:
                        similarity = sim_matrix[idx_a, idx_b]
                        neighbors.append((similarity, j))
                
                neighbors.sort(key=lambda x: x[0], reverse=True)
                
                # Take top K
                for sim, idx_b in neighbors[:knn_edges]:
                    node_b = node_mapping[idx_b]
                    # Avoid self-edges if we merged into the same global node
                    if node_a.id != node_b.id:
                        edges.append(Edge(
                            from_node_id=node_a.id,
                            to_node_id=node_b.id,
                            relation="refines",
                            weight=float(sim)
                        ))
                    
        # 5. Batch Ingestion
        logger.info(f"Pushing {len(unique_nodes)} Nodes and {len(edges)} Edges to Database...")
        if unique_nodes:
            self.store.add_nodes_batch(unique_nodes)
        if edges:
            self.store.add_edges_batch(edges)
            
        return {
            "nodes_added": len(unique_nodes),
            "edges_added": len(edges),
            "duplicates_dropped": dropped_count
        }

    def process_batch_deep(
        self, 
        texts: List[str], 
        user_id: str, 
        window_size: Optional[int] = None,
        deduplication_threshold: float = 0.95,
        knn_edges: int = 3,
        global_deduplication: bool = False
    ) -> Dict:
        """
        Deep Batch Processing (Variant 2.75).
        Combines fast deduplication with windowed LLM extraction for high-signal ingestion.
        
        Args:
            texts: List of raw text chunks.
            user_id: Unique identifier for the user.
            window_size: Number of chunks per LLM call.
            deduplication_threshold: Cosine similarity for dropping redundant chunks.
            knn_edges: Number of nearest-neighbor semantic edges per node.
            global_deduplication: If True, performs cross-batch entity merging.
            
        Returns:
            A summary of added nodes, edges, and dropped duplicates.
        """
        if not texts:
            return {"nodes_added": 0, "edges_added": 0, "duplicates_dropped": 0}
            
        window_size = window_size or settings.batch_window_size
        logger.info(f"Deep-processing batch of {len(texts)} chunks for user {user_id}...")
        
        # 1. Batch Embed and De-duplicate (Pure Memory Logic)
        import numpy as np
        embeddings = self.embedder.embed_batch(texts)
        emb_matrix = np.array(embeddings)
        
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_matrix = emb_matrix / norms
        sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        
        keep_indices = []
        dropped_count = 0
        for i in range(len(texts)):
            is_duplicate = False
            for j in keep_indices:
                if sim_matrix[i, j] >= deduplication_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)
            else:
                dropped_count += 1
                
        unique_texts = [texts[i] for i in keep_indices]
        unique_embeddings = [embeddings[i] for i in keep_indices]
        
        # 2. Windowed Metadata Extraction (LLM Logic)
        logger.info(f"Extracting metadata for {len(unique_texts)} chunks (Windows of {window_size})...")
        all_abstractions = []
        for i in range(0, len(unique_texts), window_size):
            window = unique_texts[i : i + window_size]
            abstractions = self.engine.batch_extract(window)
            all_abstractions.extend(abstractions)
            
        # 3. Create Nodes (and check for Global Duplicates)
        new_nodes = []
        node_lookup = {} # i -> Node (either new or global existing)
        
        for i, abstraction in enumerate(all_abstractions):
            node = Node(
                user_id=user_id,
                label=abstraction.label,
                confidence=abstraction.confidence,
                domain_tags=abstraction.domain_tags,
                entities=abstraction.entities,
                temporal_stability=abstraction.temporal_stability,
                abstraction_level=abstraction.abstraction_level,
                embedding=unique_embeddings[i]
            )
            
            if global_deduplication:
                matches = self.store.search_nearest_nodes(node.embedding, user_id, k=1)
                if matches:
                    best_match = matches[0]
                    sim = float(np.dot(node.embedding, best_match.embedding))
                    if sim >= deduplication_threshold:
                        # EVIDENCE INCREMENT + ENTITY MERGE
                        best_match.evidence_count += 1
                        best_match.last_confirmed_at = datetime.now(timezone.utc)
                        best_match.confidence = min(best_match.confidence + 0.05, 0.95)
                        
                        # Link any new entities discovered
                        if node.entities:
                            existing_entities = set(best_match.entities)
                            for ent in node.entities:
                                if ent not in existing_entities:
                                    best_match.entities.append(ent)
                        
                        self.store.update_node(best_match)
                        node_lookup[i] = best_match
                        dropped_count += 1
                        continue

            new_nodes.append(node)
            node_lookup[i] = node
            
        # 4. Construct Associative Edges
        edges = []
        
        # A. K-NN Semantic Edges (within the batch)
        if len(node_lookup) > 1:
            for i, idx_a in enumerate(keep_indices):
                node_a = node_lookup[i]
                neighbors = []
                for j, idx_b in enumerate(keep_indices):
                    if i != j:
                        similarity = sim_matrix[idx_a, idx_b]
                        neighbors.append((similarity, j))
                neighbors.sort(key=lambda x: x[0], reverse=True)
                for sim, j in neighbors[:knn_edges]:
                    node_b = node_lookup[j]
                    # Avoid self-edges if we merged into the same global node
                    if node_a.id != node_b.id:
                        edges.append(Edge(
                            from_node_id=node_a.id,
                            to_node_id=node_b.id,
                            relation="refines",
                            weight=float(sim)
                        ))
                    
        # B. Entity Hub Linking (Crossing sessions)
        entity_map = {}
        for i in range(len(node_lookup)):
            node = node_lookup[i]
            for ent in node.entities:
                if ent not in entity_map: entity_map[ent] = []
                entity_map[ent].append(node)
        
        for entity, nodes in entity_map.items():
            # Local Batch Linking (Star topology to avoid O(N^2))
            if len(nodes) > 1:
                hub_node = nodes[0]
                for other in nodes[1:]:
                    edges.append(Edge(
                        from_node_id=other.id,
                        to_node_id=hub_node.id,
                        relation="refines",
                        weight=0.8
                    ))
            
            # Global Hub Linking
            global_hubs = self.store.get_nodes_by_entity(entity, user_id)
            if global_hubs:
                hub = global_hubs[0]
                for node in nodes:
                    if node.id != hub.id:
                        edges.append(Edge(
                            from_node_id=node.id,
                            to_node_id=hub.id,
                            relation="refines",
                            weight=0.9
                        ))
                        
        # 5. Bulk Store
        logger.info(f"Storing {len(new_nodes)} Nodes and {len(edges)} Edges...")
        if new_nodes:
            self.store.add_nodes_batch(new_nodes)
        if edges:
            self.store.add_edges_batch(edges)
            
        return {
            "nodes_added": len(new_nodes),
            "edges_added": len(edges),
            "duplicates_dropped": dropped_count
        }
