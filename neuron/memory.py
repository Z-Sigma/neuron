from typing import List, Dict, Optional
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
    """
    NEURON Cognitive Memory Engine.
    
    This is the primary entry point for the NEURON framework. It coordinates 
    abstraction, storage, and retrieval of beliefs across different database backends.
    """
    def __init__(
        self, 
        store: Optional[GraphStore] = None,
        embedder: Optional[Embedder] = None,
        engine: Optional[AbstractionEngine] = None
    ):
        """
        Initialize the NEURON Memory engine.
        
        Args:
            store: Optional custom GraphStore implementation. Defaults to config choice.
            embedder: Optional custom Embedder implementation.
            engine: Optional custom AbstractionEngine implementation.
        """
        if store:
            self.store = store
        else:
            self.store = self._init_store()
            
        self.embedder = embedder or Embedder()
        self.engine = engine or AbstractionEngine()
        
        self.filter = SurpriseFilter(self.store, self.embedder)
        self.retriever = Retriever(self.store, self.embedder)

    def maintenance(
        self, 
        user_id: str, 
        stale_days: int = 30, 
        min_confidence: float = 0.35,
        consolidate: bool = True
    ) -> Dict:
        """
        Perform cognitive maintenance on the user's memory graph.
        
        This process 'prunes' old, low-confidence beliefs and 'consolidates' 
        clusters of related memories into higher-level abstractions.
        
        Args:
            user_id: The user to perform maintenance for.
            stale_days: Inactivity period before a node is considered 'stale'.
            min_confidence: Threshold below which stale nodes are deprecated.
            consolidate: Whether to perform semantic merging of similar nodes.
            
        Returns:
            A summary of actions taken (nodes pruned, nodes consolidated).
        """
        logger.info(f"Starting maintenance for user {user_id}")
        stats = {"pruned": 0, "consolidated": 0}
        
        # 1. PRUNING: Mark stale, low-confidence nodes as deprecated
        stale_nodes = self.store.get_stale_nodes(user_id, days=stale_days, conf_threshold=min_confidence)
        for node in stale_nodes:
            node.deprecated = True
            self.store.update_node(node)
            stats["pruned"] += 1
            
        # 2. CONSOLIDATION: Merge dense clusters of similar nodes
        if consolidate:
            all_nodes = self.store.list_nodes(user_id, limit=500)
            # Find clusters based on entity overlap
            entity_map = {}
            for node in all_nodes:
                for ent in node.entities:
                    if ent not in entity_map: entity_map[ent] = []
                    entity_map[ent].append(node)
            
            # Consolidate clusters with more than 5 nodes
            for ent, cluster in entity_map.items():
                if len(cluster) >= 5:
                    self._consolidate_cluster(cluster, user_id)
                    stats["consolidated"] += 1
            
        return stats

    def _consolidate_cluster(self, nodes: List[Node], user_id: str):
        """Merges a group of related nodes into a single high-level abstraction."""
        texts = [n.label for n in nodes]
        combined_text = "\n".join(texts)
        
        # Use LLM to create a 'Master Belief'
        master_abstraction = self.engine.abstract(f"Synthesize these related memories into one core truth:\n{combined_text}")
        
        # Create the new Master Node
        master_node = Node(
            user_id=user_id,
            label=master_abstraction.label,
            confidence=master_abstraction.confidence,
            domain_tags=list(set(sum([n.domain_tags for n in nodes], []))),
            entities=list(set(sum([n.entities for n in nodes], []))),
            abstraction_level="pattern",
            embedding=self.embedder.embed(master_abstraction.label)
        )
        self.store.add_node(master_node)
        
        # Archive the old nodes
        for node in nodes:
            node.deprecated = True
            self.store.update_node(node)

    def _init_store(self) -> GraphStore:
        store_type = settings.graph_store_type.lower()
        if store_type == "postgres":
            try:
                return PostgresGraphStore()
            except Exception as e:
                logger.warning(f"Postgres failed: {e}. Falling back to In-Memory.")
                return InMemoryGraphStore()
        elif store_type == "neo4j":
            from neuron.graph.neo4j_store import Neo4jGraphStore
            return Neo4jGraphStore(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password
            )
        else:
            return InMemoryGraphStore()

    def process(self, text: str, user_id: str) -> Optional[Node]:
        """
        Full pipeline: Filter -> Abstract -> Store
        Returns the new Node if created, None if filtered out.
        """
        try:
            # 1. Evaluate Surprise
            result: SurpriseResult = self.filter.evaluate(text, user_id)
            
            # 2. Handle Confirmations
            for target_id in result.confirmation_targets:
                node = self.store.get_node(target_id)
                if node:
                    node.evidence_count += 1
                    node.last_confirmed_at = datetime.now(timezone.utc)
                    node.confidence = min(node.confidence + 0.05, 0.95)
                    self.store.update_node(node)

            # 3. If Novel, Abstract and Store
            if result.is_novel:
                abstraction: AbstractionResult = self.engine.extract(text)
                
                # Create new node
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
                
                # 4. Create Associative Edges
                # A. Entity-based linking: Use embedding search per entity for O(log N) instead of O(N)
                if abstraction.entities:
                    for entity in abstraction.entities:
                        entity_embedding = self.embedder.embed(entity)
                        candidates = self.store.search_nearest_nodes(entity_embedding, user_id, k=10)
                        for match in candidates:
                            if match.id != new_node.id and entity.lower() in match.label.lower():
                                self.store.add_edge(Edge(
                                    from_node_id=new_node.id,
                                    to_node_id=match.id,
                                    relation="refines",
                                    weight=0.9
                                ))

                # B. Semantic linking (Implicit)
                for i, nearest_id in enumerate(result.nearest_node_ids[:2]):
                    similarity = result.nearest_node_similarities[i]
                    if not any(e.to_node_id == nearest_id for e in self.store.get_edges(new_node.id)):
                        self.store.add_edge(Edge(
                            from_node_id=new_node.id,
                            to_node_id=nearest_id,
                            relation="refines",
                            weight=min(1.0, max(0.0, float(similarity)))
                        ))
                
                return new_node
            
            return None
        except Exception as e:
            logger.error(f"Memory processing failed: {e}")
            return None

    def retrieve(self, query: str, user_id: str, **kwargs) -> Dict:
        """
        Perform a graph-based retrieval of relevant context for a query.
        
        This method uses a hybrid approach:
        1. Vector Search to find 'Seed' nodes (starting points).
        2. Graph Walk (BFS) up to traversal_depth to find related context.
        3. Attention Filtering to return only the Top-K most relevant nodes.
        
        Args:
            query: The user's question or topic.
            user_id: Unique identifier to ensure user privacy.
            **kwargs: Strategy overrides (k_seeds, traversal_depth, etc).
            
        Returns:
            A dictionary containing:
                - direct_beliefs: Top vector matches (label, confidence, evidence_count).
                - related_context: Nodes reached via graph traversal from seeds.
                - unresolved_tensions: Contradicting belief pairs from traversed edges.
        """
        return self.retriever.retrieve(query, user_id, **kwargs)

    def add_manual(self, text: str, user_id: str, confidence: float = 0.8) -> Node:
        """
        Manually add a memory, bypassing the surprise filter.
        """
        abstraction = self.engine.extract(text)
        new_node = Node(
            user_id=user_id,
            label=abstraction.label,
            confidence=max(confidence, abstraction.confidence),
            domain_tags=abstraction.domain_tags,
            embedding=self.embedder.embed(abstraction.label)
        )
        self.store.add_node(new_node)
        return new_node

    def process_batch(self, texts: List[str], user_id: str, global_deduplication: bool = False) -> Dict:
        """
        Standard batch processing alias.
        Defaults to process_batch_fast for high-speed ingestion.
        """
        return self.process_batch_fast(texts, user_id, global_deduplication=global_deduplication)

    def process_batch_fast(
        self, 
        texts: List[str], 
        user_id: str, 
        deduplication_threshold: float = 0.95,
        knn_edges: int = 3,
        global_deduplication: bool = False
    ) -> Dict:
        """
        Fast Batch Processing (Option 2.5).
        Bypasses the LLM and DB Surprise Filter. Performs deduplication and semantic
        edge generation purely in local memory using matrix math before bulk ingestion.
        """
        if not texts:
            return {"nodes_added": 0, "edges_added": 0, "duplicates_dropped": 0}
            
        logger.info(f"Fast-processing batch of {len(texts)} chunks...")
        
        # 1. Batch Embed
        import numpy as np
        embeddings = self.embedder.embed_batch(texts)
        emb_matrix = np.array(embeddings)
        
        # Normalize vectors for fast cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10 # Avoid division by zero
        normalized_matrix = emb_matrix / norms
        
        # Compute pairwise cosine similarity matrix
        sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        
        # 2. In-Memory De-duplication
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
                
        logger.info(f"Dropped {dropped_count} duplicates (Threshold {deduplication_threshold})")
        
        # Create Nodes for the unique indices
        unique_nodes = []
        node_mapping = {} # To keep track for edges
        
        for idx in keep_indices:
            node = Node(
                user_id=user_id,
                label=texts[idx][:2000], # Truncate label if it's a massive chunk
                confidence=1.0, # Default high confidence for raw ingestion
                embedding=embeddings[idx]
            )
            
            if global_deduplication:
                # Cross-Batch Deduplication: Check against store
                matches = self.store.search_nearest_nodes(node.embedding, user_id, k=1)
                if matches:
                    best_match = matches[0]
                    # Compute similarity manually (embeddings are normalized)
                    sim = float(np.dot(node.embedding, best_match.embedding))
                    if sim >= deduplication_threshold:
                        # EVIDENCE INCREMENT: Instead of adding, strengthen existing node
                        best_match.evidence_count += 1
                        best_match.last_confirmed_at = datetime.now(timezone.utc)
                        best_match.confidence = min(best_match.confidence + 0.05, 0.95)
                        self.store.update_node(best_match)
                        
                        node_mapping[idx] = best_match
                        dropped_count += 1
                        continue
            
            unique_nodes.append(node)
            node_mapping[idx] = node
            
        # 3. K-NN Semantic Edges
        edges = []
        if len(unique_nodes) > 1:
            for idx_a in keep_indices:
                node_a = node_mapping[idx_a]
                
                # Get similarities for node_a against all OTHER unique nodes
                neighbors = []
                for idx_b in keep_indices:
                    if idx_a != idx_b:
                        similarity = sim_matrix[idx_a, idx_b]
                        neighbors.append((similarity, idx_b))
                        
                # Sort by highest similarity
                neighbors.sort(key=lambda x: x[0], reverse=True)
                
                # Take top K
                for sim, idx_b in neighbors[:knn_edges]:
                    node_b = node_mapping[idx_b]
                    edges.append(Edge(
                        from_node_id=node_a.id,
                        to_node_id=node_b.id,
                        relation="refines",
                        weight=float(sim) # Convert numpy float to python float
                    ))
                    
        # 4. Batch Ingestion
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
            
        Returns:
            A summary of added nodes, edges, and dropped duplicates.
        """
        if not texts:
            return {"nodes_added": 0, "edges_added": 0, "duplicates_dropped": 0}
            
        window_size = window_size or settings.batch_window_size
        logger.info(f"Deep-processing batch of {len(texts)} chunks...")
        
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
