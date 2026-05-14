# NEURON API Reference

This document provides a comprehensive guide to the NEURON framework's Python API.

---

## 1. The `Memory` Class
The base engine for ingestion and retrieval. Use this for standard graph-backed memory.

### `__init__`
Initializes the memory engine with specific backends.

**Parameters:**
*   `store` (Optional[GraphStore]): A custom database backend. Defaults to `settings.graph_store_type`.
*   `embedder` (Optional[Embedder]): A custom embedding engine. Defaults to standard OpenAI/Local transformer.
*   `engine` (Optional[AbstractionEngine]): A custom LLM abstraction engine.

**Example:**
```python
from neuron import Memory
memory = Memory()
```

---

### `process`
The primary ingestion method. Filters for novelty, abstracts text, and stores beliefs.

**Parameters:**
*   `text` (str): Raw input text to remember.
*   `user_id` (str): Unique identifier for data isolation.

**Returns:** `Optional[Node]` — The new belief node, or `None` if the information was filtered as redundant.

**Example:**
```python
node = memory.process("Project Apollo started in 1961.", user_id="user_123")
if node:
    print(f"New belief stored: {node.label}")
```

---

### `retrieve`
Performs hybrid vector-graph retrieval.

**Parameters:**
*   `query` (str): The search term or question.
*   `user_id` (str): The user context to search within.
*   `**kwargs`: Optional strategy overrides:
    *   `k_seeds` (int): Number of initial vector matches.
    *   `traversal_depth` (int): Number of hops in the graph.
    *   `min_edge_weight` (float): Minimum strength of relationship to follow.

**Returns:** `Dict` containing:
*   `direct_beliefs`: Strongest semantic matches.
*   `related_context`: Neighbors found via graph walk.
*   `unresolved_tensions`: Conflicting pairs found in context.

**Example:**
```python
context = memory.retrieve("Tell me about Project Apollo", user_id="user_123", traversal_depth=2)
for belief in context['direct_beliefs']:
    print(belief['label'])
```

---

### `process_batch_fast`
High-speed bulk ingestion bypassing the LLM.

**Parameters:**
*   `texts` (List[str]): List of raw chunks.
*   `user_id` (str): Owner ID.
*   `deduplication_threshold` (float): Cosine similarity at which to drop duplicates (default 0.95).
*   `knn_edges` (int): Number of semantic edges to create per node.

**Example:**
```python
results = memory.process_batch_fast(corpus_chunks, user_id="corp_user")
print(f"Added {results['nodes_added']} nodes.")
```

---

### `maintenance`
Optimizes the graph by pruning stale data and consolidating entities.

**Parameters:**
*   `user_id` (str): User to optimize.
*   `stale_days` (int): Days of inactivity before pruning (default 30).
*   `min_confidence` (float): Confidence floor for pruning (default 0.35).
*   `consolidate` (bool): Whether to merge entity-clusters (default True).

**Example:**
```python
memory.maintenance(user_id="user_123", stale_days=15)
```

---

## 2. The `AdaptiveMemory` Class
Extends `Memory` with personalized, self-optimizing retrieval strategies and automated "sleep" cycles. Strategies are evolved per-user, ensuring that search parameters are optimized for each individual's unique data graph.

### `retrieve` (Adaptive)
Same signature as `Memory.retrieve`, but automatically selects the best evolved strategy for the specific user.

**Additional Parameters:**
*   `score_feedback` (Optional[float]): Immediate quality score (0.0 - 1.0).

**Example:**
```python
from neuron import AdaptiveMemory
adaptive = AdaptiveMemory()

# Picks best strategy automatically
result = adaptive.retrieve("Complex query", user_id="user_123")
```

---

### `submit_feedback`
Trains the system by providing delayed feedback on a specific search.

**Parameters:**
*   `event_id` (str): The UUID from a previous `retrieve` result.
*   `score` (float): Success score (0.0 to 1.0).

**Example:**
```python
# retrieve() returns an "event_id" in its result
adaptive.submit_feedback(event_id="uuid-from-result", score=0.95)
```

---

### `graph_health`
Diagnostic tool to check for noise or fragmentation.

**Parameters:**
*   `user_id` (str): User to check.

**Returns:** `Dict` with `status` ("healthy", "noisy", "empty"), `confidence_ratio`, and `prune_ratio`.

**Example:**
```python
health = adaptive.graph_health("user_123")
print(f"Status: {health['status']}")
```

---

### `force_sleep`
Manually triggers the consolidation and strategy evolution pass.

**Example:**
```python
adaptive.force_sleep("user_123")
```

---

### `strategy_report`
Returns the current state of evolved strategies.

**Parameters:**
*   `user_id` (Optional[str]): If provided, only returns strategies for that user.

**Returns:** `List[Dict]` — A list of serialized `RetrievalStrategy` objects.

**Example:**
```python
report = adaptive.strategy_report("user_123")
for s in report:
    print(f"ID: {s['id']}, Fitness: {s['fitness_score']}")
```

---

## 3. Data Models (`neuron.models`)

### `Node`
*   `label`: The distilled text belief.
*   `confidence`: 0.0 to 1.0 strength.
*   `domain_tags`: List of associated topics.
*   `entities`: Named entities extracted.
*   `abstraction_level`: "specific", "pattern", or "principle".

### `Edge`
*   `relation`: "supports", "contradicts", "refines", "depends_on", etc.
*   `weight`: Strength of the connection (0.0 to 1.0).

---

## 4. Operational Settings
Configure via `.env` or Environment Variables.

| Env Var | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | `openai`, `anthropic`, `groq`, `ollama`. |
| `GRAPH_STORE_TYPE` | `postgres` | `postgres`, `neo4j`, or `in_memory`. |
| `ENABLE_ADAPTIVE_MEMORY` | `false` | Set to `true` to enable self-optimization. |
| `K_SEEDS` | `5` | Default number of vector neighbors to find. |
| `TRAVERSAL_DEPTH` | `3` | Default hops for context discovery. |
| `MAX_LABEL_LENGTH` | `2000` | Max characters for a distilled belief. |
