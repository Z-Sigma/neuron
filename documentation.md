# neuron — technical documentation

This document mirrors the **current codebase** (Python package `neuron`). For install and quick start, see [README.md](README.md).

---

## 1. Package layout

| Path | Responsibility |
| --- | --- |
| `neuron/config.py` | `Settings` (Pydantic `BaseSettings`) and singleton `settings`. |
| `neuron/memory.py` | `Memory` — ingest, retrieve, maintenance, batch paths. |
| `neuron/adaptive/adaptive_memory.py` | `AdaptiveMemory` — events, feedback, scheduler hook, diagnostics. |
| `neuron/adaptive/adaptive_engine.py` | `StrategyRegistry` — ε-greedy selection, fitness EMA, mutation `evolve(user_id)`. |
| `neuron/adaptive/consolidator.py` | `SleepConsolidator` — implicit scoring, evolution gate, calls `Memory.maintenance`. |
| `neuron/adaptive/scorer.py` | `ImplicitScorer` — heuristic scores today; `judge_with_llm` stub. |
| `neuron/filter/embedder.py` | `Embedder` — local / OpenAI / mock. |
| `neuron/filter/surprise_filter.py` | `SurpriseFilter` — novelty + confirmation lists. |
| `neuron/abstraction/engine.py` | `AbstractionEngine` — `extract`, `batch_extract`, JSON parsing. |
| `neuron/llm/provider.py` | `get_llm_provider`, OpenAI-compatible + Anthropic adapters. |
| `neuron/retrieval/retriever.py` | `Retriever` — seeds, walk, myelination, deep recall reactivation. |
| `neuron/graph/store_interface.py` | `GraphStore` ABC. |
| `neuron/graph/postgres_store.py` | Postgres + pgvector implementation. |
| `neuron/graph/neo4j_store.py` | Neo4j + optional `traverse_graph`. |
| `neuron/graph/in_memory_store.py` | In-process store; full `GraphStore` contract for dev/tests. |
| `neuron/daemon/coherence.py` | `CoherenceDaemon` — conflicts, prune, strengthen. |
| `neuron/daemon/scheduler.py` | `SleepScheduler` — periodic `force_sleep` per user. |
| `neuron/daemon/tasks.py` | Celery app + `run_coherence_cycle` task. |
| `neuron/api/server.py` | FastAPI app. |
| `neuron/cli.py` | `neuron init` entry point. |
| `neuron/integrations/*` | LangChain, LangGraph, LlamaIndex, CrewAI bridges. |
| `neuron/sdk-ts/memory.ts` | Minimal HTTP client. |

---

## 2. Data models (`neuron.models`)

### `Node`

Belief vertex: `id`, `user_id`, `label`, `confidence`, `evidence_count`, `contradiction_count`, timestamps, `domain_tags`, `entities`, `temporal_stability` (`stable` \| `volatile` \| `time-bound`), `abstraction_level` (`specific` \| `pattern` \| `principle`), `embedding`, `deprecated`.

Labels are truncated to `settings.max_label_length` in validators.

### `Edge`

`from_node_id`, `to_node_id`, `relation` (`supports` \| `contradicts` \| `refines` \| `depends_on` \| `derived_from` \| `temporal_successor` \| `unresolved_tension`), `weight`, `evidence_count`, timestamps.

### Adaptive adjuncts

- `ActivityType`, `ActivityLog` — write/read/maintenance audit.
- `RetrievalStrategy` — `k_seeds`, `traversal_depth`, `min_edge_weight`, `fitness_score`, `generations_survived`, `parent_id`.
- `RetrievalEvent` — one logged retrieval for feedback and evolution.

---

## 3. Ingestion pipelines (exact behavior)

### 3.1 `Memory.process(text, user_id)`

1. `SurpriseFilter.evaluate` embeds text, finds `k_seeds` nearest nodes, computes cosine sims (dot product; assumes normalized embeddings).
2. **Novelty**: `novelty_score = 1 - max_sim`. If `novelty_score >= base_novelty_threshold` → `is_novel` true (subject to having neighbors).
3. **Confirmation band**: for each neighbor with similarity in `[confirmation_range_start, confirmation_range_end]`, that node’s id is a **confirmation target**; `process` increments `evidence_count`, bumps confidence by `+0.05` capped at `0.95`, updates `last_confirmed_at`.
4. If **novel**: `AbstractionEngine.extract` → new `Node` with embedding of abstracted **label**; `add_node`; optional **entity** linking (embed entity string, nearest nodes, substring check on label); semantic edges to top two nearest ids from surprise result with weight = similarity.
5. Exceptions are logged; `None` returned on failure.

### 3.2 `Memory.process_batch_fast`

1. `embed_batch` → numpy cosine matrix; greedy dedupe by `deduplication_threshold`.
2. **Global Deduplication** (Optional): If `global_deduplication` is True, search store for nearest node. If `sim >= threshold`, increment `evidence_count` and update `confidence` of the existing node; use its ID for any batch edges.
3. Nodes use **raw chunk text** as `label` (truncated to 2000 chars in code), confidence `1.0` (or updated confidence if merged).
4. For each kept row, top `knn_edges` neighbors by similarity → `Edge(relation="refines", weight=sim)`.
5. `add_nodes_batch` / `add_edges_batch`.

### 3.3 `Memory.process_batch_deep`

Same dedupe as fast, then `AbstractionEngine.batch_extract` per `window_size` (default `settings.batch_window_size`).

**Global Deduplication** (Optional): Similar to fast mode, but also performs **Entity Merging** — new entities discovered in the batch are appended to the existing global node's entities list.

Nodes use abstraction metadata; embeddings from **pre-dedupe** vectors aligned to kept rows.

Edges: batch kNN as fast; **in-batch** entity star links; **global** `get_nodes_by_entity` links to first hub per entity.

---

## 4. Retrieval (`Retriever.retrieve`)

1. Embed query; `search_nearest_nodes` with `settings.k_seeds`.
2. **Deep recall**: if no seeds **or** top seed’s dot similarity `< 0.6`, query `search_deprecated_nodes` (k=1). If archived node similarity `> 0.6`, set `deprecated=False` and **prepend** to seeds.
3. **Graph expansion**: if store has `traverse_graph` (Neo4j), use it once with all seed ids; else iterative BFS using `get_edges_batch`, skipping edges with `weight < min_edge_weight`, skipping deprecated targets, cap `max_context_nodes`.
4. **Myelination**: traversed edges get `weight *= 1.05` (cap 1.0) and `evidence_count += 1`; persisted via `update_edges_batch`.
5. Output assembly: `direct_beliefs` from seeds with `confidence >= min_confidence`; `unresolved_tensions` from `contradicts` / `unresolved_tension` edges among collected nodes; `related_context` for non-seed nodes.

**Note:** `Memory.retrieve` returns the retriever dict with keys: `direct_beliefs`, `related_context`, and `unresolved_tensions`.

---

## 5. Adaptive memory (implementation-accurate)

### 5.1 Enabling

- Set `ENABLE_ADAPTIVE_MEMORY=true`.
- Construct **`AdaptiveMemory`** (not required for base `Memory`).
- If `ADAPTIVE_MODE=auto`, `__init__` starts **`SleepScheduler`** (daemon thread): every `maintenance_interval_hours`, loads users from `get_users_needing_maintenance` and calls `force_sleep(user_id)`.

### 5.2 `AdaptiveMemory.retrieve`

When adaptive is enabled:

1. `StrategyRegistry.select_strategy(user_id)` — ε-greedy over strategies evolved for that specific user.
2. **Passes strategy parameters to `retrieve`** (k_seeds, traversal_depth, min_edge_weight) instead of mutating global settings. This ensure thread-safety.
3. Calls `Memory.retrieve`.
4. Logs `RetrievalEvent` with `nodes_found` = count of `direct_beliefs` + `related_context` entries.
5. Sets `result["event_id"]` when logging; optional `score_feedback` updates fitness immediately.

### 5.3 `submit_feedback`

Scans `get_retrieval_events(None, limit=1000)` globally to find the event, then calls `registry.update_fitness(..., user_id)` to ensure personalization.

### 5.4 `SleepConsolidator.perform_sleep_cycle`

1. Load last 50 retrieval events for user.
2. “Unscored” = `score == 0.0`; `ImplicitScorer.score_events` assigns heuristic scores; updates strategy fitness.
3. If `len(events) >= 50`, `registry.evolve(user_id)` (clone best, mutate, `add_strategy`).
4. Constructs `Memory(store=self.store)` and runs `maintenance(user_id)`.
5. Logs maintenance activity.

### 5.5 `graph_health`

Sample up to 1000 nodes: ratios of high confidence (≥ 0.8) and deprecated; `status` is `empty`, or `healthy` if high-conf ratio > 0.5 else `noisy`.

---

## 6. Maintenance and coherence

### `Memory.maintenance`

- `get_stale_nodes(user_id, days, conf_threshold)` → mark `deprecated`.
- If `consolidate`: clusters nodes sharing entities with ≥5 nodes → LLM “master belief” → new node; deprecate cluster.

**Postgres caveat:** `get_stale_nodes` SQL uses `INTERVAL '%s days'` with a bound parameter; verify dialect correctness in your Postgres version.

### `CoherenceDaemon.run_cycle`

Requires store methods **`get_contradiction_pairs`**, **`get_stale_nodes`** (no extra args vs memory’s call — Neo4j/Postgres must match), **`get_nodes_for_strengthening`**. **In-memory** returns empty for these — coherence is effectively a no-op there.

---

## 7. LLM and embedding matrix

| Provider | Chat endpoint class | Model fields |
| --- | --- | --- |
| `openai` | OpenAI SDK default URL | `DEFAULT_MODEL` |
| `anthropic` | Anthropic SDK | `ABSTRACTION_MODEL` |
| `groq` | OpenAI-compatible @ `api.groq.com` | `LOCAL_LLM_MODEL` |
| `ollama` | OpenAI-compatible @ `localhost:11434` | `LOCAL_LLM_MODEL` |

`Embedder`: if `openai_api_key` is set **and** provider is not `groq`/`ollama`, prefers OpenAI embeddings (`text-embedding-3-small` unless `embedding_model_name` contains `text-embedding`). Otherwise local `sentence-transformers` when installed; else mock hash vectors of length `embedding_dimension`.

---

## 8. HTTP ↔ TypeScript contract

- `POST /v1/memory/process` returns Pydantic `Node` serialized in JSON (UUIDs as strings).
- `POST /v1/memory/retrieve` returns the retriever dict under `context`.
- TS client types `direct_beliefs` as `Node[]` but API returns **dicts** with `label`, `confidence`, `evidence` — adjust TS types if you need strict parity.

---

## 9. Operational checklist

1. Match **`EMBEDDING_DIMENSION`** to the model output dimension.
2. For Postgres, enable **pgvector** and run from repo root so `neuron/graph/schema.sql` resolves.
3. For production adaptive feedback on Postgres, plan a small store change: `get_retrieval_events` by event id or global listing.

---

## 10. Glossary

| Term | Meaning |
| --- | --- |
| Seed | Top vector neighbor(s) for a query or text embedding. |
| Myelination | Increasing edge weight when an edge participates in retrieval traversal. |
| Deep recall | Promoting a strongly matching **archived** node back to active memory during retrieve. |
| Sleep cycle | `SleepConsolidator` / `force_sleep` / scheduler-driven consolidation pass. |
