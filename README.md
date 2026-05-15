<div align="center">
  <img src="logo.png" alt="Neuron Logo" width="200"/>
  <h1>🧠 neuron | Cognitive Memory Layer for AI Agents</h1>
</div>

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

**neuron** is a graph-backed cognitive memory layer for AI agents: it ingests text, abstracts it with an LLM (optional paths skip or batch that step), stores **nodes** (beliefs) and **edges** (relations), and retrieves context with **vector seeds + graph expansion**, optional **adaptive strategy evolution**, and **maintenance / coherence** helpers.

---

## Architecture (how the pieces fit)

High-level data flow:

```mermaid
flowchart TD
    %% 1. Ingestion Paths
    subgraph Ingestion["1. Ingestion Pipelines"]
        UI[User Input / Text Chunks]
        
        subgraph StandardMode["Standard Mode"]
            SF[Surprise Filter]
            SF -- "Matched (High Sim)" --> Confirm[Evidence Increment]
            SF -- "Novel (Low Sim)" --> AE[Abstraction Engine]
            AE -- Structured Metadata --> StoreNode[Create Node]
        end
        
        subgraph BatchMode["Batch Modes (Fast/Deep)"]
            BE[Batch Embedder]
            IDD[Intra-Batch Dedupe]
            GDD{Global Dedupe?}
            EHL[Entity Hub Linking]
            
            BE --> IDD
            IDD --> GDD
            GDD -- "Deep Mode Only" --> EHL
        end
        
        UI --> SF
        UI --> BE
    end

    %% 2. Global Deduplication & Evidence
    subgraph GlobalLogic["2. Global Logic & Evidence"]
        GDD -- Yes --> Match[Search Store for Match]
        Match -- "Found Match" --> EI[Evidence Increment + Entity Merge]
        Match -- "Unique" --> AE2[LLM Metadata Extract]
        GDD -- No --> StoreNode2[Create New Nodes]
        
        Confirm --> GS
        EI -- Update --> GS
        AE2 --> GS
        StoreNode2 --> GS
        EHL -- "Global Entity Search" --> GS
    end

    %% 3. Storage Layer
    subgraph Storage["3. Storage Layer (GraphStore)"]
        GS[(Knowledge Graph)]
        NB[Nodes: Beliefs/Facts]
        EB[Edges: Semantic Relations]
        GS --> NB
        GS --> EB
    end

    %% 4. Retrieval Path
    subgraph Retrieval["4. Retrieval Pipeline"]
        Query[User Query]
        SR[Strategy Registry]
        QE[Query Embedder]
        DR{Deep Recall?}
        GW[Graph Walk / Expansion]
        MY[Myelination: Edge Weight Up]
        CA[Context Assembly]
        
        Query --> SR
        SR -- Pick Strategy --> QE
        QE --> DR
        DR -- Fails --> Archive[Search Deprecated Nodes]
        DR -- Success --> GW
        Archive -- Reactivate --> GW
        GW -- Traversal --> GS
        GW --> MY
        MY -- Persist Weights --> GS
        GW --> CA
        CA --> Result[Final Context]
    end

    %% 5. Adaptive Learning Loop
    subgraph Adaptive["5. Adaptive Learning Loop"]
        Result --> Log[Log Retrieval Event]
        Log --> FB[User Feedback / Score]
        FB --> Registry[Update Strategy Fitness]
        
        subgraph SleepCycle["Sleep Cycle (Evolution & Coherence)"]
            SC[Sleep Scheduler]
            Cons[Sleep Consolidator]
            Evol[Evolution Engine]
            Maint[Memory Maintenance]
            CD[Coherence Daemon]
            
            SC -- Trigger --> Cons
            Cons -- Genetic Select --> Evol
            Evol -- Mutate --> Registry
            Cons --> Maint
            Maint -- Prune / Merge Clusters --> GS
            Cons --> CD
            CD -- Resolve Conflicts --> GS
        end
    end
```

| Layer | Role |
| --- | --- |
| **`Memory`** | Main façade: `process`, `retrieve`, batch ingest, `maintenance`, `add_manual`. Wires `GraphStore`, `Embedder`, `SurpriseFilter`, `AbstractionEngine`, `Retriever`. |
| **`SurpriseFilter`** | Embeds input, compares to nearest stored nodes; decides novelty vs **confirmation** (boost existing node) using `base_novelty_threshold` and `confirmation_range_*`. |
| **`AbstractionEngine`** | Calls configured **LLM provider** with `abstraction_prompt` / `batch_abstraction_prompt`; returns structured `AbstractionResult` (JSON). |
| **`Embedder`** | Local **SentenceTransformers** when no OpenAI key or provider is `groq` / `ollama`; otherwise OpenAI embeddings if key present; else deterministic **mock** vectors. |
| **`Retriever`** | `k_seeds` nearest nodes → graph walk up to `traversal_depth` respecting `min_edge_weight` / `max_context_nodes` → **myelination** (edge weights ↑ on use) → `direct_beliefs`, `related_context`, `unresolved_tensions`. Optional **deep recall**: weak active match can **reactivate** a strong archived node. |
| **`GraphStore`** | Pluggable backend: **Postgres + pgvector**, **Neo4j**, or **in-memory**. |
| **`AdaptiveMemory`** | Extends `Memory`: logs writes, **personalized strategy selection**, retrieval **events** + `event_id`, `submit_feedback`, `force_sleep`, `strategy_report`, `graph_health`, `search_archive`; optional **background `SleepScheduler`**. |
| **`CoherenceDaemon`** | LLM-assisted conflict resolution + stale prune + strengthen; used by **HTTP API** consolidate route. |
| **`SleepConsolidator`** | Offline cycle: implicit scoring of retrieval events, strategy **evolution** when enough events, then `Memory.maintenance`. |

Extension entry points (dependency injection on `Memory.__init__`): pass a custom **`GraphStore`**, **`Embedder`**, or **`AbstractionEngine`** if you implement the same interfaces.

---

## Installation

**From PyPI / local checkout (editable):**

```bash
cd neuron
pip install -e .
```

Requires **Python 3.11+** (see `pyproject.toml`).

**Core runtime deps** include FastAPI, uvicorn, pydantic-settings, psycopg2, pgvector, openai, anthropic, numpy, sentence-transformers, neo4j, etc.

---

## Configuration

Settings load from **environment variables** and optional **`.env`** (via `pydantic-settings`). Names follow the usual rule: field `llm_provider` → env `LLM_PROVIDER` (case-insensitive).

### Exhaustive settings table

| Field / env | Type | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_PROVIDER` | str | `groq` | Abstraction chat provider: `openai`, `anthropic`, `groq`, `ollama`. |
| `OPENAI_API_KEY` | str? | `None` | OpenAI API key (embeddings and/or `openai` provider). |
| `ANTHROPIC_API_KEY` | str? | `None` | Anthropic key for `anthropic` provider. |
| `GROQ_API_KEY` | str? | `None` | Groq key for `groq` provider. |
| `LOCAL_LLM_MODEL` | str | `llama-3.3-70b-versatile` | Model id for Groq/Ollama OpenAI-compatible chat. |
| `DEFAULT_MODEL` | str | `gpt-4o-mini` | Chat model when `LLM_PROVIDER=openai`. |
| `ABSTRACTION_MODEL` | str | `claude-3-5-sonnet-20240620` | Model when `LLM_PROVIDER=anthropic`. |
| `EMBEDDING_MODEL_NAME` | str | `all-MiniLM-L6-v2` | Local HF model name, or must contain `text-embedding` to use as OpenAI embedding model name. |
| `EMBEDDING_DIMENSION` | int | `384` | Vector size (must match embedding model; Neo4j vector index uses this). |
| `ABSTRACTION_PROMPT` | str | (long default in code) | System instructions + JSON shape for **single** chunk abstraction. |
| `BATCH_ABSTRACTION_PROMPT` | str | (long default in code) | Same for **windowed batch** extraction in `process_batch_deep`. |
| `GRAPH_STORE_TYPE` | str | `postgres` | `postgres`, `neo4j`, or `in_memory`. |
| `DATABASE_URL` | str | `postgresql://postgres:postgres@localhost:5432/neuron` | Postgres connection string. |
| `NEO4J_URI` | str | `bolt://localhost:7687` | Neo4j Bolt URI. |
| `NEO4J_USER` | str | `neo4j` | Neo4j user. |
| `NEO4J_PASSWORD` | str | `password` | Neo4j password. |
| `REDIS_HOST` | str | `localhost` | Redis host for Celery broker URL in `neuron.daemon.tasks` (`redis://HOST:6379/0`). |
| `BASE_NOVELTY_THRESHOLD` | float | `0.3` | Surprise filter: `novelty_score >= threshold` → treat as novel (new node path). `novelty_score = 1 - max_cosine_sim`. |
| `CONFIRMATION_RANGE_START` | float | `0.75` | If nearest similarity in `[start, end]`, node ids go to **confirmation** list. |
| `CONFIRMATION_RANGE_END` | float | `0.90` | Upper bound for confirmation band. |
| `K_SEEDS` | int | `5` | Vector nearest neighbors used as retrieval seeds (and surprise search breadth). |
| `TRAVERSAL_DEPTH` | int | `3` | BFS hops for graph context (or native `traverse_graph` on Neo4j). |
| `MAX_CONTEXT_NODES` | int | `50` | Cap on nodes collected during retrieval. |
| `MIN_EDGE_WEIGHT` | float | `0.6` | Edges below this weight are skipped when walking the graph. |
| `MIN_CONFIDENCE` | float | `0.35` | Seeds below this confidence are omitted from `direct_beliefs`. |
| `BATCH_WINDOW_SIZE` | int | `20` | Default LLM window size in `process_batch_deep`. |
| `MAX_LABEL_LENGTH` | int | `2000` | Truncate labels on `Node` / `AbstractionResult` validation. |
| `ENABLE_ADAPTIVE_MEMORY` | bool | `false` | Turn on adaptive logging, events, scheduler (with `AdaptiveMemory`). |
| `ADAPTIVE_MODE` | str | `auto` | With `AdaptiveMemory`: `auto` starts `SleepScheduler`; `manual` relies on `force_sleep`; `off` still respects `enable_adaptive_memory` for other hooks (scheduler only starts when `auto`). |
| `MAINTENANCE_INTERVAL_HOURS` | int | `24` | Background sleep cycle interval (also window for “users needing maintenance” in Postgres). |
| `STRATEGY_POPULATION_SIZE` | int | `8` | Intended population cap (registry reads `settings`; stores seed default strategy). |
| `EXPLORATION_RATE` | float | `0.2` | ε in ε-greedy strategy selection. |
| `MIN_EVENTS_BEFORE_EVOLUTION` | int | `50` | Defined in settings; consolidator currently evolves when `len(events) >= 50` (same intent). |
| `JUDGE_LLM_MODEL` | str | `gpt-4o-mini` | Reserved for LLM judging; `ImplicitScorer` currently uses heuristics. |

In code, access the same values as attributes on **`from neuron import settings`** (e.g. `settings.k_seeds`).

---

## How to run

### 1. Local Python (library + demos)

```bash
pip install -e .
# Optional: copy and edit env
copy .env.example .env   # Windows — or create .env manually
python examples/demo.py
python examples/example_usage.py
```

Set **`GRAPH_STORE_TYPE=in_memory`** for zero DB setup (Postgres connection failure also falls back to in-memory when postgres is selected but unreachable).

### 2. Postgres + pgvector (recommended for persistence)

```bash
docker compose up -d db
# Ensure DATABASE_URL matches compose (postgres/postgres, db neuron)
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/neuron
set GRAPH_STORE_TYPE=postgres
```

Postgres `setup()` runs `neuron/graph/schema.sql` (path relative to process **current working directory**; run from repo root).

### 3. HTTP API (FastAPI)

```bash
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/neuron
set GRAPH_STORE_TYPE=postgres
python -m uvicorn neuron.api.server:app --host 0.0.0.0 --port 8000
```

Docker: use **`docker compose`**; ensure the API image command overrides the Dockerfile default if it still points at another module — run explicitly:

```bash
docker compose run --rm api uvicorn neuron.api.server:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Body / params | Response |
| --- | --- | --- | --- |
| POST | `/v1/memory/process` | `{ "text", "user_id" }` | `{ "status", "node" }` — node may be `null` if filtered |
| POST | `/v1/memory/retrieve` | `{ "query", "user_id" }` | `{ "status", "context" }` — shape below |
| POST | `/v1/memory/consolidate/{user_id}` | — | Runs `CoherenceDaemon.run_cycle` |
| GET | `/health` | — | `{ "status": "healthy" }` |

### 4. CLI

```bash
neuron init
```

Creates **`.env.example`** in the current directory with starter keys.

### 5. Neo4j

```bash
set GRAPH_STORE_TYPE=neo4j
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password
```

Neo4j provides **`traverse_graph`** for faster multi-hop reads; Postgres uses layered BFS in Python.

### 6. Celery (optional)

`neuron.daemon.tasks` defines a Celery app with broker `redis://{REDIS_HOST}:6379/0`. Run Redis and wire workers separately if you use this.

---

## Usage modes (ingest)

| Mode | Method | LLM | Typical use |
| --- | --- | --- | --- |
| **Interactive / chat** | `Memory.process(text, user_id)` | Yes (`extract`) | Single utterances; surprise filter + entity/similarity linking. |
| **Fast bulk** | `Memory.process_batch_fast(texts, user_id, deduplication_threshold=0.95, knn_edges=3, global_deduplication=False)` | No | Large corpora: embed all, dedupe by cosine matrix, kNN edges, bulk write. Supports global cross-batch deduplication. |
| **Deep bulk** | `Memory.process_batch_deep(texts, user_id, window_size=None, ..., global_deduplication=False)` | Yes (`batch_extract` per window) | Semantically rich corpora after dedupe; entity hub linking across batch + existing graph. Supports global cross-batch deduplication + entity merge. |
| **Manual seed** | `Memory.add_manual(text, user_id, confidence=0.8)` | Yes | Bypass surprise filter; still abstracts and stores. |

`AdaptiveMemory.process_batch(...)` is an **alias** for `process_batch_fast`. Supports `global_deduplication`.

---

## Retrieval result shape

`Memory.retrieve` / API return:

```python
{
  "direct_beliefs": [{"label", "confidence", "evidence"}, ...],
  "related_context": [{"label", "confidence"}, ...],
  "unresolved_tensions": [{"belief_a", "belief_b", "relation"}, ...],
}
```

With **`ENABLE_ADAPTIVE_MEMORY=true`** and **`AdaptiveMemory.retrieve`**, the dict also includes **`"event_id"`** (string UUID) for `submit_feedback`.

---

## Public Python API (user-facing)

Import from **`neuron`**:

| Symbol | Description |
| --- | --- |
| `Memory` | Primary class — see methods below. |
| `AdaptiveMemory` | Subclass with adaptive features. |
| `Node`, `Edge` | Pydantic models for graph primitives. |
| `settings` | Live `Settings` singleton from env / `.env`. |

### `Memory` methods

| Method | Returns | Summary |
| --- | --- | --- |
| `__init__(store=None, embedder=None, engine=None)` | `Memory` | Default store from `GRAPH_STORE_TYPE`; default embedder/engine from settings. |
| `process(text, user_id)` | `Optional[Node]` | Surprise → maybe confirm/boost → else abstract, add node, add edges. |
| `retrieve(query, user_id)` | `dict` | Hybrid retrieval; see shape above. |
| `add_manual(text, user_id, confidence=0.8)` | `Node` | Skip surprise filter. |
| `process_batch_fast(texts, user_id, ...)` | `dict` | `nodes_added`, `edges_added`, `duplicates_dropped`. Supports `global_deduplication`. |
| `process_batch(texts, user_id, global_deduplication=False)` | `dict` | Alias of `process_batch_fast`. |
| `process_batch_deep(texts, user_id, window_size=None, ..., global_deduplication=False)` | `dict` | Same stats keys as fast batch. |
| `maintenance(user_id, stale_days=30, min_confidence=0.35, consolidate=True)` | `dict` | Prune stale low-confidence nodes; optional entity-cluster consolidation via LLM. |

### `AdaptiveMemory` additional methods

| Method | Returns | Summary |
| --- | --- | --- |
| `retrieve(query, user_id, score_feedback=None)` | `dict` | When adaptive enabled: picks **personalized** strategy, temporarily overrides search parameters, logs event; optional immediate `score_feedback`. |
| `submit_feedback(event_id, score)` | `None` | Updates strategy fitness from stored retrieval event. |
| `force_sleep(user_id)` | `None` | Runs `SleepConsolidator.perform_sleep_cycle`. |
| `strategy_report(user_id=None)` | `list[dict]` | `model_dump()` of each `RetrievalStrategy` (filtered by user if provided). |
| `graph_health(user_id)` | `dict` | `total_nodes`, ratios, `status` in `healthy` / `noisy` / `empty`. |
| `search_archive(query, user_id)` | `list` | Vector search **deprecated** nodes only. |
| `set_user_preferences(user_id, prefs)` | `None` | Set maintenance controls (e.g. `{"pruning": false}`). |
| `get_user_preferences(user_id)` | `dict` | Returns active preferences for the user. |
| `process_batch(texts, user_id, global_deduplication=False)` | `dict` | Alias of `process_batch_fast`. |

`graph_health` and `strategy_report` exist only on **`AdaptiveMemory`**, not on base `Memory`.

---

## Framework integrations

| Module | Class | Behavior |
| --- | --- | --- |
| `neuron.integrations.langchain` | `NEURONMemory` | LangChain `BaseMemory`: `load_memory_variables` calls `retrieve`; `save_context` calls `process` on user input. |
| `neuron.integrations.langgraph` | `NEURONCheckpointer` | Skeleton checkpointer: `get_tuple` / `put` round-trip through `Memory`. |
| `neuron.integrations.llamaindex` | `NEURONMemoryStore` | `get` / `put` bridge to `Memory`. |
| `neuron.integrations.crewai` | `NEURONSharedMemory` | `save` / `search` for crew-scoped `user_id` string (`crew_id`). |

---

## TypeScript HTTP client

`neuron/sdk-ts/memory.ts` exports **`Memory`**: `process`, `retrieve`, `consolidate` against the FastAPI base URL.

---

## GraphStore interface (advanced / custom backends)

Implement `neuron.graph.store_interface.GraphStore` — abstract methods include `setup`, `add_node`, `add_nodes_batch`, `search_nearest_nodes`, `search_deprecated_nodes`, edge CRUD, `list_nodes`, `get_nodes_by_entity`, adaptive hooks (`log_activity`, `log_retrieval_event`, strategies, `get_retrieval_events`, `get_retrieval_event`, `get_users_needing_maintenance`), plus **`get_stale_nodes`**, and for coherence **`get_contradiction_pairs`**, **`get_nodes_for_strengthening`**. Neo4j and Postgres implement **`traverse_graph`**.

---

## Tests

```bash
pip install -e .
pytest tests/ -q   # if pytest configured; else:
python tests/batch_fast_test.py
python tests/adaptive_test.py
python tests/scale_test.py
```

---

## Modifying behavior for your workload

- **Cheaper ingest**: use `process_batch_fast` or increase `process_batch_deep` `deduplication_threshold` to drop more near-duplicates; enable `global_deduplication` to avoid cross-batch redundancy; increase `window_size` to fewer LLM calls (less precise NER risk).
- **Narrower memory**: raise `BASE_NOVELTY_THRESHOLD` so only more novel text creates new nodes; widen confirmation band to merge more into existing beliefs.
- **Richer retrieval context**: increase `K_SEEDS` and/or `TRAVERSAL_DEPTH` / `MAX_CONTEXT_NODES`; lower `MIN_EDGE_WEIGHT` to follow weaker edges (noisier).
- **Stricter context**: lower depth and seeds; raise `MIN_EDGE_WEIGHT` and `MIN_CONFIDENCE`.
- **Adaptive / scheduled tuning**: set `ENABLE_ADAPTIVE_MEMORY=true`, use `AdaptiveMemory`, choose `ADAPTIVE_MODE=manual` and call `force_sleep` off-peak; tune `EXPLORATION_RATE` and `MAINTENANCE_INTERVAL_HOURS`.
- **Custom prompts**: override `ABSTRACTION_PROMPT` / `BATCH_ABSTRACTION_PROMPT` in `.env` (escape quotes carefully) or subclass `AbstractionEngine`.

---

## Further reading

- **[api_reference.md](api_reference.md)**: Full API documentation with method signatures and parameter details.
- **[documentation.md](documentation.md)**: Technical blueprint (adaptive loop, retrieval walk, internal notes).

---

## License

MIT. Copyright © 2026 Z-Sigma (per project notice).
