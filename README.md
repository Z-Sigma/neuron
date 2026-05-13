# 🧠 neuron Cognitive Memory Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Database](https://img.shields.io/badge/Database-Neo4j%20%7C%20Postgres-success)
![License](https://img.shields.io/badge/License-MIT-purple)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

**neuron** is an enterprise-grade, database-agnostic Cognitive Memory Engine designed for Autonomous AI Agents. It simulates human biological memory by automatically abstracting, filtering, connecting, and naturally forgetting data over time.

Instead of a standard Vector DB that just stores isolated text chunks, neuron builds an interconnected **"Semantic Brain"** using Graph databases (Neo4j, Postgres) combined with Vector Search (HNSW) and LLM Intelligence.

---

## 🎯 High-Value Use Cases

By bridging the gap between Vector Retrieval and Graph Traversal, neuron enables entirely new classes of AI applications:

*   **Long-Horizon Autonomous Agents:** Agents that operate for months (like Auto-Researchers or Coding Assistants) need to remember context without exceeding their token limits. neuron's "Synaptic Pruning" ensures they forget useless trivia while retaining core, verified beliefs.
*   **Hyper-Personalized Tutors:** A tutoring bot can use neuron to map out a student's exact knowledge graph, tracking what concepts they understand (direct beliefs) and what concepts they are struggling with (unresolved tensions), allowing for dynamic curriculum adjustments.
*   **Customer Support Oracles:** Support bots can instantly ingest thousands of PDF manuals using the *Fast Batch* workflow. When a user asks a complex troubleshooting question, the bot uses *Graph Traversal* to pull in context from 3 different manuals simultaneously.
*   **Multi-Agent Communication:** Because neuron supports `user_id` isolation, a massive Multi-Agent system can use a single centralized Neo4j cloud instance to store isolated, private memories for thousands of different agents simultaneously.

---

## ⚡ Quick Start

### 1. Installation

**Option A: Install directly into your project (Without cloning)**
You can install neuron directly from GitHub into your Python environment. This keeps your project folder clean:
```bash
pip install git+https://github.com/Z-Sigma/neuron.git
```

**Option B: Clone for Local Development**
If you want to view or edit the neuron source code locally:
```bash
git clone https://github.com/Z-Sigma/neuron.git
cd neuron
pip install -e .
```

### 2. Configuration (`.env`)
neuron is highly modular. Create a `.env` file in your root directory. The system will automatically load these parameters.

**Complete List of Environment Variables:**

| Variable | Description | Default |
| :--- | :--- | :--- |
| `llm_provider` | The AI provider for abstraction (`groq`, `openai`, `anthropic`, `ollama`) | `"groq"` |
| `groq_api_key` | Required if using Groq | `None` |
| `openai_api_key` | Required if using OpenAI | `None` |
| `anthropic_api_key` | Required if using Anthropic | `None` |
| `local_llm_model` | The specific model name to use for abstraction | `"llama-3.3-70b-versatile"` |
| `embedding_model_name` | The model used for vectors (Local HuggingFace or OpenAI) | `"all-MiniLM-L6-v2"` |
| `embedding_dimension` | Vector dimension size (must match the model) | `384` |
| `graph_store_type` | The database backend (`neo4j`, `postgres`, `in_memory`) | `"postgres"` |
| `database_url` | Connection string for Postgres | `"postgresql://..."` |
| `neo4j_uri` | Connection string for Neo4j | `"bolt://localhost:7687"` |
| `neo4j_user` | Username for Neo4j | `"neo4j"` |
| `neo4j_password` | Password for Neo4j | `"password"` |
| `k_seeds` | The number of initial nodes found via Vector Search | `5` |
| `traversal_depth` | How many Graph Hops outward the algorithm should walk | `3` |
| `max_context_nodes` | The absolute maximum number of nodes returned to the LLM | `50` |
| `min_edge_weight` | Minimum edge similarity score required to follow a graph path | `0.6` |

---

## 🧠 Core Methods (The API)

The core interface is the `Memory` object.

```python
from neuron import Memory

# Automatically initializes based on your .env configuration
brain = Memory()
```

### 1. The Chat Workflow (Option 1: Maximum Intelligence)
Use this when you want the AI to read a single sentence, extract the core facts, filter duplicates, and automatically draw semantic graph edges. This is ideal for chatbots or live Agent interaction.

```python
# The Brain processes the sentence, extracts the meaning, embeds it, and stores it.
brain.process(
    text="My secret project is called Stardust.", 
    user_id="user_123"
)
```
**Arguments:**
*   `text` *(str)*: The raw input string to be abstracted.
*   `user_id` *(str)*: A unique identifier for the user (to support multi-tenant graph isolation).

---

### 2. The Bulk Injection Workflow (Option 2.5: Fast Intelligence)
If you have a 500-page PDF and want to upload it instantly, use `process_batch_fast`. It bypasses the slow LLM, computes a local matrix to drop duplicate chunks, automatically generates mathematical k-NN edges, and blasts the raw data into the cloud in seconds.

```python
pdf_chunks = ["Paragraph 1", "Paragraph 2", "Paragraph 3"]

# Pushes thousands of nodes instantly with full Graph Wiring
brain.process_batch_fast(
    texts=pdf_chunks, 
    user_id="user_123",
    deduplication_threshold=0.95, 
    knn_edges=3 
)
```
**Arguments:**
*   `texts` *(List[str])*: A list of raw text chunks to be embedded and stored.
*   `user_id` *(str)*: A unique identifier for the user.
*   `deduplication_threshold` *(float, default=0.95)*: The Cosine Similarity threshold above which a chunk is mathematically considered a duplicate and dropped.
*   `knn_edges` *(int, default=3)*: The number of closest semantic neighbors within the batch that each node should automatically draw an Edge to.

---

### 3. The Retrieval Workflow (Deep Recall)
When you want to search the brain, neuron uses a Hybrid Search. It uses HNSW to find the closest vector "Seed", and then uses Graph algorithms to walk outward, gathering full surrounding context.

```python
context = brain.retrieve(
    query="What is the secret project?", 
    user_id="user_123"
)

print("Direct Matches:")
for node in context["direct_beliefs"]:
    print(f"- {node['label']} (Confidence: {node['confidence']})")

print("Associated Graph Context:")
for node in context["related_context"]:
    print(f"- {node['label']} (Confidence: {node['confidence']})")
```
**Arguments:**
*   `query` *(str)*: The question or context the Agent is searching for.
*   `user_id` *(str)*: A unique identifier for the user.

**Returns (`Dict`):**
*   `direct_beliefs`: A list of the Top K nodes found via Vector Search.
*   `related_context`: A list of nodes discovered by walking across the Graph edges.
*   `unresolved_tensions`: Any directly contradicting facts discovered.

---

### 4. Synaptic Pruning (Lifecycle Maintenance)
Turn your database into a living brain. Run this periodically via a cron job or Celery task. It scans the database, decays the confidence score of unused facts, and eventually "archives" them to save space and context bloat.

```python
# Weakens memories that haven't been used in 30 days
brain.maintenance(
    user_id="user_123", 
    stale_days=30,
    min_confidence=0.35,
    consolidate=True
)
```
**Arguments:**
*   `user_id` *(str)*: A unique identifier for the user.
*   `stale_days` *(int, default=30)*: Number of days since a memory was last recalled before it starts losing confidence points.
*   `min_confidence` *(float, default=0.35)*: The absolute minimum confidence score. If a memory drops below this, it is soft-deleted/archived.
*   `consolidate` *(bool, default=True)*: If True, the engine will attempt to merge highly similar clusters of memories together to free up database space.

---

## 🚀 How to Run the Tests

To verify that the system is fully operational on your specific cloud setup, run the included examples:

**1. The Cognitive Demo (`demo.py`)**
Tests the AI's ability to abstract, forget, and deep-recall specific facts.
```bash
python examples/demo.py
```

**2. The Speed Benchmark (`scale_test.py`)**
Tests your database's speed by generating 2,000 synthetic nodes and 10,000 edges, blasting them into the cloud, and timing the HNSW retrieval.
```bash
python tests/scale_test.py
```

**3. The Option 2.5 Batch Test (`batch_fast_test.py`)**
Tests the high-speed local matrix mathematics to ensure duplicates are accurately dropped and semantic edges are generated locally.
```bash
python tests/batch_fast_test.py
```

---

## 🤝 Collaborations & Copyright

**Copyright © 2026 Z-Sigma.** All rights reserved.

The `neuron` engine is open-source under the MIT License. We actively welcome contributions from the community! If you are a developer or researcher interested in Autonomous Agents, Graph Theory, or Cognitive Architecture, please feel free to:
1. **Fork the repository** and submit Pull Requests.
2. **Open an Issue** for feature requests or bug reports.
3. Reach out directly for enterprise collaborations or integration partnerships.

*Built to push the boundaries of Agentic Memory.*
