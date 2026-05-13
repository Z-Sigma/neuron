-- SQL Schema for NEURON
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS nodes (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    label VARCHAR(150) NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_confirmed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_contradicted_at TIMESTAMP WITH TIME ZONE,
    domain_tags TEXT[],
    entities TEXT[],
    temporal_stability TEXT CHECK (temporal_stability IN ('stable', 'volatile', 'time-bound')),
    abstraction_level TEXT CHECK (abstraction_level IN ('specific', 'pattern', 'principle')),
    embedding vector(384),
    deprecated BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS edges (
    id UUID PRIMARY KEY,
    from_node_id UUID REFERENCES nodes(id),
    to_node_id UUID REFERENCES nodes(id),
    relation TEXT NOT NULL,
    weight FLOAT NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS nodes_vector_idx ON nodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS nodes_user_idx ON nodes(user_id);
CREATE INDEX IF NOT EXISTS edges_from_idx ON edges(from_node_id);
CREATE INDEX IF NOT EXISTS edges_to_idx ON edges(to_node_id);

-- Adaptive Memory Extensions
CREATE TABLE IF NOT EXISTS activity_log (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    details TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_strategies (
    id TEXT PRIMARY KEY,
    k_seeds INTEGER NOT NULL,
    traversal_depth INTEGER NOT NULL,
    min_edge_weight FLOAT NOT NULL,
    fitness_score FLOAT DEFAULT 0.0,
    generations_survived INTEGER DEFAULT 0,
    parent_id TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    strategy_id TEXT REFERENCES retrieval_strategies(id),
    nodes_found INTEGER NOT NULL,
    score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS activity_user_idx ON activity_log(user_id);
CREATE INDEX IF NOT EXISTS retrieval_user_idx ON retrieval_events(user_id);
