from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Settings
    llm_provider: str = "groq" # Options: openai, anthropic, groq, ollama
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    local_llm_model: str = "llama-3.3-70b-versatile"
    default_model: str = "gpt-4o-mini"
    abstraction_model: str = "claude-3-5-sonnet-20240620"
    
    # Embedding Settings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # DB Settings
    abstraction_prompt: str = """
    Distill the input into a high-signal semantic belief. 
    Preserve all Named Entities (Project Codes, People, Companies, Locations).
    
    Return JSON:
    {
        "label": "The core belief/fact",
        "confidence": 0.0-1.0,
        "domain_tags": ["tag1", "tag2"],
        "entities": ["Entity1", "Entity2"],
        "abstraction_level": "specific", "pattern", or "principle",
        "temporal_stability": "stable", "volatile", or "time-bound"
    }
    """
    
    batch_abstraction_prompt: str = """
    Analyze these chunks of text. For each chunk, extract its semantic metadata.
    Preserve all Named Entities (Project Codes, People, Companies, Locations).
    
    Return a JSON array of objects, one for each chunk in the EXACT same order:
    [
        {
            "label": "The core belief/fact for chunk 1",
            "confidence": 0.0-1.0,
            "domain_tags": ["tag1", "tag2"],
            "entities": ["Entity1", "Entity2"],
            "abstraction_level": "specific", "pattern", or "principle",
            "temporal_stability": "stable", "volatile", or "time-bound"
        },
        ...
    ]
    """
    graph_store_type: str = "postgres" # Options: postgres, neo4j, in_memory
    database_url: str = "postgresql://postgres:postgres@localhost:5432/neuron"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    redis_host: str = "localhost"
    
    # Memory Settings
    base_novelty_threshold: float = 0.3
    confirmation_range_start: float = 0.75
    confirmation_range_end: float = 0.90
    
    # Retrieval Settings
    k_seeds: int = 5
    traversal_depth: int = 3
    max_context_nodes: int = 50
    min_edge_weight: float = 0.6
    min_confidence: float = 0.35
    batch_window_size: int = 20
    max_label_length: int = 2000
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
