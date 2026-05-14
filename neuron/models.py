from typing import List, Optional, Literal, Dict
from enum import Enum
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator

class Node(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    label: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_count: int = 1
    contradiction_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_confirmed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_contradicted_at: Optional[datetime] = None
    domain_tags: List[str] = []
    entities: List[str] = []
    temporal_stability: Literal["stable", "volatile", "time-bound"] = "stable"
    abstraction_level: Literal["specific", "pattern", "principle"] = "specific"
    embedding: Optional[List[float]] = None
    deprecated: bool = False

    @field_validator("label")
    @classmethod
    def validate_label_length(cls, v: str) -> str:
        from neuron.config import settings
        if len(v) > settings.max_label_length:
            return v[:settings.max_label_length] # Truncate gracefully instead of crashing
        return v

class Edge(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    from_node_id: UUID
    to_node_id: UUID
    relation: Literal["supports", "contradicts", "refines", "depends_on", "derived_from", "temporal_successor", "unresolved_tension"]
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_count: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SurpriseResult(BaseModel):
    is_novel: bool
    novelty_score: float
    nearest_node_ids: List[UUID] = []
    nearest_node_similarities: List[float] = []
    contradiction_candidates: List[UUID] = []
    confirmation_targets: List[UUID] = []

class AbstractionResult(BaseModel):
    label: str
    confidence: float
    related_concepts: List[str] = []
    contradiction_candidates: List[str] = []
    domain_tags: List[str] = []
    entities: List[str] = []
    temporal_stability: Literal["stable", "volatile", "time-bound"] = "stable"
    abstraction_level: Literal["specific", "pattern", "principle"] = "specific"

    @field_validator("label")
    @classmethod
    def validate_label_length(cls, v: str) -> str:
        from neuron.config import settings
        if len(v) > settings.max_label_length:
            return v[:settings.max_label_length]
        return v

# --- Adaptive Memory Extensions ---

class ActivityType(str, Enum):
    WRITE = "write"
    READ = "read"
    MAINTENANCE = "maintenance"

class ActivityLog(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    activity_type: ActivityType
    details: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RetrievalStrategy(BaseModel):
    id: str # Human-readable ID like "strategy_v1_mutant_3"
    k_seeds: int = 5
    traversal_depth: int = 3
    min_edge_weight: float = 0.6
    fitness_score: float = 0.0
    generations_survived: int = 0
    parent_id: Optional[str] = None

class RetrievalEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    query: str
    strategy_id: str
    nodes_found: int
    score: float = 0.0 # Feedback score (0.0 - 1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
