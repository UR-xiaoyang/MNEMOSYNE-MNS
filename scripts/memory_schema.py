"""
Memory entity schema and native feature extraction for MNEMOSYNE-MNS.

This module defines the complete memory entity structure and extracts
system-native features for ranking beyond text-based signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Scope:
    level: Literal["session", "project", "global"]
    session_id: str | None = None
    project_id: str | None = None
    visibility: list[str] = field(default_factory=lambda: ["session", "project", "global"])


@dataclass
class Authority:
    class_: Literal["user_current", "user_persistent", "system_rule", "inferred", "permanent_constraint"]
    priority: int  # 0-100
    source: Literal["user_explicit", "tool_verified", "model_inferred", "imported"]
    reliability: float  # 0.0-1.0
    
    @classmethod
    def default_priority(cls, class_: str) -> int:
        return {
            "user_current": 90,
            "permanent_constraint": 85,
            "system_rule": 80,
            "user_persistent": 70,
            "inferred": 30,
        }.get(class_, 50)


@dataclass
class Validity:
    state: Literal["active", "stale", "conflicted", "superseded", "deprecated"]
    last_verified_at: float | None = None
    stale_after: float | None = None
    superseded_by: str | None = None
    conflict_with: list[str] = field(default_factory=list)


@dataclass
class Retrieval:
    path: str  # e.g., "tech/git/config"
    tags: list[str] = field(default_factory=list)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_version: int = 1


@dataclass
class Usage:
    retrieved_count: int = 0
    cited_count: int = 0
    corrected_count: int = 0
    last_retrieved_at: float | None = None
    last_cited_at: float | None = None
    
    @property
    def citation_rate(self) -> float:
        return self.cited_count / self.retrieved_count if self.retrieved_count > 0 else 0.0
    
    @property
    def correction_rate(self) -> float:
        return self.corrected_count / self.cited_count if self.cited_count > 0 else 0.0


@dataclass
class Egress:
    allowed_destinations: list[str] = field(default_factory=lambda: ["local"])
    sensitivity: Literal["public", "private", "credential", "pii"] = "private"
    redaction_required: bool = False


@dataclass
class Provenance:
    created_by: Literal["user", "agent", "import"]
    import_source: str | None = None
    parent_memory_id: str | None = None


@dataclass
class MemoryEntity:
    """Complete memory entity with system-native metadata."""
    
    memory_id: str
    content: str
    created_at: float
    updated_at: float
    version: int
    
    scope: Scope
    authority: Authority
    validity: Validity
    retrieval: Retrieval
    usage: Usage
    egress: Egress
    provenance: Provenance
    
    def hard_filter(self, query_scope_level: str, query_destination: str, strict_validity: bool = True) -> bool:
        """Check if this memory passes hard filters for a query."""
        # Scope filter
        if query_scope_level not in self.scope.visibility:
            return False
        
        # Egress filter
        if query_destination not in self.egress.allowed_destinations:
            return False
        
        # Validity filter
        if self.validity.state in ["superseded", "deprecated"]:
            return False
        
        if strict_validity and self.validity.state == "conflicted":
            return False
        
        return True


@dataclass
class Query:
    """Query context for memory retrieval."""
    
    text: str
    scope: Scope
    destination: str  # "local" | "cloud_anthropic" | "cloud_openai"
    path: str  # Classification tree path hint
    requires_high_authority: bool = False


def extract_native_features(query: Query, memory: MemoryEntity) -> np.ndarray:
    """
    Extract system-native ranking features from query and memory entity.
    
    Returns:
        features: [14] array of native features
    """
    now = time.time()
    
    def days_since(timestamp: float | None) -> float:
        if timestamp is None:
            return 999.0  # Large sentinel for "never"
        return max(0.0, (now - timestamp) / 86400.0)
    
    features = [
        # Scope (2)
        float(query.scope.level == memory.scope.level),
        float(query.scope.level in memory.scope.visibility),
        
        # Authority (3)
        memory.authority.priority / 100.0,
        memory.authority.reliability,
        float(memory.authority.class_.startswith("user")),
        
        # Validity (4)
        float(memory.validity.state == "active"),
        float(memory.validity.state == "stale"),
        float(memory.validity.state == "conflicted"),
        np.log1p(days_since(memory.validity.last_verified_at)),
        
        # Usage (3)
        memory.usage.citation_rate,
        memory.usage.correction_rate,
        np.log1p(days_since(memory.usage.last_cited_at)),
        
        # Path (2)
        float(query.path.startswith(memory.retrieval.path)),
        float(query.path == memory.retrieval.path),
    ]
    
    return np.asarray(features, dtype=np.float32)


def synthesize_native_metadata_for_alpsbench(memory_text: str, index: int, dataset: str) -> MemoryEntity:
    """
    Synthesize plausible native metadata for AlpsBench memories.
    
    This is a placeholder until real agent runtime provides actual metadata.
    Used for extended baseline training and shadow mode preparation.
    """
    now = time.time()
    
    # Infer scope from content heuristics
    if any(kw in memory_text.lower() for kw in ["session", "current", "this time", "now"]):
        scope_level = "session"
    elif any(kw in memory_text.lower() for kw in ["project", "repository", "codebase"]):
        scope_level = "project"
    else:
        scope_level = "global"
    
    # Infer authority from content
    if any(kw in memory_text.lower() for kw in ["must", "never", "always", "禁止", "必须"]):
        authority_class = "permanent_constraint"
    elif any(kw in memory_text.lower() for kw in ["prefer", "usually", "typically"]):
        authority_class = "user_persistent"
    else:
        authority_class = "inferred"
    
    # Synthetic usage: higher index = retrieved more often (simulates real distribution)
    retrieved_count = max(1, int(np.random.poisson(5.0)))
    cited_count = int(retrieved_count * np.random.beta(2, 3))  # Most memories cited < 100%
    corrected_count = int(cited_count * np.random.beta(1, 9))  # Few corrections
    
    return MemoryEntity(
        memory_id=f"alpsbench_{dataset}_{index:06d}",
        content=memory_text,
        created_at=now - np.random.uniform(0, 90 * 86400),  # 0-90 days ago
        updated_at=now - np.random.uniform(0, 30 * 86400),
        version=1,
        scope=Scope(
            level=scope_level,
            session_id=None if scope_level != "session" else f"session_{index % 10}",
            project_id=None if scope_level == "session" else f"project_{index % 3}",
        ),
        authority=Authority(
            class_=authority_class,
            priority=Authority.default_priority(authority_class),
            source="imported",
            reliability=0.85,  # AlpsBench is human-verified
        ),
        validity=Validity(
            state="active",
            last_verified_at=now - np.random.uniform(0, 7 * 86400),  # Last week
        ),
        retrieval=Retrieval(
            path="imported/alpsbench",
            tags=[dataset],
        ),
        usage=Usage(
            retrieved_count=retrieved_count,
            cited_count=cited_count,
            corrected_count=corrected_count,
            last_retrieved_at=now - np.random.uniform(0, 7 * 86400),
            last_cited_at=now - np.random.uniform(0, 14 * 86400) if cited_count > 0 else None,
        ),
        egress=Egress(
            allowed_destinations=["local", "cloud_anthropic", "cloud_openai"],
            sensitivity="public",  # AlpsBench is public data
        ),
        provenance=Provenance(
            created_by="import",
            import_source="alpsbench",
        ),
    )


def memory_from_api(
    memory_id: str,
    content: str,
    *,
    destination: str = "local",
    scope_level: Literal["session", "project", "global"] = "global",
) -> MemoryEntity:
    """Create deterministic metadata for an API-supplied memory.

    Unlike the offline AlpsBench synthesizer, this constructor never samples
    usage or timestamps. External IDs remain stable so API responses can map
    ranked results back to the caller's memory records.
    """
    now = time.time()
    return MemoryEntity(
        memory_id=memory_id,
        content=content,
        created_at=now,
        updated_at=now,
        version=1,
        scope=Scope(level=scope_level),
        authority=Authority(
            class_="inferred",
            priority=30,
            source="imported",
            reliability=0.5,
        ),
        validity=Validity(state="active", last_verified_at=now),
        retrieval=Retrieval(path="api/input", tags=["api"]),
        usage=Usage(),
        egress=Egress(allowed_destinations=[destination], sensitivity="private"),
        provenance=Provenance(created_by="import", import_source="api"),
    )
