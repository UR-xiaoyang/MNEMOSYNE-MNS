# MNEMOSYNE-MNS Memory Entity Schema v1

## Overview

This document defines the complete metadata schema for memory entities in the MNEMOSYNE-MNS system. These fields support scope isolation, conflict detection, validity tracking, and authority-based filtering.

## Core Schema

```python
{
  "memory_id": str,           # UUID or deterministic hash
  "content": str,             # Memory text content
  "created_at": float,        # Unix timestamp
  "updated_at": float,        # Unix timestamp
  "version": int,             # Monotonic version counter
  
  # Scope & Visibility
  "scope": {
    "level": str,             # "session" | "project" | "global"
    "session_id": str | None, # Session UUID if level=session
    "project_id": str | None, # Project slug if level=project
    "visibility": list[str],  # Allowed destination scopes
  },
  
  # Authority & Conflict Resolution
  "authority": {
    "class": str,             # "user_current" | "user_persistent" | "system_rule" | "inferred" | "permanent_constraint"
    "priority": int,          # Higher = wins in conflict (0-100)
    "source": str,            # "user_explicit" | "tool_verified" | "model_inferred" | "imported"
    "reliability": float,     # 0.0-1.0, affects write threshold
  },
  
  # Validity & State
  "validity": {
    "state": str,             # "active" | "stale" | "conflicted" | "superseded" | "deprecated"
    "last_verified_at": float | None,  # Unix timestamp of last verification
    "stale_after": float | None,       # Unix timestamp when this becomes stale
    "superseded_by": str | None,       # memory_id that replaced this
    "conflict_with": list[str],        # memory_ids this conflicts with
  },
  
  # Retrieval Metadata
  "retrieval": {
    "path": str,              # Classification tree path, e.g., "tech/git/config"
    "tags": list[str],        # User or system tags
    "embedding_model": str,   # Model used for semantic embedding
    "embedding_version": int, # Embedding model version
  },
  
  # Usage Tracking (for learning)
  "usage": {
    "retrieved_count": int,
    "cited_count": int,
    "corrected_count": int,
    "last_retrieved_at": float | None,
    "last_cited_at": float | None,
  },
  
  # Privacy & Egress Control
  "egress": {
    "allowed_destinations": list[str],  # ["local", "cloud_openai", "cloud_anthropic"]
    "sensitivity": str,                 # "public" | "private" | "credential" | "pii"
    "redaction_required": bool,
  },
  
  # Provenance
  "provenance": {
    "created_by": str,        # "user" | "agent" | "import"
    "import_source": str | None,  # e.g., "alpsbench", "claude_history"
    "parent_memory_id": str | None,  # If this was derived/updated from another
  }
}
```

## Field Definitions

### Scope

**`scope.level`**
- `"session"`: Only visible within the current session
- `"project"`: Visible across all sessions within a project
- `"global"`: Visible everywhere

**`scope.visibility`**
- List of scope levels where this memory can be retrieved
- Example: `["session", "project"]` means retrievable in current session or project, but not globally

### Authority

**`authority.class`**
- `"user_current"`: User instruction in current turn (highest priority for session-scoped conflicts)
- `"user_persistent"`: User preference saved across sessions
- `"system_rule"`: System safety or operational constraint
- `"inferred"`: Model-extracted memory (lowest reliability)
- `"permanent_constraint"`: Immutable project-level rule

**`authority.priority`**
- 0-100 scale
- Default priorities:
  - `user_current`: 90
  - `permanent_constraint`: 85
  - `system_rule`: 80
  - `user_persistent`: 70
  - `inferred`: 30

**`authority.source`**
- How this memory was obtained
- Affects `authority.reliability` and write threshold

**`authority.reliability`**
- 0.0-1.0 confidence score
- Used in write admission: `p_truth > threshold(reliability)`

### Validity

**`validity.state`**
- `"active"`: Currently valid and safe to inject
- `"stale"`: May be outdated; needs verification before use
- `"conflicted"`: Contradicts another memory; requires resolution
- `"superseded"`: Replaced by a newer version
- `"deprecated"`: Marked for deletion

**`validity.stale_after`**
- Optional expiration timestamp
- Example: environment variables, temporary configs

**`validity.superseded_by`**
- Points to the memory_id that replaced this one
- Forms a version chain

**`validity.conflict_with`**
- List of memory_ids that contradict this memory
- Both memories enter `conflicted` state until resolved

### Retrieval

**`retrieval.path`**
- Position in the classification tree
- Used for fast routing: `tech/git/config` vs `personal/preferences/editor`

**`retrieval.embedding_model`**
- Tracks which model generated the semantic embedding
- Supports multi-model coexistence and migration

### Usage Tracking

**`usage.retrieved_count`**
- How many times this memory was returned by retrieval

**`usage.cited_count`**
- How many times the model actually referenced this memory in its response

**`usage.corrected_count`**
- How many times the user corrected this memory after it was cited
- High correction rate → low reliability

### Egress Control

**`egress.allowed_destinations`**
- Controls where this memory can be sent
- `["local"]` means never leave the local machine
- `["local", "cloud_anthropic"]` allows Anthropic models only

**`egress.sensitivity`**
- `"public"`: No restrictions
- `"private"`: User data, not shareable
- `"credential"`: API keys, passwords
- `"pii"`: Personal identifiable information

### Provenance

**`provenance.created_by`**
- Who/what created this memory

**`provenance.import_source`**
- If imported from a dataset or history export

**`provenance.parent_memory_id`**
- Forms update chains: memory A updated to memory B

## Feature Extraction for Ranking

Given a query and candidate memory entity, extract these ranking features:

```python
def extract_native_features(query: Query, memory: MemoryEntity) -> dict:
    return {
        # Scope match
        "scope_exact_match": query.scope.level == memory.scope.level,
        "scope_compatible": query.scope.level in memory.scope.visibility,
        
        # Authority
        "authority_priority": memory.authority.priority / 100.0,
        "authority_reliable": memory.authority.reliability,
        "authority_is_user": memory.authority.class.startswith("user"),
        
        # Validity
        "validity_is_active": memory.validity.state == "active",
        "validity_is_stale": memory.validity.state == "stale",
        "validity_is_conflicted": memory.validity.state == "conflicted",
        "validity_days_since_verified": days_since(memory.validity.last_verified_at),
        
        # Usage signals
        "usage_citation_rate": safe_div(memory.usage.cited_count, memory.usage.retrieved_count),
        "usage_correction_rate": safe_div(memory.usage.corrected_count, memory.usage.cited_count),
        "usage_recency_days": days_since(memory.usage.last_cited_at),
        
        # Path agreement
        "path_prefix_match": query.path.startswith(memory.retrieval.path),
        "path_exact_match": query.path == memory.retrieval.path,
        
        # Egress compatibility
        "egress_compatible": query.destination in memory.egress.allowed_destinations,
    }
```

## Hard Filters (Pre-Ranking)

These conditions must be satisfied before a memory enters the ranking pool:

```python
def hard_filter(query: Query, memory: MemoryEntity) -> bool:
    # Scope filter
    if not memory.scope.visibility or query.scope.level not in memory.scope.visibility:
        return False
    
    # Egress filter
    if query.destination not in memory.egress.allowed_destinations:
        return False
    
    # Validity filter (configurable strictness)
    if memory.validity.state in ["superseded", "deprecated"]:
        return False
    
    # Authority filter for certain query types
    if query.requires_high_authority and memory.authority.priority < 70:
        return False
    
    return True
```

## Next Steps

1. Implement schema in Python dataclasses or Pydantic models
2. Add native feature extraction to ranking pipeline
3. Train extended hybrid ranker with 13 original + 14 native = 27 features
4. Define hard filter policy for shadow mode
5. Design trace collection schema for `retrieved → cited → corrected`

## References

- AlpsBench does not include these fields; they must be synthesized for shadow mode
- Real production deployment will populate these fields from agent runtime state
