"""NLI-based semantic conflict detection for memory entities.

Replaces heuristic keyword matching with a Cross-Encoder NLI model to detect:
- Contradictions (semantic conflicts)
- Entailment (supersede candidates)
- Neutral (independent memories)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from memory_schema import MemoryEntity

# Lazy import for Cross-Encoder to avoid startup penalty
_nli_model = None


def _get_nli_model():
    """Lazy-load the NLI model on first use."""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for NLI conflict detection. "
                "Install it with: pip install sentence-transformers"
            ) from e
    return _nli_model


@dataclass(frozen=True)
class ConflictResult:
    """Result of semantic conflict analysis between two memories."""
    left_id: str
    right_id: str
    conflict_type: Literal["contradiction", "entailment", "neutral"]
    confidence: float
    reason: str
    scores: dict[str, float]  # Raw NLI scores
    latency_ms: float


@dataclass(frozen=True)
class ConflictProposal:
    """Backward-compatible proposal for conflict resolution."""
    left_id: str
    right_id: str
    reason: str
    confidence: float


def detect_conflict_nli(left: MemoryEntity, right: MemoryEntity) -> ConflictResult:
    """
    Detect semantic conflict using NLI Cross-Encoder.
    
    Args:
        left: First memory entity
        right: Second memory entity
    
    Returns:
        ConflictResult with type, confidence, and raw scores
    """
    start_time = time.perf_counter()
    
    # Get model
    model = _get_nli_model()
    
    # Predict NLI relationship
    # Model outputs: [contradiction, entailment, neutral]
    raw_scores = model.predict([(left.content, right.content)])[0]
    
    # Apply softmax to convert logits to probabilities
    import numpy as np
    exp_scores = np.exp(raw_scores - np.max(raw_scores))  # Subtract max for numerical stability
    scores = exp_scores / exp_scores.sum()
    
    # Parse scores
    contradiction_score = float(scores[0])
    entailment_score = float(scores[1])
    neutral_score = float(scores[2])
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Determine conflict type and confidence
    # Use stricter threshold for contradiction to reduce false positives
    if contradiction_score > 0.75 and contradiction_score > entailment_score and contradiction_score > neutral_score:
        conflict_type = "contradiction"
        confidence = contradiction_score
        reason = "semantic_contradiction"
    elif entailment_score > 0.70 and entailment_score > contradiction_score and entailment_score > neutral_score:
        conflict_type = "entailment"
        confidence = entailment_score
        reason = "semantic_entailment"
    else:
        conflict_type = "neutral"
        confidence = max(contradiction_score, entailment_score, neutral_score)
        reason = "independent"
    
    return ConflictResult(
        left_id=left.memory_id,
        right_id=right.memory_id,
        conflict_type=conflict_type,
        confidence=confidence,
        reason=reason,
        scores={
            "contradiction": contradiction_score,
            "entailment": entailment_score,
            "neutral": neutral_score,
        },
        latency_ms=latency_ms,
    )


def detect_conflict(left: MemoryEntity, right: MemoryEntity) -> ConflictProposal | None:
    """
    Backward-compatible conflict detection interface.
    
    Returns ConflictProposal only for contradictions (actual conflicts).
    Returns None for entailment and neutral relationships.
    """
    result = detect_conflict_nli(left, right)
    
    if result.conflict_type == "contradiction":
        return ConflictProposal(
            left_id=result.left_id,
            right_id=result.right_id,
            reason=result.reason,
            confidence=result.confidence,
        )
    
    return None


def find_conflicts(memories: list[MemoryEntity]) -> list[ConflictProposal]:
    """
    Find all conflicts in a list of memories using NLI.
    
    Backward-compatible with the original API.
    """
    proposals: list[ConflictProposal] = []
    for index, left in enumerate(memories):
        for right in memories[index + 1:]:
            proposal = detect_conflict(left, right)
            if proposal is not None:
                proposals.append(proposal)
    return proposals


def find_all_relationships(memories: list[MemoryEntity]) -> list[ConflictResult]:
    """
    Analyze all pairwise relationships (contradiction, entailment, neutral).
    
    Use this for more nuanced analysis beyond conflict detection.
    """
    results: list[ConflictResult] = []
    for index, left in enumerate(memories):
        for right in memories[index + 1:]:
            result = detect_conflict_nli(left, right)
            results.append(result)
    return results


def find_supersede_candidates(memories: list[MemoryEntity]) -> list[tuple[str, str, float]]:
    """
    Find memories that entail others (supersede candidates).
    
    Returns:
        List of (newer_id, older_id, confidence) tuples
    """
    candidates = []
    results = find_all_relationships(memories)
    
    for result in results:
        if result.conflict_type == "entailment" and result.confidence > 0.7:
            # Determine which memory is newer
            left_mem = next(m for m in memories if m.memory_id == result.left_id)
            right_mem = next(m for m in memories if m.memory_id == result.right_id)
            
            if left_mem.created_at > right_mem.created_at:
                candidates.append((result.left_id, result.right_id, result.confidence))
            else:
                candidates.append((result.right_id, result.left_id, result.confidence))
    
    return candidates


def batch_detect_conflicts(
    memories: list[MemoryEntity],
    batch_size: int = 32,
) -> list[ConflictProposal]:
    """
    Batch processing for large memory sets.
    
    Groups pairs into batches for efficient GPU utilization.
    """
    model = _get_nli_model()
    
    # Generate all pairs
    pairs = []
    pair_metadata = []
    
    for index, left in enumerate(memories):
        for right in memories[index + 1:]:
            pairs.append((left.content, right.content))
            pair_metadata.append((left.memory_id, right.memory_id))
    
    if not pairs:
        return []
    
    # Batch predict
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_scores = model.predict(batch)
        all_scores.extend(batch_scores)
    
    # Convert to proposals
    proposals = []
    for (left_id, right_id), scores in zip(pair_metadata, all_scores):
        contradiction_score = float(scores[0])
        if contradiction_score > 0.7:
            proposals.append(
                ConflictProposal(
                    left_id=left_id,
                    right_id=right_id,
                    reason="semantic_contradiction",
                    confidence=contradiction_score,
                )
            )
    
    return proposals
