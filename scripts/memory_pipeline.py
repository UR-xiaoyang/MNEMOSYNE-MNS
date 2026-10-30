"""
End-to-end memory retrieval and ranking pipeline for MNEMOSYNE-MNS.

Usage:
    from memory_pipeline import MemoryPipeline
    
    pipeline = MemoryPipeline(ranker_path="runs/rank_hybrid_v1")
    results = pipeline.retrieve_and_rank(
        query_text="如何配置代理？",
        candidate_memories=memory_pool,
        scope_level="session",
        destination="local",
        top_k=5
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_schema import MemoryEntity, Query, Scope, extract_native_features
from rank_inference import RankingBaseline, resolve_model_path
from structured_features import build_structured_features
from context_assembly import assemble_context as budgeted_assemble_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MemoryPipeline:
    """
    Complete memory retrieval pipeline:
    1. Hard filters (scope, egress, validity)
    2. Feature extraction (semantic + structured + native)
    3. Ranking
    4. Top-K selection
    5. Budget-aware assembly
    """
    
    def __init__(
        self,
        ranker_path: str | Path,
        device: str = "cpu",
        strict_validity: bool = False,
    ):
        """
        Initialize pipeline with a trained ranker.
        
        Args:
            ranker_path: Path to model directory (contains metadata.json and artifacts)
            device: "cpu" or "cuda"
            strict_validity: If True, filter out "stale" and "conflicted" memories
        """
        self.ranker = RankingBaseline(resolve_model_path(ranker_path), device=device)
        self.strict_validity = strict_validity
    
    def retrieve_and_rank(
        self,
        query_text: str,
        candidate_memories: list[MemoryEntity],
        scope_level: str = "global",
        destination: str = "local",
        path_hint: str = "",
        requires_high_authority: bool = False,
        top_k: int = 10,
        extra_features: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve and rank memories for a query.
        
        Args:
            query_text: User's query
            candidate_memories: Pool of candidate MemoryEntity objects
            scope_level: "session" | "project" | "global"
            destination: "local" | "cloud_anthropic" | etc
            path_hint: Optional classification path hint
            requires_high_authority: Filter for high-authority memories only
            top_k: Return top K results
            extra_features: Optional [N x 12] structured features (if None, will be zeros)
        
        Returns:
            List of ranked memory dicts with keys:
                - memory_id, content, rank, score
                - filter_passed, injected
                - features (semantic, lexical, native)
        """
        query = Query(
            text=query_text,
            scope=Scope(level=scope_level),
            destination=destination,
            path=path_hint,
            requires_high_authority=requires_high_authority,
        )
        
        # Step 1: Hard filters
        filtered_candidates = []
        for memory in candidate_memories:
            if memory.hard_filter(scope_level, destination, self.strict_validity):
                if requires_high_authority and memory.authority.priority < 70:
                    continue
                filtered_candidates.append(memory)
        
        if not filtered_candidates:
            return []
        
        # Step 2: Feature extraction
        queries = [query_text] * len(filtered_candidates)
        memories = [m.content for m in filtered_candidates]
        
        # Compute the same 12-column structured feature contract used by the
        # trained hybrid/tree rankers when callers do not supply it.
        if extra_features is None:
            extra_features, _ = build_structured_features(
                queries,
                memories,
                vectorizer=getattr(self.ranker, "vectorizer", None),
            )
        else:
            extra_features = np.asarray(extra_features, dtype=np.float32)
            if extra_features.shape != (len(filtered_candidates), 12):
                raise ValueError("extra_features must have shape [candidate_count, 12]")
        native_features = np.asarray([
            extract_native_features(query, memory)
            for memory in filtered_candidates
        ], dtype=np.float32)
        
        # Combine: extra_features already includes structured features from feature engineering
        # For models trained with extended features, we need semantic + structured + native
        # The ranker will handle this internally based on its trained feature set
        
        # Step 3: Ranking
        # Note: RankingBaseline.score() expects extra_features = structured features
        # For extended ranker, we need to append native features to extra_features
        if self.ranker.model_id == "MNEMOSYNE-MNS-RANK-EXTENDED-01":
            if extra_features is None:
                raise ValueError("Extended ranker requires explicit 12 structured features plus native features")
            combined_extra = np.concatenate([extra_features, native_features], axis=1)
        else:
            combined_extra = extra_features

        scores = self.ranker.score(queries, memories, combined_extra)
        structured_for_results = extra_features

        
        # Step 4: Rank and select top-K
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for position, idx in enumerate(ranked_indices, start=1):
            memory = filtered_candidates[idx]
            results.append({
                "memory_id": memory.memory_id,
                "content": memory.content,
                "rank": position,
                "score": float(scores[idx]),
                "filter_passed": True,
                "injected": True,  # Will be determined by budget assembly
                "features": {
                    "native": native_features[idx].tolist(),
                    "structured": structured_for_results[idx].tolist() if structured_for_results is not None else [],
                },
                "metadata": {
                    "scope": memory.scope.level,
                    "authority_class": memory.authority.class_,
                    "authority_priority": memory.authority.priority,
                    "validity_state": memory.validity.state,
                },
            })
        
        return results
    
    def assemble_context(
        self,
        ranked_results: list[dict[str, Any]],
        max_tokens: int = 4000,
        tokens_per_memory: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Assemble final context from ranked results with budget constraints.
        
        Simple greedy strategy: take top-ranked until budget exhausted.
        Future: add redundancy detection, conflict resolution.
        
        Args:
            ranked_results: Output from retrieve_and_rank()
            max_tokens: Maximum token budget
            tokens_per_memory: Estimated tokens per memory (simple heuristic)
        
        Returns:
            List of memories to inject, with injected=True
        """
        return budgeted_assemble_context(
            ranked_results,
            max_tokens=max_tokens,
            tokens_per_memory=tokens_per_memory,
        )


def create_pipeline(model_name: str = "hybrid") -> MemoryPipeline:
    """
    Factory function to create pipeline with different models.
    
    Args:
        model_name: "hybrid" | "extended" | "e5" | "tree"
    
    Returns:
        Configured MemoryPipeline
    """
    model_paths = {
        "hybrid": "deploy/model_bundle" if (PROJECT_ROOT / "deploy/model_bundle").is_dir() else "runs/rank_hybrid_v1",
        "extended": "runs/rank_extended_v1",
        "e5": "runs/rank_embedding_e5_v1",
        "tree": "runs/rank_pairwise_tree_v1",
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_paths.keys())}")
    
    return MemoryPipeline(ranker_path=model_paths[model_name])
