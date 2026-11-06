"""
Shared feature extraction for RANK Enhanced V2.

Extracted from train_rank_enhanced_v2.py for use in production inference.
"""
import numpy as np
from typing import List, Dict


def extract_enhanced_features(
    query_texts: List[str],
    memory_texts: List[str],
    e5_cosine: np.ndarray
) -> np.ndarray:
    """
    Extract 23-dimensional features for RANK Enhanced V2.
    
    Args:
        query_texts: List of query strings
        memory_texts: List of memory strings
        e5_cosine: E5 cosine similarity (N, 1)
    
    Returns:
        features: (N, 23) array with all features
    """
    n = len(query_texts)
    features = []
    
    for i in range(n):
        query_text = str(query_texts[i] or "")
        memory_text = str(memory_texts[i] or "")
        
        # Basic features
        query_words = set(query_text.lower().split())
        memory_words = set(memory_text.lower().split())
        
        jaccard = (
            len(query_words & memory_words) / len(query_words | memory_words)
            if len(query_words | memory_words) > 0
            else 0.0
        )
        
        containment = (
            len(query_words & memory_words) / len(query_words)
            if len(query_words) > 0
            else 0.0
        )
        
        query_len = max(1, len(query_text))
        memory_len = max(1, len(memory_text))
        
        # Linguistic features
        query_has_ref = 1.0 if any(w in query_text.lower() for w in ["what", "which", "how", "when", "where", "why"]) else 0.0
        query_has_neg = 1.0 if any(w in query_text.lower() for w in ["not", "no", "never", "neither"]) else 0.0
        memory_has_neg = 1.0 if any(w in memory_text.lower() for w in ["not", "no", "never", "neither"]) else 0.0
        memory_has_update = 1.0 if any(w in memory_text.lower() for w in ["update", "change", "modify", "correct"]) else 0.0
        
        # Enhanced features (placeholders for production - would need full context)
        # In production, these would be extracted from actual memory metadata
        same_source = 0.0  # Would check memory.source == query.source
        weak_proxy = 1.0   # Default weight
        sample_weight = 1.0  # Default weight
        scope_match = 0.0  # Would check memory.scope == query.scope
        authority = 0.7    # Default baseline authority
        staleness = 0.0    # Would compute from timestamps
        version_consistency = 1.0  # Default
        egress_match = 0.0  # Would check destination alignment
        conflict_depth = 0.0  # Would trace conflict chains
        citation_score = 0.0  # Would aggregate historical citations
        correction_risk = 0.0  # Would check correction patterns
        continuation_rate = 0.5  # Default baseline
        temporal_gap = 0.0  # Would compute from timestamps
        
        row_features = [
            e5_cosine[i, 0],  # e5_cosine
            0.0,  # tfidf_cosine (deprecated, kept for compatibility)
            jaccard,  # lexical_jaccard
            containment,  # lexical_containment
            np.log1p(query_len),  # query_len_log
            np.log1p(memory_len),  # memory_len_log
            query_has_ref,  # query_has_reference
            query_has_neg,  # query_has_negation
            memory_has_neg,  # memory_has_negation
            memory_has_update,  # memory_has_update
            same_source,  # same_source_public
            weak_proxy,  # weak_proxy
            sample_weight,  # sample_weight
            scope_match,  # scope_level_match
            authority,  # authority_signal
            staleness,  # staleness_penalty
            version_consistency,  # version_consistency
            egress_match,  # egress_destination_match
            conflict_depth,  # conflict_chain_depth
            citation_score,  # citation_history_score
            correction_risk,  # correction_risk_score
            continuation_rate,  # session_continuation_rate
            temporal_gap,  # query_memory_temporal_gap
        ]
        
        features.append(row_features)
    
    return np.array(features, dtype=np.float32)
