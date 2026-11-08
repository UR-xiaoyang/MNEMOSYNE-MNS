#!/usr/bin/env python3
"""
Feature extraction for RANK V3 Final.

V3 uses 19 features:
- 8 basic lexical features
- 3 E5 semantic features  
- 2 enhanced features
- 6 Bayesian uncertainty features

For production inference without real Bayesian statistics,
Bayesian features default to prior values (zeros).
"""
import numpy as np
import re


def extract_v3_features(
    queries: list[str],
    memories: list[str], 
    e5_cosine: np.ndarray
) -> np.ndarray:
    """
    Extract 19-dimensional features for RANK V3.
    
    Args:
        queries: List of query strings
        memories: List of memory strings
        e5_cosine: [N, 1] E5 cosine similarity
        
    Returns:
        [N, 19] feature matrix
    """
    n = len(queries)
    features = np.zeros((n, 19), dtype=np.float32)
    
    for i in range(n):
        query = queries[i]
        memory = memories[i]
        
        # Basic features
        query_tokens = set(query.split())
        memory_tokens = set(memory.split())
        query_len = len(query)
        memory_len = len(memory)
        
        # 0: query_len
        features[i, 0] = query_len
        
        # 1: memory_len
        features[i, 1] = memory_len
        
        # 2: len_ratio
        features[i, 2] = query_len / max(memory_len, 1)
        
        # 3: lexical_overlap
        overlap = len(query_tokens & memory_tokens)
        features[i, 3] = overlap
        
        # 4: lexical_jaccard
        union = len(query_tokens | memory_tokens)
        features[i, 4] = overlap / max(union, 1)
        
        # 5: lexical_containment
        features[i, 5] = overlap / max(len(query_tokens), 1)
        
        # 6: query_negation
        negation_pattern = r'\b(不|没|无|非|否|别|未)\b'
        features[i, 6] = float(bool(re.search(negation_pattern, query)))
        
        # 7: memory_negation
        features[i, 7] = float(bool(re.search(negation_pattern, memory)))
        
        # 8: e5_cosine
        features[i, 8] = e5_cosine[i, 0]
        
        # 9: e5_euclidean (approximate from cosine)
        features[i, 9] = np.sqrt(2 - 2 * e5_cosine[i, 0])
        
        # 10: e5_manhattan (approximate)
        features[i, 10] = 2 * (1 - e5_cosine[i, 0])
        
        # 11: query_specificity (inverse document frequency proxy)
        avg_token_len = np.mean([len(t) for t in query_tokens]) if query_tokens else 0
        features[i, 11] = avg_token_len
        
        # 12: memory_informativeness
        avg_token_len_mem = np.mean([len(t) for t in memory_tokens]) if memory_tokens else 0
        features[i, 12] = avg_token_len_mem
        
        # Bayesian features (13-18): default to zeros without real statistics
        # In production with shadow mode, these would be populated from memory.db
        # 13: bayes_mean
        features[i, 13] = 0.0
        
        # 14: bayes_std  
        features[i, 14] = 0.0
        
        # 15: bayes_ci_width
        features[i, 15] = 0.0
        
        # 16: bayes_coverage (has data flag)
        features[i, 16] = 0.0
        
        # 17: bayes_consistency
        features[i, 17] = 0.0
        
        # 18: bayes_confidence
        features[i, 18] = 0.0
    
    return features


def extract_v3_features_with_bayesian(
    queries: list[str],
    memories: list[str],
    memory_ids: list[str],
    e5_cosine: np.ndarray,
    bayesian_stats: dict
) -> np.ndarray:
    """
    Extract V3 features with real Bayesian statistics.
    
    Args:
        queries: List of query strings
        memories: List of memory strings
        memory_ids: List of memory IDs for stats lookup
        e5_cosine: [N, 1] E5 cosine similarity
        bayesian_stats: Dict mapping memory_id to {n_retrievals, n_citations}
        
    Returns:
        [N, 19] feature matrix with populated Bayesian features
    """
    # Start with base features
    features = extract_v3_features(queries, memories, e5_cosine)
    
    # Populate Bayesian features from stats
    for i, mem_id in enumerate(memory_ids):
        if mem_id in bayesian_stats:
            stats = bayesian_stats[mem_id]
            n_ret = stats['n_retrievals']
            n_cit = stats['n_citations']
            
            if n_ret >= 5:  # Minimum threshold for reliable statistics
                # Beta posterior parameters
                alpha = 1.0 + n_cit
                beta = 1.0 + (n_ret - n_cit)
                
                # 13: bayes_mean (posterior mean)
                features[i, 13] = alpha / (alpha + beta)
                
                # 14: bayes_std (posterior std)
                mean = alpha / (alpha + beta)
                features[i, 14] = np.sqrt(alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1)))
                
                # 15: bayes_ci_width (95% credible interval width)
                # Approximate as 4 * std
                features[i, 15] = 4 * features[i, 14]
                
                # 16: bayes_coverage (has sufficient data)
                features[i, 16] = 1.0
                
                # 17: bayes_consistency (ratio of citations to retrievals)
                features[i, 17] = n_cit / max(n_ret, 1)
                
                # 18: bayes_confidence (inverse of uncertainty)
                features[i, 18] = 1.0 / (features[i, 14] + 1e-6)
    
    return features
