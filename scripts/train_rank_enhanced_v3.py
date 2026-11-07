#!/usr/bin/env python3
"""
Train RANK Enhanced V3 with Bayesian uncertainty features.

Enhancement over V2:
- V2: 23-dim features (lexical, semantic, linguistic)
- V3: 29-dim features (V2 + 6 Bayesian uncertainty features)

New Bayesian features:
1. bayesian_posterior_mean - Expected memory quality
2. bayesian_posterior_std - Uncertainty measure
3. bayesian_ci_width - Credible interval width (95%)
4. bayesian_has_data - Binary flag (has retrieval history)
5. bayesian_n_retrievals_log - Log-transformed retrieval count
6. bayesian_uncertainty_penalty - λ * posterior_std
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# Import Bayesian module
sys.path.insert(0, str(Path(__file__).parent))
from bayesian_rank_integration import BayesianMemoryQuality, extract_bayesian_features

# Import V2 feature extraction
try:
    from train_rank_enhanced_v2 import (
        load_data,
        extract_basic_features,
        extract_enhanced_features,
        compute_e5_embeddings
    )
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    print("⚠️  RANK Enhanced V2 not found. Cannot extract base features.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not installed. Run: pip install lightgbm")

def build_bayesian_model_from_synthetic_data(
    synthetic_path: str = "data/synthetic_retrieval/retrieval_history.jsonl"
) -> BayesianMemoryQuality:
    """
    Build Bayesian model from synthetic retrieval history.
    
    In production, this would load real shadow trace statistics.
    """
    bayesian_model = BayesianMemoryQuality(alpha_prior=1.0, beta_prior=1.0)
    
    if not Path(synthetic_path).exists():
        print(f"⚠️  Synthetic data not found: {synthetic_path}")
        print("   Using empty Bayesian model (all priors)")
        return bayesian_model
    
    # Load synthetic statistics
    with open(synthetic_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            mem_id = record['memory_id']
            n_ret = record['n_retrievals']
            n_cit = record['n_citations']
            
            # Populate model
            bayesian_model.memory_stats[mem_id] = {
                'n_retrievals': n_ret,
                'n_citations': n_cit,
            }
    
    print(f"✓ Loaded Bayesian statistics for {len(bayesian_model.memory_stats)} memories")
    return bayesian_model

def extract_v3_features(
    query: str,
    memory: dict,
    query_emb: np.ndarray,
    memory_emb: np.ndarray,
    bayesian_model: BayesianMemoryQuality
) -> dict:
    """
    Extract 29-dim features for RANK Enhanced V3.
    
    Combines:
    - 8 basic features (from V2)
    - 15 enhanced features (from V2)
    - 6 Bayesian features (new)
    """
    if not V2_AVAILABLE:
        raise RuntimeError("V2 feature extraction not available")
    
    # V2 features (23-dim)
    basic_features = extract_basic_features(query, memory)
    enhanced_features = extract_enhanced_features(query, memory, query_emb, memory_emb)
    
    # Bayesian features (6-dim)
    bayesian_features = extract_bayesian_features(query, memory, bayesian_model)
    
    # Combine
    all_features = {**basic_features, **enhanced_features, **bayesian_features}
    
    return all_features

def train_rank_v3(
    train_path: str,
    val_path: str,
    output_dir: str,
    e5_model_path: str = "deploy/e5_model"
):
    """Train RANK Enhanced V3 with Bayesian features."""
    
    print("="*60)
    print("RANK Enhanced V3 Training with Bayesian Features")
    print("="*60)
    
    if not V2_AVAILABLE:
        print("❌ V2 feature extraction required")
        print("   Ensure train_rank_enhanced_v2.py is in the same directory")
        return
    
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM required")
        print("   pip install lightgbm")
        return
    
    # Step 1: Load data
    print("\n1. Loading training data...")
    train_samples = load_data(train_path)
    val_samples = load_data(val_path)
    print(f"   Train: {len(train_samples)} samples")
    print(f"   Val: {len(val_samples)} samples")
    
    # Step 2: Build Bayesian model
    print("\n2. Building Bayesian model from synthetic data...")
    bayesian_model = build_bayesian_model_from_synthetic_data()
    
    # Step 3: Compute E5 embeddings
    print("\n3. Computing E5 embeddings...")
    print("   (This may take 10-15 minutes for large datasets)")
    
    train_queries = [s['query_text'] for s in train_samples]
    train_memories = [s['memory_text'] for s in train_samples]
    val_queries = [s['query_text'] for s in val_samples]
    val_memories = [s['memory_text'] for s in val_samples]
    
    print("   Encoding training set...")
    train_query_embs = compute_e5_embeddings(train_queries, e5_model_path)
    train_memory_embs = compute_e5_embeddings(train_memories, e5_model_path)
    
    print("   Encoding validation set...")
    val_query_embs = compute_e5_embeddings(val_queries, e5_model_path)
    val_memory_embs = compute_e5_embeddings(val_memories, e5_model_path)
    
    # Step 4: Extract V3 features (29-dim)
    print("\n4. Extracting V3 features (29-dim: 23 + 6 Bayesian)...")
    
    X_train = []
    y_train = []
    groups_train = []
    
    for i, sample in enumerate(train_samples):
        features = extract_v3_features(
            sample['query_text'],
            {'memory_text': sample['memory_text'], 'memory_id': sample.get('memory_id')},
            train_query_embs[i],
            train_memory_embs[i],
            bayesian_model
        )
        
        feature_vector = [
            # Basic features (8)
            features['query_len_log'],
            features['memory_len_log'],
            features['len_ratio'],
            features['lexical_overlap'],
            features['lexical_jaccard'],
            features['lexical_containment'],
            features['query_has_negation'],
            features['memory_has_negation'],
            # Enhanced features (15)
            features['scope_level_match'],
            features['authority_signal'],
            features['staleness_penalty'],
            features['version_consistency'],
            features['egress_destination_match'],
            features['conflict_chain_depth'],
            features['citation_history_score'],
            features['correction_risk_score'],
            features['session_continuation_rate'],
            features['query_memory_temporal_gap'],
            features['e5_cosine'],
            features['e5_euclidean'],
            features['e5_manhattan'],
            features['query_specificity'],
            features['memory_informativeness'],
            # Bayesian features (6)
            features['bayesian_posterior_mean'],
            features['bayesian_posterior_std'],
            features['bayesian_ci_width'],
            features['bayesian_has_data'],
            features['bayesian_n_retrievals_log'],
            features['bayesian_uncertainty_penalty'],
        ]
        
        X_train.append(feature_vector)
        y_train.append(1 if sample['label'] in ['relevant', 'citation', 'correction'] else 0)
        groups_train.append(sample.get('query_id', i))
    
    X_val = []
    y_val = []
    groups_val = []
    
    for i, sample in enumerate(val_samples):
        features = extract_v3_features(
            sample['query_text'],
            {'memory_text': sample['memory_text'], 'memory_id': sample.get('memory_id')},
            val_query_embs[i],
            val_memory_embs[i],
            bayesian_model
        )
        
        feature_vector = [
            features['query_len_log'],
            features['memory_len_log'],
            features['len_ratio'],
            features['lexical_overlap'],
            features['lexical_jaccard'],
            features['lexical_containment'],
            features['query_has_negation'],
            features['memory_has_negation'],
            features['scope_level_match'],
            features['authority_signal'],
            features['staleness_penalty'],
            features['version_consistency'],
            features['egress_destination_match'],
            features['conflict_chain_depth'],
            features['citation_history_score'],
            features['correction_risk_score'],
            features['session_continuation_rate'],
            features['query_memory_temporal_gap'],
            features['e5_cosine'],
            features['e5_euclidean'],
            features['e5_manhattan'],
            features['query_specificity'],
            features['memory_informativeness'],
            features['bayesian_posterior_mean'],
            features['bayesian_posterior_std'],
            features['bayesian_ci_width'],
            features['bayesian_has_data'],
            features['bayesian_n_retrievals_log'],
            features['bayesian_uncertainty_penalty'],
        ]
        
        X_val.append(feature_vector)
        y_val.append(1 if sample['label'] in ['relevant', 'citation', 'correction'] else 0)
        groups_val.append(sample.get('query_id', i))
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    
    # Step 5: Train LightGBM
    print("\n5. Training LightGBM LambdaMART (listwise ranking)...")
    
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid']
    )
    
    # Step 6: Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(output_path / "rank_enhanced_v3.txt"))
    print(f"\n✓ Model saved to {output_path / 'rank_enhanced_v3.txt'}")
    
    # Step 7: Feature importance
    feature_names = [
        'query_len_log', 'memory_len_log', 'len_ratio',
        'lexical_overlap', 'lexical_jaccard', 'lexical_containment',
        'query_has_negation', 'memory_has_negation',
        'scope_level_match', 'authority_signal', 'staleness_penalty',
        'version_consistency', 'egress_destination_match', 'conflict_chain_depth',
        'citation_history_score', 'correction_risk_score', 'session_continuation_rate',
        'query_memory_temporal_gap', 'e5_cosine', 'e5_euclidean', 'e5_manhattan',
        'query_specificity', 'memory_informativeness',
        'bayesian_posterior_mean', 'bayesian_posterior_std', 'bayesian_ci_width',
        'bayesian_has_data', 'bayesian_n_retrievals_log', 'bayesian_uncertainty_penalty',
    ]
    
    importance = model.feature_importance()
    importance_dict = dict(zip(feature_names, importance.tolist()))
    
    with open(output_path / "feature_importance_v3.json", 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    print("\nTop 10 features:")
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, imp in sorted_features:
        print(f"  {name:<35}: {imp:>10.0f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    # Configuration
    train_path = "data/unified/train_v2.jsonl"
    val_path = "data/unified/val_v2.jsonl"
    output_dir = "runs/rank_enhanced_v3"
    
    # Check if data exists
    if not Path(train_path).exists():
        print(f"❌ Training data not found: {train_path}")
        print("\nPlease ensure unified v2 dataset exists.")
        sys.exit(1)
    
    # Train
    train_rank_v3(train_path, val_path, output_dir)
