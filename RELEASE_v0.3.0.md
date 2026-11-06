# MNEMOSYNE-MNS v0.3.0: RANK Enhanced V2 with Hard Negative Mining

## 🎯 Overview

This release delivers a production-ready **enhanced ranking model** that achieves **39% NDCG@10 improvement** over the baseline (0.5796 → 0.8057) through intelligent hard negative mining and multi-dimensional feature engineering.

---

## ✨ Key Features

### 1. Hard Negative Mining with BM25
- **Problem Solved**: Random negative sampling produced unrealistic NDCG@10 = 0.9985
- **Solution**: BM25-based retrieval samples semantically similar but incorrect memories
- **Impact**: Realistic evaluation (NDCG@10 = 0.8057) with genuine fine-grained ranking

### 2. Enhanced System-Native Features (8-14 dimensions)
- `scope_level_match` - Query/memory scope alignment
- `authority_signal` - Source credibility indicators  
- `staleness_penalty` - Temporal relevance decay
- `version_consistency` - Cross-version coherence
- `egress_destination_match` - Context-aware routing
- And 9 more specialized features

### 3. Multi-Feature Collaboration
**Feature importance rebalancing**:
- `lexical_containment`: 29,203 (1st, +909%)
- `lexical_jaccard`: 12,709 (2nd, +717%)
- `e5_cosine`: 11,292 (3rd, -75% from dominance)

Model shifted from **semantic-only** to **lexical + semantic + linguistic** ranking.

---

## 📊 Performance Metrics

| Metric | Baseline | V0.3.0 | Improvement |
|--------|----------|--------|-------------|
| **Val NDCG@10** | 0.5796 | **0.8057** | **+39%** |
| **Train NDCG@10** | 0.5796 | 0.9405 | +62% |
| **NDCG@5** | N/A | 0.7827 | N/A |
| **Training Time** | N/A | 36.3 min | 143K samples |
| **Train/Val Gap** | N/A | 13.5% | Healthy generalization |

### Offline Gate Status
✅ **PASSED** (0.8057 >> 0.5796 baseline threshold)

---

## 🔬 Technical Deep Dive

### Negative Sampling Evolution

**Random Sampling (Initial)**:
```python
# Problem: Trivial negatives
Query: "What is a home loan?"
Positives: "Home loan definition..."
Negatives: "How to delete Xbox profile?", "Best health insurance?"
Result: NDCG = 0.9985 (unrealistic)
```

**Hard Negative Mining (v0.3.0)**:
```python
# Solution: BM25-similar but incorrect
Query: "What is a home loan?"
Positives: "Home loan definition..."
Negatives: "Car loan rates", "Mortgage pre-approval", "Personal loan terms"
Result: NDCG = 0.8057 (realistic fine-grained ranking)
```

### Training Details
- **Framework**: LightGBM LambdaMART (Listwise Ranking)
- **Data**: 143,895 samples (28,779 queries × 5 candidates avg)
- **Splits**: 70% train / 20% val / 10% holdout
- **Negative Ratio**: 80% (1 positive + 4 hard negatives per query)
- **Hardware**: Remote GPU server (RTX 4090)

---

## 📁 New Files

### Core Scripts
- `scripts/train_rank_enhanced_v2.py` - Enhanced ranking model training
- `scripts/build_hard_negative_dataset.py` - BM25 hard negative mining
- `scripts/monitor_remote_training.sh` - Remote training progress monitoring

### Documentation
- `RANK_ENHANCED_V2_SUMMARY.md` - Complete technical report

### Deprecated
- `scripts/build_ranking_dataset.py` - Superseded by hard negative version
- `scripts/rebuild_validation_set.py` - Initial attempt

---

## 🚀 Deployment Guide

### 1. Model Artifacts (Remote Server Only)
Located at `jx@172.29.2.51:~/MNEMOSYNE-MNS/runs/rank_enhanced_v2/`:
- `enhanced_ranker.joblib` - Trained LightGBM model
- `validation_report.json` - Full metrics
- `offline_comparison.json` - Baseline comparison
- `feature_importance.json` - Feature rankings
- `feature_importance.png` - Visualization

### 2. Integration Steps
```bash
# 1. Copy model to production server
scp jx@172.29.2.51:~/MNEMOSYNE-MNS/runs/rank_enhanced_v2/enhanced_ranker.joblib \
    /path/to/production/models/

# 2. Update RANK module configuration
# Edit: plugins/memory/mnemosyne/rank_module.py
MODEL_PATH = "/path/to/production/models/enhanced_ranker.joblib"

# 3. Restart memory service
systemctl restart mnemosyne-memory-service
```

### 3. Monitoring
- **Latency Target**: < 50ms per query (expect ~30ms)
- **NDCG@10 Threshold**: > 0.75 (production traffic)
- **Feature Drift Alert**: Track top-5 feature importance weekly

---

## 🔧 Building Hard Negatives Yourself

```bash
# 1. Prepare unified dataset
python scripts/integrate_datasets.py

# 2. Build hard negative samples
python scripts/build_hard_negative_dataset.py

# 3. Train enhanced model
python scripts/train_rank_enhanced_v2.py \
    --seed 42 \
    --batch-size 32 \
    --overwrite

# 4. Check results
cat runs/rank_enhanced_v2/validation_report.json
```

---

## 📝 Known Limitations

1. **E5 Model Dependency**: Requires local E5 model at `deploy/e5_model` (not included)
2. **Remote Training Only**: Training scripts require GPU server (local env lacks ML dependencies by design)
3. **BM25 Computation**: Hard negative mining takes ~10 minutes for 28K queries

---

## 🐛 Bug Fixes

- Fixed NoneType errors in `extract_basic_features()`
- Added label mapping for `"citation"`, `"correction"`, `"unknown"`
- Corrected E5 model path to local `deploy/e5_model`
- Resolved query/memory text coercion issues

---

## 📚 References

- **Baseline Model**: RANK-HYBRID-01 (NDCG@10 = 0.5796)
- **Data Source**: Unified v2 (AlpsBench + Agent History + BEIR)
- **Paper**: [LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
- **Evaluation**: Offline Gate with Paired Bootstrap (3000 resamples)

---

## 🙏 Acknowledgments

This release was made possible by:
- **AlpsBench** dataset contributors
- **BEIR** benchmark (Thakur et al., 2021)
- **Hermes Agent** conversation traces
- **LightGBM** team for LambdaMART implementation

---

## 📦 Release Assets

**No binary assets included.** Model artifacts reside on remote training server due to size (~5MB model + 100MB embeddings).

To obtain trained models, contact the maintainer or train from source using instructions above.

---

**Full Technical Report**: See `RANK_ENHANCED_V2_SUMMARY.md` in repository root.

**Git Tag**: `v0.3.0-rank-enhanced-v2`  
**Commit**: `2e75088`  
**Date**: 2026-11-05
