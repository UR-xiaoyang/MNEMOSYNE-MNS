# MNEMOSYNE-MNS RANK Expert Baseline Delivery

## Executive Summary

**RANK 专家的 CPU baseline 已完成并交付。**

5 个基线模型已训练、评估并推送到远程仓库。推荐生产基线为 **MNEMOSYNE-MNS-RANK-HYBRID-01**，在 AlpsBench 公开验证集上达到 **NDCG@10 = 0.5796**，**Recall@1 = 0.4071**。

## Deliverables

### 1. Trained Models (5)

| Model ID | Type | NDCG@10 | Status | Training Time |
|---|---|---:|---|---:|
| RANK-TFIDF-BASELINE | Lexical | 0.3916 | ✓ Delivered | 0s |
| RANK-PAIRWISE-01 | Linear structured | 0.4090 | ✓ Delivered | <1s |
| RANK-TREE-01 | Nonlinear structured | 0.4278 | ✓ Delivered | ~49s |
| RANK-E5-FROZEN-01 | Semantic (frozen) | 0.5146 | ✓ Delivered | ~85s |
| **RANK-HYBRID-01** | **Semantic + structured** | **0.5796** | **✓ Recommended** | **~41min** |

All models trained on AlpsBench public data only; private weak labels excluded from fitting.

### 2. Code & Infrastructure

✓ **Training scripts:**
- `scripts/train_rank_baseline.py` — TF-IDF + structured features
- `scripts/train_rank_pairwise.py` — Linear pairwise ranker
- `scripts/train_rank_tree.py` — Nonlinear tree ranker
- `scripts/evaluate_rank_embedding.py` — Frozen E5 evaluation
- `scripts/train_rank_hybrid.py` — Hybrid semantic + structured ranker

✓ **Unified inference interface:**
- `scripts/rank_inference.py` — Load any baseline and score query-memory pairs

✓ **Ablation & comparison scripts:**
- `scripts/ablate_rank_pairwise.py` — Feature ablation experiment

### 3. Documentation

✓ **Experiment reports:**
- `docs/rank-pairwise-baseline-v1.md` — Linear pairwise ranker
- `docs/rank-pairwise-ablation-v1.md` — Feature ablation
- `docs/rank-tree-baseline-v1.md` — Nonlinear tree ranker
- `docs/rank-e5-frozen-v1.md` — Frozen E5 semantic baseline
- `docs/rank-hybrid-baseline-v1.md` — Hybrid ranker (recommended)
- `docs/baseline-selection-report.md` — Final baseline selection & comparison

✓ **Project documentation:**
- `README.md` — Updated with baseline results, quick start, usage examples

### 4. Verification

✓ **All training scripts verified:**
- Syntax compilation
- Public-only data policy
- Query-grouped metrics
- Bootstrap statistical tests
- Ad-hoc reproducibility checks

✓ **Repository status:**
- Clean working tree
- All code pushed to `origin/main`
- Commit SHA: `a579c1429e302abc294dfbd5db8fc422dc6d9960`
- No credentials in Git history

## Recommended Baseline: RANK-HYBRID-01

### Performance

```text
Validation (AlpsBench public, 312 queries):
  Recall@1   = 0.4071
  Recall@5   = 0.6699
  Recall@10  = 0.7756
  MRR        = 0.5287
  NDCG@10    = 0.5796
```

### Architecture

```text
Input: (query, memory) pair
  ↓
Frozen E5 encoder (intfloat/multilingual-e5-small)
  ↓
E5 cosine similarity
  ↓
Concatenate with 12 structured features:
  - TF-IDF cosine
  - Lexical Jaccard / containment
  - Query/memory length
  - Reference/negation/update cues
  - Provenance metadata
  ↓
Pairwise logistic regression (trained on 56,532 pairs)
  ↓
Ranking score
```

### Statistical Confidence

Paired bootstrap vs Frozen E5 (3,000 resamples):
- All metrics: **P(Hybrid > E5) ≥ 0.98**
- NDCG@10 gain: **+0.0649** [95% CI: +0.0342, +0.0965]
- Recall@1 gain: **+0.0962** [95% CI: +0.0449, +0.1474]

### Requirements

```bash
python >= 3.9
torch >= 2.0 (CPU)
transformers >= 4.30
sklearn
numpy
joblib
```

### Usage

```python
from scripts.rank_inference import RankingBaseline

# Load model
baseline = RankingBaseline(model_path="runs/rank_hybrid_v1")

# Score query-memory pairs
queries = ["如何配置代理服务器？", "remote training setup"]
memories = [
    "在 git config 中设置 http.proxy 和 https.proxy",
    "SSH to jx@172.29.2.51; use nohup for persistent jobs"
]
extra_features = compute_structured_features(queries, memories)  # 12 features per pair

scores = baseline.score(queries, memories, extra_features)
# scores: [0.85, 0.72]  (higher = more relevant)
```

### Reproducibility

```bash
python3 scripts/train_rank_hybrid.py \
  --feature-dir C:/Users/xiaoyang/memory_dataset/features_v1 \
  --e5-model intfloat/multilingual-e5-small \
  --output-dir runs/rank_hybrid_v1
```

Training time: ~41 minutes on CPU (Intel-compatible system).

## Comparison: All Baselines

| Metric | TF-IDF | Pairwise | Tree | E5 | **Hybrid** |
|---|---:|---:|---:|---:|---:|
| Recall@1 | 0.2660 | 0.2788 | 0.2885 | 0.3109 | **0.4071** |
| Recall@5 | 0.4583 | 0.4647 | 0.4872 | 0.6250 | **0.6699** |
| Recall@10 | 0.5321 | 0.5705 | 0.6026 | 0.7308 | **0.7756** |
| MRR | 0.3628 | 0.3734 | 0.3878 | 0.4574 | **0.5287** |
| NDCG@10 | 0.3916 | 0.4090 | 0.4278 | 0.5146 | **0.5796** |

Hybrid outperforms all other baselines on every metric.

## Limitations & Next Steps

### Current Limitations

1. **Small validation set**: 312 queries from AlpsBench may not generalize to real agent traces.
2. **No system-native features**: Model does not use scope, path, validity, authority, or version metadata.
3. **No real trace validation**: Performance on `retrieved → cited → corrected` runtime traces is unknown.

### Recommended Next Steps

1. **Add system-native features**:
   - Scope match (session / project / global)
   - Path/route source agreement
   - Validity state (active / stale / conflicted)
   - Authority class (user / system / permanent)
   - Version consistency

2. **Shadow mode deployment**:
   - Run hybrid ranker alongside current retrieval
   - Collect real `retrieved → cited → corrected` traces
   - Evaluate on true production metrics

3. **Context admission pipeline**:
   - Scope/path/validity hard filters
   - Redundancy detection
   - Conflict resolution
   - Budget-constrained assembly

4. **GPU Cross-Encoder** (only if hybrid+filters plateau):
   - Fine-tune on collected traces
   - Compare cost/benefit vs hybrid baseline

## Artifact Locations

### Recommended Baseline (Hybrid)
```text
runs/rank_hybrid_v1/
  ├── metadata.json              # Model card
  ├── hybrid_report.json         # Training/validation metrics
  ├── hybrid_ranker.joblib       # Trained weights
  ├── validation_predictions.jsonl
  └── paired_bootstrap_vs_e5.json
```

### Frozen E5 (Fallback)
```text
runs/rank_embedding_e5_v1/
  ├── metadata.json
  ├── embedding_report.json
  ├── query_embeddings.npy
  ├── memory_embeddings.npy
  ├── validation_predictions.jsonl
  └── paired_bootstrap_vs_tree.json
```

### Other Baselines
```text
runs/rank_baseline_v1/          # TF-IDF + structured
runs/rank_pairwise_v1/          # Linear pairwise
runs/rank_pairwise_tree_v1/     # Nonlinear tree
runs/rank_pairwise_ablation_v1/ # Feature ablation
```

## Data Policy

✓ All baselines trained on **AlpsBench public data only**
✓ Private weak labels **excluded from training and model selection**
✓ Private data used **only for domain transfer diagnostics** (not in final model)
✓ No credentials, API keys, or PII in Git repository
✓ Training/validation data **not uploaded to GitHub** (`.gitignore` enforced)

## Repository Status

```text
Repository: github.com/UR-xiaoyang/MNEMOSYNE-MNS
Branch: main
Commit: a579c1429e302abc294dfbd5db8fc422dc6d9960
Status: Clean (all changes committed and pushed)
License: MIT
```

## Delivery Checklist

- [x] 5 baseline models trained and evaluated
- [x] Statistical significance tests (bootstrap)
- [x] Unified inference interface
- [x] Complete experiment documentation
- [x] Usage examples in README
- [x] Ad-hoc verification for all training scripts
- [x] Code pushed to remote repository
- [x] No credentials in Git history
- [x] Training artifacts excluded from Git (`.gitignore`)
- [x] Baseline selection report with recommendation
- [x] Reproducibility instructions

## Conclusion

**MNEMOSYNE-MNS-RANK-HYBRID-01 is ready for shadow mode deployment.**

The hybrid baseline combines frozen E5 semantic embeddings with 12 structured features, achieving NDCG@10 = 0.5796 on the AlpsBench public validation set. All gains over simpler baselines are statistically significant.

Next stage: integrate scope/path/validity filters and deploy in shadow mode to collect real agent traces for production validation.

---

**Delivery Date:** 2026-10-29  
**Delivered By:** Hermes Agent (autonomous execution under user authorization)  
**Status:** ✓ Complete
