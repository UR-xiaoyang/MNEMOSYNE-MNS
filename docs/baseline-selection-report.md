# MNEMOSYNE-MNS Baseline Selection Report

## Executive Summary

This report compares all trained ranking baselines on the AlpsBench public validation set and selects the recommended baseline for the `RANK` expert in the MNEMOSYNE-MNS system.

**Recommended baseline:** `MNEMOSYNE-MNS-RANK-HYBRID-01`

## Evaluation Protocol

- Dataset: AlpsBench public validation split
- Queries: 312
- Candidate pairs: 30,069
- Unique candidate memories: 5,719
- Metrics: query-grouped Recall@k, MRR, NDCG@k
- Training policy: public AlpsBench train split only; private weak labels excluded from fitting

## Results

| Model | Type | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | Training time |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF cosine | Lexical baseline | 0.2660 | 0.4583 | 0.5321 | 0.3628 | 0.3916 | 0s |
| Pairwise logistic | Linear structured | 0.2788 | 0.4647 | 0.5705 | 0.3734 | 0.4090 | <1s |
| Pairwise tree | Nonlinear structured | 0.2885 | 0.4872 | 0.6026 | 0.3878 | 0.4278 | ~49s |
| Frozen E5 | Semantic | 0.3109 | 0.6250 | 0.7308 | 0.4574 | 0.5146 | ~85s |
| **Hybrid E5+features** | **Semantic + structured** | **0.4071** | **0.6699** | **0.7756** | **0.5287** | **0.5796** | **~41min** |

## Analysis

### Why Hybrid is recommended

1. **Strongest performance**: Hybrid achieves the highest metrics across all ranks, including **Recall@1 = 0.4071** and **NDCG@10 = 0.5796**.
2. **Combines semantic and lexical signals**: E5 embeddings provide multilingual semantic understanding; structured features (TF-IDF, containment, length, cues) add lexical precision and metadata context.
3. **Statistically robust improvement over E5**: Bootstrap shows all gains are significant (P ≥ 0.98).
4. **CPU training and inference**: No GPU required; 41-minute training is acceptable for baseline development.

### Statistical confidence: Hybrid vs Frozen E5

3,000-resample query-level paired bootstrap:

| Metric | Hybrid advantage | 95% interval | P(Hybrid > E5) |
|---|---:|---:|---:|
| Recall@1 | +0.0962 | **[+0.0449, +0.1474]** | **0.999** |
| Recall@5 | +0.0449 | **[+0.0032, +0.0865]** | **0.980** |
| Recall@10 | +0.0449 | **[+0.0064, +0.0833]** | **0.986** |
| MRR | +0.0713 | **[+0.0383, +0.1065]** | **1.000** |
| NDCG@10 | +0.0649 | **[+0.0342, +0.0965]** | **1.000** |

The hybrid gain is statistically resolved across all metrics.

### Limitations acknowledged

- **Validation set is small and task-specific**: 312 queries from AlpsBench may not represent real `retrieved → cited → corrected` runtime traces.
- **No scope/path/validity features**: The hybrid model uses only text and basic structured features. System-native metadata (scope, authority, version, staleness) is not yet included.
- **Training time**: 41 minutes is acceptable for development but longer than frozen E5. For rapid iteration, E5 remains a viable fallback.

## Decision

The recommended production path is:

```text
1. Hybrid E5 + structured features (current baseline)
2. + scope/path/validity/authority hard filters
3. + context admission with redundancy/conflict/budget constraints
4. Shadow mode deployment with real trace collection
```

Only after this pipeline is tested with real `retrieved → cited → corrected` traces should GPU Cross-Encoder fine-tuning be considered.

## Baseline artifact locations

**Hybrid (recommended):**
```text
runs/rank_hybrid_v1/hybrid_report.json
runs/rank_hybrid_v1/hybrid_ranker.joblib
runs/rank_hybrid_v1/validation_predictions.jsonl
```

**Frozen E5 (fallback):**
```text
runs/rank_embedding_e5_v1/embedding_report.json
runs/rank_embedding_e5_v1/query_embeddings.npy
runs/rank_embedding_e5_v1/memory_embeddings.npy
runs/rank_embedding_e5_v1/validation_predictions.jsonl
```

## Reproducibility

All training scripts, evaluation protocols, and bootstrap comparisons are version-controlled in this repository.

**To reproduce the hybrid baseline:**

```bash
python3 scripts/train_rank_hybrid.py \
  --feature-dir C:/Users/xiaoyang/memory_dataset/features_v1 \
  --e5-model intfloat/multilingual-e5-small \
  --output-dir runs/rank_hybrid_v1
```

CPU-only; no GPU required. Training takes approximately 41 minutes on the reference system.

**To reproduce the frozen E5 baseline:**

```bash
python3 scripts/evaluate_rank_embedding.py \
  --feature-dir C:/Users/xiaoyang/memory_dataset/features_v1 \
  --model intfloat/multilingual-e5-small \
  --output-dir runs/rank_embedding_e5_v1
```

No training required; inference only (~85 seconds).
