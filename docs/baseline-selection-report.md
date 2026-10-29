# MNEMOSYNE-MNS Baseline Selection Report

## Executive Summary

This report compares all trained ranking baselines on the AlpsBench public validation set and selects the recommended baseline for the `RANK` expert in the MNEMOSYNE-MNS system.

**Recommended baseline:** `MNEMOSYNE-MNS-RANK-E5-FROZEN-01`

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
| **Frozen E5** | **Semantic** | **0.3109** | **0.6250** | **0.7308** | **0.4574** | **0.5146** | **~85s** |
| Hybrid E5+features | Semantic + structured | TBD | TBD | TBD | TBD | TBD | TBD |

## Analysis

### Why E5 is recommended

1. **Strongest retrieval performance**: E5 achieves the highest metrics across all ranks.
2. **No fine-tuning required**: Uses frozen `intfloat/multilingual-e5-small` weights, avoiding overfitting risk on the small public training set.
3. **Multilingual support**: Handles Chinese and English queries and memories uniformly.
4. **Inference cost acceptable**: CPU inference at ~85s for 312 queries is viable for development baseline.

### Statistical confidence

3,000-resample query-level paired bootstrap comparing E5 to pairwise tree:

| Metric | E5 advantage | 95% interval | P(E5 > tree) |
|---|---:|---:|---:|
| Recall@5 | +0.1378 | [+0.0737, +0.2019] | 1.000 |
| Recall@10 | +0.1282 | [+0.0673, +0.1923] | 1.000 |
| MRR | +0.0696 | [+0.0213, +0.1175] | 0.996 |
| NDCG@10 | +0.0869 | [+0.0384, +0.1363] | 1.000 |

The E5 gain is statistically robust for Recall@5+, MRR, and NDCG@10.

### Limitations acknowledged

- **Top-1 recall not resolved**: Recall@1 confidence interval crosses zero. For single-memory context budgets, further filtering is needed.
- **Validation set is small and task-specific**: 312 queries from AlpsBench may not represent real `retrieved → cited → corrected` runtime traces.
- **No scope/path/validity features**: E5 only sees text. System-native metadata (scope, authority, version, staleness) is not yet included.

## Decision

The recommended production path is:

```text
1. E5 semantic recall (current baseline)
2. + scope/path/validity hard filters
3. + hybrid reranker (E5 + structured features)
4. + context admission with redundancy/conflict/budget constraints
```

Only after this pipeline is tested with real runtime traces should GPU Cross-Encoder fine-tuning be considered.

## Baseline artifact locations

```text
runs/rank_embedding_e5_v1/embedding_report.json
runs/rank_embedding_e5_v1/query_embeddings.npy
runs/rank_embedding_e5_v1/memory_embeddings.npy
runs/rank_embedding_e5_v1/validation_predictions.jsonl
```

## Reproducibility

All training scripts, evaluation protocols, and bootstrap comparisons are version-controlled in this repository. The frozen E5 baseline requires no training and can be reproduced by running:

```bash
python3 scripts/evaluate_rank_embedding.py \
  --feature-dir C:/Users/xiaoyang/memory_dataset/features_v1 \
  --output-dir runs/rank_embedding_e5_v1
```

CPU-only; no GPU required.
