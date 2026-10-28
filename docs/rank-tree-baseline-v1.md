# MNEMOSYNE-MNS-RANK-TREE-01 Baseline Report

## Scope

This experiment evaluates whether nonlinear interactions among the existing 12 structured reranking features improve query-grouped ranking over the linear pairwise baseline.

- Training: AlpsBench public train split only
- Validation: AlpsBench public validation split only
- Private weak labels: excluded from fitting and checkpoint selection
- Pair construction: within-query positive-vs-negative differences, with reversed pairs as negatives
- Model: `HistGradientBoostingClassifier`
- Candidate scoring: mean pairwise win probability against all candidates in the same query

## Result

| Model | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| TF-IDF cosine | 0.2660 | 0.4583 | 0.5321 | 0.3628 | 0.3916 |
| Pairwise logistic | 0.2788 | 0.4647 | 0.5705 | 0.3734 | 0.4090 |
| **Pairwise tree** | **0.2885** | **0.4872** | **0.6026** | **0.3878** | **0.4278** |

Validation set: 312 queries, each with 89–101 candidates.

## Paired bootstrap: tree minus pairwise logistic

3,000 query-level paired bootstrap resamples:

| Metric | Delta | 95% interval | P(delta > 0) |
|---|---:|---:|---:|
| Recall@1 | +0.0096 | [-0.0128, +0.0321] | 0.768 |
| Recall@5 | +0.0224 | [-0.0064, +0.0513] | 0.922 |
| MRR | +0.0144 | [-0.0024, +0.0309] | 0.958 |
| NDCG@10 | +0.0187 | **[+0.0033, +0.0348]** | **0.993** |

## Interpretation

`MNEMOSYNE-MNS-RANK-TREE-01` is the strongest current CPU baseline. It improves every reported point estimate relative to both TF-IDF cosine and the linear pairwise model.

The evidence is strongest for **NDCG@10**. The intervals for Recall@1, Recall@5, and MRR still cross zero, so those gains are promising but not statistically resolved with this validation size.

The model should be retained as the current development checkpoint, but not treated as production-ready. It has not been evaluated on true `retrieved → cited → corrected` runtime traces, and the public benchmark remains the primary selection signal.

## Technical correction made during this experiment

The initial nonlinear pairwise implementation produced implausibly poor training and validation metrics. The cause was a faulty reversed-pair construction: negating the full feature vector also negated absolute-difference and interaction features, producing examples inconsistent with inference-time pair features.

The corrected construction is:

```text
forward = [x_positive - x_negative,
           |x_positive - x_negative|,
           x_positive * x_negative]

reverse = [x_negative - x_positive,
           |x_negative - x_positive|,
           x_negative * x_positive]
```

After correction, train/validation values are coherent and validation improves over the linear baseline.

## Decision

```text
Current development checkpoint:
MNEMOSYNE-MNS-RANK-TREE-01
```

Next step is not GPU fine-tuning. First perform qualitative error analysis of the top-ranked failures and add system-native features unavailable in AlpsBench alone:

- scope match;
- record/version validity;
- authority class;
- stale/conflicted state;
- route source agreement;
- current-session write-through state.

Only after those features are available and the CPU gain plateaus should a frozen embedding model or GPU Cross-Encoder be evaluated.
