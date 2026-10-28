# MNEMOSYNE-MNS-RANK-PAIRWISE-01 Baseline Report

## Scope

This is the first CPU ranking baseline for the `RANK` expert. It was trained only on the public AlpsBench reranking training split; private weak adaptation data was not used for fitting.

The comparison is query-grouped. Each validation query has roughly 89–101 candidate memories; metrics are computed after sorting candidates within each query.

## Models

1. **TF-IDF cosine** — nonparametric lexical baseline.
2. **MNEMOSYNE-MNS-RANK-PAIRWISE-01** — logistic pairwise ranker trained on positive-minus-negative candidate feature differences.

The pairwise model uses 12 structured features from the existing feature pipeline. Its training pairs are constructed only within the same public training query:

```text
positive_candidate - negative_candidate → label 1
negative_candidate - positive_candidate → label 0
```

## Results

| Metric | TF-IDF cosine | Pairwise ranker | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.2660 | **0.2788** | +0.0128 |
| Recall@5 | 0.4583 | **0.4647** | +0.0064 |
| Recall@10 | 0.5321 | **0.5705** | +0.0385 |
| MRR | 0.3628 | **0.3734** | +0.0106 |
| NDCG@5 | 0.3680 | **0.3753** | +0.0073 |
| NDCG@10 | 0.3916 | **0.4090** | +0.0174 |

Validation size: 312 queries, 30,069 candidate pairs, 340 positive pairs.

A 2,000-resample query bootstrap gives these approximate paired intervals for `pairwise - TF-IDF`:

| Metric | Delta | 95% bootstrap interval | P(delta > 0) |
|---|---:|---:|---:|
| Recall@1 | +0.0128 | [-0.0096, +0.0353] | 0.8425 |
| Recall@5 | +0.0064 | [-0.0224, +0.0353] | 0.6245 |
| MRR | +0.0106 | [-0.0042, +0.0264] | 0.9125 |
| NDCG@10 | +0.0174 | [+0.0001, +0.0342] | 0.9770 |

## Interpretation

The pairwise model is the current best baseline, but the improvement is modest. The only metric with a bootstrap interval clearly above zero in this quick analysis is NDCG@10, and even that interval is close to zero. This is evidence of a useful ranking signal, not evidence that the ranker is ready for production.

The model should remain in shadow mode. It has not been evaluated on an independent real-world memory-retrieval test with true historical retrieval logs.

## Private weak-transfer diagnostic

The pairwise model was evaluated on 33 isolated private weak-label queries after training. This is a domain-transfer diagnostic, not a model-selection result:

```text
Recall@1   0.2727
Recall@5   1.0000
Recall@10  1.0000
MRR        0.5530
NDCG@10    0.6660
```

These labels are weak temporal/lexical proxies and must not be interpreted as ground-truth accuracy.

## Decision

- Keep `MNEMOSYNE-MNS-RANK-PAIRWISE-01` as the current CPU baseline.
- Do not start GPU fine-tuning yet.
- Next experiment: feature ablation and hard-negative analysis, followed by a neural reranker only if the improvement survives a cleaner independent evaluation.
- Keep private weak data out of the public-supervised checkpoint selection.
