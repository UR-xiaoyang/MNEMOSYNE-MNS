# Pairwise Ranker Feature Ablation v1

## Setup

- Model family: `MNEMOSYNE-MNS-RANK-PAIRWISE-01`
- Training: AlpsBench public train split only
- Validation: AlpsBench public validation split only
- Private weak data: excluded
- Training method: query-grouped positive-minus-negative pairwise logistic ranking
- Validation: 312 queries, each with roughly 89–101 candidates

This is a diagnostic feature ablation, not final model selection.

## Results

| Variant | Recall@1 | Recall@5 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| all 12 features | **0.2788** | 0.4647 | 0.3733 | **0.4090** |
| similarity + cues | 0.2724 | 0.4744 | **0.3743** | 0.4025 |
| similarity + length | 0.2756 | 0.4551 | 0.3707 | 0.4014 |
| similarity only | 0.2692 | **0.4776** | 0.3700 | 0.3979 |
| similarity + provenance | 0.2692 | **0.4776** | 0.3700 | 0.3979 |
| no TF-IDF cosine | 0.0833 | 0.1795 | 0.1456 | 0.1590 |

## Findings

1. **TF-IDF cosine is indispensable in this feature set.** Removing it collapses NDCG@10 from 0.4090 to 0.1590. The current model is still fundamentally lexical.
2. **The all-feature variant is best on Recall@1 and NDCG@10.** It is retained as the baseline checkpoint.
3. **Similarity-only has slightly stronger Recall@5 but weaker top-rank quality.** This suggests auxiliary length and cue features help refine the ordering, even though they do not improve broad candidate coverage.
4. **Provenance fields make no difference on this public-only experiment.** `same_source_public`, `weak_proxy`, and `sample_weight` are constants after filtering to public data. They should not be treated as useful ranking inputs in public-only training.
5. **Results remain modest.** This experiment validates a lexical ranking signal, not readiness for production or a reason to start GPU fine-tuning.

## Decision

Keep:

```text
MNEMOSYNE-MNS-RANK-PAIRWISE-01
all 12 features
```

Next work should focus on better candidates and semantic features, not merely a larger linear feature set:

```text
- inspect top-rank errors;
- add proper path/scope features once the memory entity store exists;
- evaluate a frozen sentence-embedding reranker;
- only then consider GPU Cross-Encoder training.
```
