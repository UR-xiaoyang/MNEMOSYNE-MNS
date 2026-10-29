# MNEMOSYNE-MNS-RANK-E5-FROZEN-01 Baseline Report

## Scope

This is the first semantic reranking baseline. It uses `intfloat/multilingual-e5-small` as a frozen encoder; no model weights were fine-tuned.

- Encoder inference: CPU
- Query prefix: `query: `
- Memory prefix: `passage: `
- Max length: 256
- Training/fitting: none
- Evaluation: AlpsBench public validation only
- Private weak labels: excluded

## Results

| Model | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| TF-IDF cosine | 0.2660 | 0.4583 | 0.5321 | 0.3628 | 0.3916 |
| Pairwise tree | 0.2885 | 0.4872 | 0.6026 | 0.3878 | 0.4278 |
| **Frozen multilingual E5** | **0.3109** | **0.6250** | **0.7308** | **0.4574** | **0.5146** |

Validation set: 312 queries, 30,069 candidate pairs, 5,719 unique candidate memories.

## Paired bootstrap: frozen E5 minus pairwise tree

3,000 query-level paired bootstrap resamples:

| Metric | Delta | 95% interval | P(delta > 0) |
|---|---:|---:|---:|
| Recall@1 | +0.0224 | [-0.0353, +0.0833] | 0.751 |
| Recall@5 | +0.1378 | **[+0.0737, +0.2019]** | **1.000** |
| Recall@10 | +0.1282 | **[+0.0673, +0.1923]** | **1.000** |
| MRR | +0.0696 | **[+0.0213, +0.1175]** | **0.996** |
| NDCG@10 | +0.0869 | **[+0.0384, +0.1363]** | **1.000** |

## Interpretation

Frozen multilingual semantic representations provide substantially more useful retrieval information than the current lexical/structured CPU baselines, especially beyond the first candidate. The gain is not caused by fine-tuning: the encoder was used only for inference.

The top-1 gain is not statistically resolved on this 312-query validation set. This matters for a production context budget that injects only one memory: the system still needs better candidate filtering, scope/path features, and possibly a trained reranker on top of embeddings.

The result does justify moving to the next stage:

```text
frozen E5 recall layer
→ scope/path/validity hard filters
→ pairwise or Cross-Encoder reranker
→ budgeted context admission
```

It does not yet justify training a large model. The current public validation set is small and task-specific; true runtime `retrieved → cited → corrected` traces are still absent.

## Decision

- Promote `MNEMOSYNE-MNS-RANK-E5-FROZEN-01` as the current retrieval baseline.
- Keep `MNEMOSYNE-MNS-RANK-TREE-01` as a structured-feature comparison model.
- Next experiment: hybrid E5 + lexical/path/authority features, evaluated with the same query-grouped protocol.
- Only after hybrid reranking is evaluated should we consider GPU Cross-Encoder fine-tuning.
