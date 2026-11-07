# MNEMOSYNE-MNS-RANK-HYBRID-01 Baseline Reproducibility Report

**Generated**: 2026-11-03  
**Baseline ID**: MNEMOSYNE-MNS-RANK-HYBRID-01  
**Historical Run**: runs/rank_hybrid_v1/

## Reproducibility Status

✓ **Frozen and verified** (evaluation module)  
⚠ **Training reproducibility enhanced** — New manifest protocol applies to future runs

**As of 2026-11-03**:
- Evaluation module (`evaluate_ranking.py`) is frozen and fully reproducible
- Historical artifact `runs/rank_hybrid_v1/` predates manifest protocol (lacks `manifest.json`)
- New training runs (using updated `train_rank_hybrid.py`) now generate:
  - Complete `manifest.json` with Git SHA, dependency versions, input file SHA-256, seed
  - Deterministic seed control via `--seed` (required argument)
  - Safe output directory handling (explicit `--overwrite` required)
- Historical performance remains valid; metrics are independently verifiable via `evaluate_ranking.py`

## Historical Performance (AlpsBench Validation)

From `runs/rank_hybrid_v1/hybrid_report.json`:

| Metric | Value |
|--------|-------|
| Recall@1 | 0.4071 |
| Recall@5 | 0.6699 |
| Recall@10 | 0.7756 |
| MRR | 0.5287 |
| NDCG@5 | 0.5463 |
| **NDCG@10** | **0.5796** |
| Queries | 312 |

## Verification with evaluate_ranking.py

```bash
python scripts/evaluate_ranking.py \
  --predictions runs/rank_hybrid_v1/validation_predictions.jsonl \
  --score-key hybrid_score \
  --bootstrap-resamples 3000 \
  --seed 42
```

**Result**: ✓ Metrics match historical report exactly

## Reproducible Training Command

**Updated Command** (with new reproducibility features):

```bash
python scripts/train_rank_hybrid.py \
  --feature-dir C:/Users/xiaoyang/memory_dataset/features_v1 \
  --e5-model intfloat/multilingual-e5-small \
  --output-dir runs/rank_hybrid_rerun_$(date +%Y%m%d_%H%M%S) \
  --batch-size 64 \
  --max-length 256 \
  --seed 42
```

**Note**: `--seed` is now **required**. To reproduce the historical run exactly, use `--seed 42`.

**New Safety Features**:
- Training refuses to overwrite existing output directories by default
- Use `--overwrite` flag to explicitly allow overwriting
- Each run generates `manifest.json` with full lineage tracking

**Fixed Parameters**:
- Random seed: 42 (hardcoded in LogisticRegression)
- C: 1.0
- max_iter: 2000
- solver: lbfgs

**Expected Runtime**: ~41 minutes (CPU-only)

## Data Manifest

**Feature Directory**: `C:/Users/xiaoyang/memory_dataset/features_v1/`

Required files:
- `rerank_train_pairs.jsonl` (25,123 pairs, 296 queries after public filter)
- `rerank_validation_pairs.jsonl` (30,069 pairs, 312 queries after public filter)
- `rerank_train_extra.npy` (12 structured features per pair)
- `rerank_validation_extra.npy` (12 structured features per pair)
- `rerank_extra_features.json` (feature names)

**Data Policy**: AlpsBench public train/validation only; private weak data excluded

## Model Components

1. **Semantic Encoder**: `intfloat/multilingual-e5-small` (frozen, CPU inference)
2. **Structured Features** (12): TF-IDF cosine, lexical Jaccard, containment, length, cue features
3. **Ranker**: Pairwise logistic regression on positive-minus-negative feature deltas

## Artifacts Saved

- `hybrid_ranker.joblib` — Scaler + LogisticRegression + metadata
- `validation_predictions.jsonl` — Per-pair predictions for bootstrap analysis
- `hybrid_report.json` — Training manifest with metrics, data policy, verification flags
- `manifest.json` — **New**: Complete run manifest for reproducibility
  - Schema version, run ID, Git SHA
  - Command arguments and random seed
  - Python + dependency versions
  - Input file paths and SHA-256 hashes
  - Row counts (total and public-filtered)
  - Training statistics (queries, candidates, pairs)

## Evaluation Module

**Tool**: `scripts/evaluate_ranking.py`

**Capabilities**:
- Query-grouped Recall@k, MRR, NDCG@k
- Paired bootstrap comparison (3000+ resamples, seeded)
- **Strict validation**: Rejects mismatched pair_id sets or labels between models
- **Local RNG**: Uses `np.random.default_rng()` to avoid polluting global state
- CLI with JSONL input/output

**Test Coverage**: 15 unit tests in `tests/test_ranking_evaluation.py`, all passing

**Bootstrap Comparison Safety**:
- Validates identical candidate pools per query before computing deltas
- Verifies label consistency across compared models
- Raises informative errors on invalid comparisons
- Prevents silent statistical errors from candidate mismatch

## Data Contract Validation

**New Tool**: `scripts/validate_dataset.py`

Enforces:
- Required fields (trace_id, session_id)
- Temporal ordering (candidates < query)
- Session split isolation
- Sensitive data redaction
- Evaluation-only exclusion from training

**Test Coverage**: 14 unit tests in `tests/test_dataset_validation.py`, all passing

## Prerequisites for Rerun

**Available**:
- ✓ Feature data (`features_v1/`)
- ✓ Training script with fixed parameters
- ✓ Evaluation module with deterministic bootstrap
- ✓ Contract validation

**Blockers**:
- None identified for local rerun

## Comparison Gate

To promote a new model over HYBRID-01:

1. Train new model with same or updated features
2. Generate `validation_predictions.jsonl` with new scores
3. Run paired bootstrap comparison:
   ```bash
   python scripts/evaluate_ranking.py \
     --predictions runs/new_model/validation_predictions.jsonl \
     --score-key new_score \
     --compare-with runs/rank_hybrid_v1/validation_predictions.jsonl \
     --compare-score-key hybrid_score \
     --bootstrap-resamples 3000 \
     --seed 42 \
     --output runs/new_model/bootstrap_vs_hybrid.json
   ```
4. Check promotion gate (from `docs/model-training-protocol-v1.md`):
   - NDCG@10 95% CI lower bound > 0
   - Recall@1 no statistical degradation
   - All contract tests pass
   - ≥90% of key slices within ±0.02 NDCG@10

## Next Steps

Per implementation plan Phase 0 completion:

**Completed**:
- [x] Task 0.1: Frozen RANK baseline with reproducible evaluation
- [x] Task 0.2: Training protocol and dataset card templates

**Not Attempted** (out of scope for this task):
- Re-training from scratch (41 min, not required to freeze baseline)
- Private weak data collection (requires shadow trace export)
- Human gold annotation (requires annotation pipeline)

**Recommended Next** (Phase 1):
- Task 1.1: Inventory and normalize private conversation sources
- Task 1.2: Redaction, session splits, and audits

## Verification Commands

```bash
# Run all new tests
python -m unittest tests.test_ranking_evaluation tests.test_dataset_validation -v

# Verify script syntax
python -m py_compile scripts/evaluate_ranking.py scripts/validate_dataset.py

# Reproduce historical metrics
python scripts/evaluate_ranking.py \
  --predictions runs/rank_hybrid_v1/validation_predictions.jsonl \
  --score-key hybrid_score
```

All commands verified on 2026-11-03.
