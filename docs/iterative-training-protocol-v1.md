# Iterative Training Protocol v1: No-Gold-Label Weak Supervision

## Overview

P3 establishes a weak-supervised iterative optimization loop that does **not** require manual gold labels as a prerequisite. The system uses behavioral evidence (citation, correction, session continuation) and shadow-mode comparison to iteratively improve ranking models while preventing self-reinforcement.

## Core Principles

1. **Weak labels are not ground truth** — they are behavioral proxies with known limitations
2. **Unknown ≠ negative** — absence of citation/correction evidence does not imply irrelevance
3. **Citation and correction are separate signals** — positive usage vs. risk/error detection
4. **Hard constraints remain rule-based** — scope, authority, validity, conflicts, and budget are never delegated to learned models
5. **Shadow mode only** — candidate models observe but don't control real context decisions
6. **Temporal holdout** — train/val/test split by time to prevent leakage

## Evidence Hierarchy

### Strong Evidence (High Confidence)

- **Citation events**: Explicit reference to retrieved memory in the response
- **Correction events**: User explicitly corrected or flagged the memory as wrong/outdated
- **Session acceptance**: User continued working with the injected memory without correction

### Weak Evidence (Lower Confidence)

- **Query-candidate lexical overlap**: TF-IDF, containment, reference cues
- **Ranking consistency within query group**: Similar candidates should rank similarly
- **Temporal recency**: More recent memories for evolving domains

### Insufficient Evidence (Excluded from supervised loss)

- **Unknown label**: No citation, correction, or behavioral signal
- **Not retrieved**: Candidate was in the pool but not selected by baseline
- **Future candidate**: Candidate timestamp > query timestamp (temporal violation)

## Pipeline Components

### 1. Iteration Manifest Builder (`scripts/iterative_rank_loop.py`)

**Purpose**: Split shadow samples temporally and assign sample weights based on label confidence.

**Key Features**:
- Requires explicit `--input-dir` authorization (no default discovery)
- Sorts samples by `query_timestamp` before splitting (temporal ordering)
- Assigns `sample_weight=0.0` to unknown labels (excluded from supervised loss)
- Separates citation_count and correction_count in manifest
- Outputs metadata only (no query/memory text, no absolute paths)

**Usage**:
```bash
python scripts/iterative_rank_loop.py \
  --input-dir <AUTHORIZED_SHADOW_DIR> \
  --output-dir ./training/iteration_001 \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --min-samples 100
```

**Data Gates**:
- **Insufficient samples**: < min_samples (default 100) returns `insufficient_evidence`
- **High unknown ratio**: > 90% unknown triggers warning but does not block
- **Temporal violations**: Future candidates rejected to `rejections.jsonl`

### 2. Offline Replay Evaluator (`scripts/replay_rank_iterations.py`)

**Purpose**: Compare baseline vs candidate models on identical query pools using paired bootstrap.

**Key Features**:
- Validates query pools and candidate sets match exactly
- Computes query-level Recall@k, MRR, NDCG@k
- Uses existing `evaluate_ranking.py` paired bootstrap (3000 resamples)
- Returns 95% confidence intervals for each metric delta
- Output contains no file paths, only metrics and query counts

**Usage**:
```bash
python scripts/replay_rank_iterations.py \
  --baseline ./predictions/baseline.jsonl \
  --candidate ./predictions/candidate_v2.jsonl \
  --output ./comparison_baseline_vs_v2.json \
  --n-bootstrap 3000
```

**Validation Checks**:
- `(query_id, pair_id)` uniqueness within each file
- `pair_id` prefix matches `query_id` when using `::` format
- Identical query sets in baseline and candidate
- Identical candidate pools per query

### 3. Active Disagreement Sampler (`scripts/active_disagreement_sampler.py`)

**Purpose**: Identify high-value samples where baseline and candidate rankings disagree.

**Key Features**:
- Finds samples where `score_diff >= 0.1` or `rank_diff > 0`
- Categorizes disagreements: `low_confidence`, `large_rank_flip`, `score_disagreement`
- Returns sample IDs and reasons only (no query/memory text)
- Sorted by disagreement magnitude (rank_diff primary, score_diff secondary)

**Usage**:
```bash
python scripts/active_disagreement_sampler.py \
  --baseline ./predictions/baseline.jsonl \
  --candidate ./predictions/candidate_v2.jsonl \
  --output ./disagreements.jsonl \
  --n-samples 100
```

**Disagreement Categories**:
- **Low confidence**: Both models < 0.55 score (decision boundary)
- **Large rank flip**: rank_diff >= 5 (major reordering)
- **Score disagreement**: score_diff > 0.3 (different relevance assessment)

## Iteration Gates

### Data Gate (Pre-Training)

**Checks**:
- Minimum sample count (train + val >= 100)
- Unknown ratio (warn if > 90%)
- Temporal ordering preserved
- Session isolation (no session spans train/val/test)
- Secret scan passes (no API keys, tokens, paths in output)

**Actions**:
- `insufficient_evidence`: Stop, do not train
- `pass_with_warning`: Proceed but log warnings
- `pass`: Proceed to offline evaluation

### Offline Gate (Post-Training)

**Checks**:
- Primary metric (NDCG@10 or Recall@1) 95% CI does not overlap zero (candidate > baseline)
- Correction-risk metric does not increase
- Hard constraint violations = 0 (scope, validity, conflicts)
- All evaluation slices reportable (no empty query groups)

**Actions**:
- **Pass**: Promote to shadow mode
- **Fail**: Stop iteration, keep baseline
- **Insufficient evidence**: Do not train, collect more shadow data

### Shadow Gate (Live Monitoring)

**Checks** (sliding window, typically 7-14 days):
- Citation proxy rate does not degrade vs baseline
- Context pollution proxy (redundancy, off-topic) does not increase
- Correction events do not increase
- Latency within budget
- Sample count >= minimum for statistical power

**Actions**:
- **Pass**: Promote candidate to production baseline
- **Fail**: Revert to previous baseline, log failure reason
- **Insufficient evidence**: Extend shadow period

## Failure Modes and Safeguards

### Self-Reinforcement Prevention

- **Temporal holdout**: Test set is strictly chronologically after train/val
- **Query-group paired bootstrap**: Resampling preserves query structure
- **Baseline disagreement monitoring**: Track when candidate diverges from stable baseline
- **Hard constraint override**: Scope/authority/validity rules always apply regardless of model score

### Label Noise Handling

- **Unknown labels excluded from supervised loss** (`sample_weight=0.0`)
- **Correction != negative**: Corrected memories are flagged risk, not removed from positive pool
- **Multi-task or separate models**: Citation (positive usage) and correction (risk) can be separate heads
- **Conservative promotion**: Require improvement across multiple metrics, not just one

### Insufficient Data Response

- **Do not lower thresholds automatically** — return `insufficient_evidence` instead
- **Do not convert unknown to negative** — keep them as `label="unknown"` with explicit reason
- **Do not fabricate metrics** — report actual sample counts and confidence intervals
- **Do not auto-promote without statistical evidence** — require minimum queries and CI separation

## Example Iteration Workflow

### Iteration 0: Baseline Establishment

1. Deploy frozen baseline ranker (existing Hybrid model)
2. Enable shadow trace collection
3. Collect >= 100 queries with retrieval candidates
4. Export shadow samples with `build_shadow_training_data.py`

### Iteration 1: First Candidate

1. Build iteration manifest: `iterative_rank_loop.py`
2. Check data gate (sample count, unknown ratio, temporal ordering)
3. Train candidate model on train split (citation positive, correction risk, unknown excluded)
4. Evaluate on val split: `replay_rank_iterations.py` (baseline vs candidate)
5. Check offline gate (NDCG@10 CI, correction risk, hard constraints)
6. If pass: deploy to shadow mode
7. Monitor shadow metrics for 7-14 days
8. Check shadow gate (citation rate, pollution, corrections, latency)
9. If pass: promote to production baseline

### Iteration 2+: Continuous Improvement

- Repeat with new shadow samples from current baseline
- Use `active_disagreement_sampler.py` to identify high-value edge cases
- Track historical iteration results in `experiments/iterations/manifest_history.jsonl`

## Reporting Requirements

Every iteration must produce:

1. **Iteration manifest** (`iteration_manifest.json`):
   - Sample counts (train/val/holdout)
   - Label distribution (citation/correction/unknown counts and ratios)
   - Temporal split boundaries (earliest/latest timestamps)
   - Data gate result (pass/warning/insufficient_evidence)

2. **Offline evaluation report** (`offline_comparison.json`):
   - Baseline and candidate metrics (Recall@1, Recall@5, Recall@10, MRR, NDCG@5, NDCG@10)
   - Paired bootstrap deltas with 95% CI
   - Query count and sample count
   - Offline gate result (pass/fail/insufficient_evidence)

3. **Shadow evaluation report** (`shadow_report.json`):
   - Citation proxy rate (baseline vs candidate)
   - Context pollution proxy
   - Correction event rate
   - Latency percentiles (p50, p90, p99)
   - Shadow gate result (pass/fail/insufficient_evidence)

## Version History

- **v1 (P3)**: Initial no-gold-label protocol with weak supervision
- Baseline: Hybrid ranker (NDCG@10 = 0.5796 on AlpsBench validation)
- Next: Consider frozen semantic encoder + structured features before Cross-Encoder fine-tuning

## References

- Shadow trace schema: `docs/shadow-mode-trace-schema.md`
- P2 completion report: `P2_COMPLETION_REPORT.md`
- Public benchmarks: `references/public-memory-benchmarks.md` (via memory-dataset-engineering skill)
- Evaluation infrastructure: `scripts/evaluate_ranking.py`
