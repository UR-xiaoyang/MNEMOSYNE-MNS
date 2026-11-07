# MNEMOSYNE-MNS Model Training Protocol v1

## Purpose

This document defines the training, evaluation, and promotion protocol for all learning components in the MNEMOSYNE-MNS memory system. It establishes data contracts, risk boundaries, and decision gates that every trained model must satisfy before deployment.

## Scope

Applies to: **RANK**, **ROUTE**, **WRITE**, **UPDATE**, **ASSEMBLE** experts.

## Core Principles

1. **No production code without passing tests** — Every model requires reproducible evaluation, frozen baselines, and contract validation before promotion.
2. **Weak labels are not ground truth** — Private weak supervision is explicitly marked, lower-weighted, and never the primary model selection signal.
3. **Hard constraints stay hard** — Models propose; rules enforce. Scope, permissions, validity, conflicts, and budget remain rule-governed.
4. **Shadow before production** — Real-world validation precedes every promotion; offline gains alone are insufficient.
5. **Rollback-ready artifacts** — Every deployed model includes version, data manifest, metrics, and a tested rollback path.

---

## 1. RANK Expert

### Task
Rerank filtered memory candidates by relevance to the current query and session context.

### Input Schema (per candidate pair)
```json
{
  "trace_id": "string (required)",
  "session_id": "string (required)", 
  "query_id": "string (required)",
  "memory_id": "string (required)",
  "query": "string",
  "memory": "string",
  "query_timestamp": "int (unix seconds)",
  "candidate_timestamp": "int (unix seconds, must be < query_timestamp)",
  "label": "int (0 or 1)",
  "dataset": "string (public|private_weak|evaluation_only)"
}
```

### Label Sources
- **Public**: AlpsBench train/validation (human-verified retrieval pairs)
- **Private weak**: `retrieved → cited` proxy from shadow traces (marked `weak_proxy`, lower weight)
- **Human verified**: Independently annotated gold set for model selection
- **Evaluation-only**: AlpsBench test, reserved benchmarks (never in training)

### Primary Metrics
- **NDCG@10** (main)
- **Recall@1** (critical for top-1 admission)
- Query-grouped, bootstrap 95% CI with 3000+ resamples

### Promotion Gates

**R1: Native features + frozen semantic encoder**
- NDCG@10 95% bootstrap lower bound > baseline
- Recall@1 no statistical degradation
- All contract tests pass
- ≥90% of key slices within ±0.02 NDCG@10

**R2: Cross-Encoder (conditional)**
- Prerequisites: ≥500 human-verified queries, R1 complete, CPU baseline plateau
- Independent held-out test NDCG@10 and Recall@1 95% CI lower > R1
- p95 latency and cost within budget
- Zero scope/authority violations

**R3: Shadow A/B**
- ≥100 active sessions, all key slices represented
- Citation coverage ≥ baseline, correction rate not increased
- Latency within budget, no permission/scope incidents

### Risk Boundaries
- Model outputs relevance scores only; admission is still gated by scope, validity, conflict, and budget rules
- Private weak labels must be marked and weighted separately
- Test set leakage, session split violation, or temporal inversion → automatic rejection

---

## 2. ROUTE Expert

### Task
Predict top-k taxonomy paths and fallback probability for efficient candidate retrieval.

### Input Schema
```json
{
  "trace_id": "string (required)",
  "session_id": "string (required)",
  "query_id": "string (required)",
  "query": "string",
  "context": "string",
  "true_path": "string (taxonomy path)",
  "dataset": "string"
}
```

### Label Sources
- Human-verified path annotations
- Explicit API routing decisions
- Rule-based outputs (when confidence is logged)

### Primary Metrics
- Top-1 path accuracy
- Top-k coverage
- Fallback precision (when model is uncertain)
- Slow-retrieval trigger recall

### Promotion Gate
- Top-k coverage improvement, paired bootstrap lower bound > 0
- High-confidence error rate ≤ rule baseline
- Low-confidence samples trigger slow fallback without recall loss
- Hard-filter recall for cited memories unchanged

### Risk Boundaries
- Router outputs probabilities; admission still enforces scope/permission hard filters
- Taxonomy version changes require label migration or rebuild
- Model must calibrate uncertainty; overconfident errors are worse than uncertain correct

---

## 3. WRITE Expert

### Task
Propose which context spans are worth persisting (precision-first).

### Input Schema
```json
{
  "trace_id": "string (required)",
  "session_id": "string (required)",
  "span_id": "string (required)",
  "context": "string",
  "span": "string",
  "label": "string (propose|reject|unclear)",
  "scope": "string (session|project|global)",
  "durability": "string (temporary|stable|permanent)",
  "contains_secret": "bool"
}
```

### Label Sources
- Human annotation with evidence spans
- User-explicit save commands
- Post-hoc utility (cited in later sessions)

### Primary Metrics
- **Proposal precision** (main)
- Human acceptance rate
- Unsafe false-positive rate (must be zero on test set)
- Proposals per 1000 turns

### Promotion Gate
- Proposal precision ≥ rule baseline on independent gold set
- **Zero false positives** on secret/sensitive test cases
- If unsafe FP > 0, model is demoted to human-assisted ranking only

### Risk Boundaries
- Model proposes; user or rule confirms
- Secrets, credentials, PII must be rejected at 100% precision
- Model does NOT have write access; proposals go to confirmation flow

---

## 4. UPDATE Expert

### Task
Classify memory update operations: add / supersede / retain / escalate.

### Input Schema
```json
{
  "trace_id": "string (required)",
  "session_id": "string (required)",
  "update_id": "string (required)",
  "existing_memory": "string",
  "new_evidence": "string",
  "action": "string (add|supersede|retain|escalate)",
  "scope_match": "bool",
  "dataset": "string"
}
```

### Label Sources
- Human-verified update decisions
- User corrections
- Explicit fact updates with evidence

### Primary Metrics
- Per-class precision/recall
- **Supersede precision** (critical — wrong supersede loses data)
- False supersede rate
- Human acceptance rate

### Promotion Gate
- Supersede precision significantly above conservative threshold, reportable per scope
- **Zero cross-scope or immutable-memory modifications**
- If supersede precision insufficient, deploy add/escalate only, disable auto-supersede

### Risk Boundaries
- Model suggests action; actual supersede requires evidence, scope match, and API/human confirmation
- Uncertain/conflict/cross-scope → automatic escalation to human
- Permanent constraints cannot be learned away

---

## 5. ASSEMBLE Expert (Future)

### Task
Select final admitted context from ranked candidates under token budget and redundancy constraints.

### Input Schema
```json
{
  "trace_id": "string (required)",
  "session_id": "string (required)",
  "query_id": "string (required)",
  "ranked_candidates": "list[dict]",
  "budget": "int (tokens)",
  "admitted": "list[string] (memory IDs)"
}
```

### Primary Metrics
- Citation coverage
- Context pollution rate
- Redundancy rate
- Budget violations (must be zero)
- Conflict injection rate

### Promotion Gate
- Hard constraints 100% satisfied (scope, egress, validity, conflict, budget)
- Citation coverage improved, pollution not increased (counterfactual replay)
- Real shadow: correction rate not increased

### Risk Boundaries
- Model predicts value/redundancy; scope, egress, validity, conflict, token budget remain hard filters
- A model cannot override security or permission boundaries

---

## Global Data Contract

### Required Fields (all datasets)
- `trace_id`: Stable record identifier
- `session_id`: Session grouping for split isolation
- `source`: Origin system (hermes|claude|codex|opencode)
- `source_file`: Relative path for traceability
- Timestamps where applicable (candidates must precede query)

### Prohibited Behaviors
1. **Session split leakage**: Same session in multiple splits → reject
2. **Temporal inversion**: Candidate timestamp > query timestamp → reject
3. **Sensitive data**: Unredacted API keys, tokens, emails, private keys, connection strings → reject
4. **Evaluation data in training**: `evaluation_only` dataset in train split → reject
5. **Unknown as negative**: `not retrieved` or `not used` ≠ negative label; mark as `unknown` and exclude from supervised loss

### Validation Pipeline
```bash
# Schema and contract
python scripts/validate_dataset.py --input <JSONL> --dataset-type ranking --split train

# Split isolation
python scripts/validate_dataset.py --split-isolation train.jsonl validation.jsonl test.jsonl

# Secret scan (post-export)
grep -E "(sk-[a-zA-Z0-9]{10,}|Bearer [a-zA-Z0-9\-_\.]+)" <EXPORT> && exit 1
```

---

## Reproducibility Requirements

Every trained model must save:
- Git SHA
- Full command line
- Random seed
- Data manifest hash (file paths, line counts, splits)
- Training sample counts
- Dependency versions
- Training device (CPU/GPU)
- All metrics on train/validation/test
- Predictions (query-level, for bootstrap)
- Error slices

Missing any artifact → model is not promotable.

---

## Model Card Template

Every promoted model requires:
- Model ID and version
- Parent baseline
- Training data sources (public/private ratio, weak label policy)
- Metrics (with bootstrap CIs)
- Known limitations (slices that degrade, failure modes)
- Prohibited use cases
- License and base model provenance
- Rollback target

---

## Continuous Monitoring

Deployed models must log:
- Per-slice retrieval volume, citation coverage, correction rate
- Fallback rate, context size, p95 latency
- Scope filter rejects, data missing rate
- Version and timestamp

Drift detection:
1. Check trace completeness, label latency, session mix, taxonomy version changes
2. Only after data quality is confirmed → diagnose model degradation

Retraining triggers:
- Data volume threshold
- Sustained quality drop
- Taxonomy schema change
- Feature contract change

Every retraining → full Phase 0 reproducibility + independent eval + shadow + promotion audit.

---

## What We Do NOT Do

- GPU fine-tuning before independent gold set exists
- Use "not cited" as negative label
- Use private weak labels for primary model selection
- Let models bypass scope/permission/validity
- Let WRITE/UPDATE auto-commit permanent changes
- Promote on single small validation point estimate

---

## Summary

| Expert | Input | Output | Main Metric | Hard Constraint | Promotion Gate |
|--------|-------|--------|-------------|-----------------|----------------|
| RANK | query + candidates | scores | NDCG@10 | scope, validity | R1 → R2 → R3 |
| ROUTE | query + context | top-k paths + p(fallback) | coverage | scope filters | Bootstrap > 0, calibrated |
| WRITE | context + span | propose / reject | precision | zero unsafe FP | Precision ≥ baseline, FP=0 |
| UPDATE | old + new | add/supersede/retain/escalate | supersede precision | no cross-scope | Supersede safe, or disable |
| ASSEMBLE | ranked + budget | admitted IDs | coverage, pollution | budget, conflict | Hard 100%, shadow ok |

**Every model**: Reproducible artifacts + contract validation + independent eval + shadow evidence + rollback path.
