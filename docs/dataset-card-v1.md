# MNEMOSYNE-MNS Dataset Card v1

## Overview

This dataset card template documents training datasets for MNEMOSYNE-MNS memory system models. Every dataset must complete this card before being used in model training or evaluation.

**Current Status:** Phase 1 complete - read-only, privacy-safe data pipeline established. Private conversation normalization, redaction, and session-isolated splitting infrastructure is ready. No private data has been processed yet; toolkit validated with toy fixtures only.

---

## Data Pipeline (Phase 1 Complete)

### Prerequisites

**Explicit Authorization Required**: Set environment variables to authorize specific conversation sources. No default or home-directory fallback exists.

- `HERMES_STATE_DB` - Path to Hermes SQLite database
- `CLAUDE_HISTORY_DIR` - Claude conversation directory
- `CODEX_SESSIONS_DIR` - Codex sessions directory
- `OPENCODE_DATA_DIR` - OpenCode data directory

**When variables are not set**: The inventory step returns empty results and does not probe the filesystem. This ensures no implicit access to home directories or standard locations.

### Pipeline Steps

#### Step 1: Inventory Sources (Read-Only)
```bash
python scripts/dataset_inventory.py \
  --output memory_dataset/raw_inventory/inventory.json
```

**What it does:**
- Discovers available conversation sources from environment variables
- Catalogs file paths, formats, sizes, and record counts
- **Does NOT copy or read actual conversation content**
- Uses read-only SQLite URIs (`file:path?mode=ro`)
- Safe to run on production databases

**Output:** `inventory.json` with source metadata only

#### Step 2: Normalize Conversations

**Authorization Contract**: Normalization requires explicit `--source <token>=<authorized-path>` mappings for every source in the inventory. The authorized path is used at runtime only and never appears in output files, stdout, or the normalized JSONL.

```bash
# Extract source tokens from the safe inventory output
# Each source has a privacy-safe token (e.g., "hermes_a1b2c3")

# Provide explicit token→path mapping for normalization
python scripts/normalize_conversations.py \
  --inventory memory_dataset/raw_inventory/inventory.json \
  --source hermes_a1b2c3=/path/to/authorized/hermes/state.db \
  --source claude_x7y8z9=/path/to/authorized/claude/history \
  --output memory_dataset/normalized/visible_messages.jsonl
```

**Multiple sources**: Repeat `--source <token>=<path>` for each source token from the inventory.

**Missing mapping rejection**: If a source token in the inventory lacks a corresponding `--source` argument, normalization fails with a clear error. This prevents accidental processing of sources without explicit runtime authorization.

**What it does:**
- Extracts only user-visible messages (user, assistant, tool roles)
- Excludes: system/developer messages, hidden reasoning, framework injections, skill text, agent/IDE events
- Generates stable message IDs for deduplication
- Truncates oversized tool output with auditable markers
- Preserves traceability: `id, source, source_file, session_id, ordinal, timestamp, role, content, metadata`
- **Privacy guarantee**: `source_file` contains a privacy-safe token (not the authorized path); absolute paths never leak to JSONL or stdout

**Output:** `visible_messages.jsonl` with normalized schema

#### Step 3: Redact Sensitive Data
```bash
python scripts/redact_and_validate_dataset.py \
  --input memory_dataset/normalized/visible_messages.jsonl \
  --output-dir memory_dataset/cleaned
```

**What it does:**
- Redacts: API keys, bearer tokens, private keys, connection strings, emails, IP addresses, home paths
- Validates schema compliance for every message
- Logs rejections with reason codes (no sensitive data in logs)
- Runs secret scan audit to verify zero high-risk patterns remain

**Output:** 
- `cleaned/messages.jsonl` - Redacted messages
- `cleaned/rejections.jsonl` - Rejection log with reason codes
- `cleaned/audit_report.json` - Secret scan and schema validation results

**Audit-only mode:**
```bash
python scripts/redact_and_validate_dataset.py \
  --input memory_dataset/cleaned/messages.jsonl \
  --output-dir memory_dataset/cleaned \
  --audit-only
```

#### Step 4: Build Session-Isolated Splits
```bash
python scripts/build_session_splits.py \
  --input memory_dataset/cleaned/messages.jsonl \
  --output-dir memory_dataset/splits \
  --seed 42 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

**What it does:**
- Groups messages by `(source, session_id)`
- Sorts sessions by earliest timestamp (chronological)
- Assigns entire sessions to train/validation/test (no session splitting)
- Creates manifest with SHA-256 hash of split assignments
- Verifies isolation: no session appears in multiple splits

**Output:**
- `splits/train.jsonl`
- `splits/validation.jsonl`
- `splits/test.jsonl`
- `splits/split_manifest.json` - Contains split hash, seed, session mapping

**Verify splits:**
```bash
python scripts/build_session_splits.py \
  --output-dir memory_dataset/splits \
  --verify
```

### Safety Guarantees

1. **Read-Only Source Access**: All source databases opened with `mode=ro` URI flag
2. **No Content Leakage**: Inventory contains only metadata (privacy-safe source tokens, counts, schemas), never message content or absolute paths
3. **Comprehensive Redaction**: 7 sensitive data categories with 20+ regex patterns
4. **Auditable Rejections**: Every rejected record logged with stable reference ID and reason code
5. **Secret Scan Enforcement**: Pipeline fails if any high-risk pattern remains after redaction
6. **Session Isolation**: Mathematical guarantee that no session appears in multiple splits
7. **Reproducibility**: Deterministic IDs, seeded splits, cryptographic manifest hashes

### Testing Without Private Data

All scripts include toy fixture tests that work without real conversation sources:

```bash
# Run Phase 1 pipeline tests
python -m pytest tests/test_dataset_inventory.py \
                 tests/test_dataset_normalization.py \
                 tests/test_dataset_redaction.py \
                 tests/test_split_isolation.py -v

# All tests use tempdir fixtures - no private data required
python -m unittest discover -s tests -q
```

---

## Dataset Identification

**Dataset Name**: [e.g., MNEMOSYNE-RANK-TRAIN-v1]  
**Version**: [e.g., 1.0.0]  
**Creation Date**: [YYYY-MM-DD]  
**Curator**: [Individual or team]  
**License**: [License for dataset use]  

---

## Purpose

**Intended Use**: [What task/model is this for? RANK reranking, ROUTE classification, etc.]  
**Out-of-Scope Use**: [What should this NOT be used for? E.g., "Not for permanent deletion decisions," "Not for cross-organization retrieval"]

---

## Data Sources

### Source Breakdown

| Source | Type | Sample Count | Date Range | License | Notes |
|--------|------|--------------|------------|---------|-------|
| AlpsBench public train | Public benchmark | 25,123 | 2024-2026 | MIT | Human-verified retrieval pairs |
| AlpsBench public validation | Public benchmark | 30,069 | 2024-2026 | MIT | Reserved for model selection |
| Private shadow traces | Private weak | 0 | N/A | Internal | Not included in v1 |
| Human gold annotations | Private verified | 0 | N/A | Internal | Planned for v2 |

**Public/Private Ratio**: 100% public / 0% private  
**Weak Label Policy**: No weak labels in this version; all labels are human-verified from AlpsBench.

### Excluded Sources
- AlpsBench test set (evaluation-only, never in training or validation)
- Internal Hermes/Claude/Codex sessions (not yet normalized or redacted)

---

## Data Collection & Annotation

### Collection Method
- **AlpsBench**: Public multilingual memory retrieval benchmark with expert annotations
- **Annotation Process**: Professional annotators judged relevance of query-memory pairs; inter-annotator agreement reported in AlpsBench paper

### Temporal Coverage
- Query timestamps: [range if available]
- Candidate memories: All candidates predate their queries (temporal constraint validated)

### Language Distribution
- Multilingual (exact distribution in AlpsBench documentation)
- Dominated by English, with significant Chinese, Spanish, French

---

## Schema & Fields

### Required Fields (per sample)
```json
{
  "trace_id": "Stable record ID",
  "session_id": "Session for split isolation",
  "query_id": "Query identifier",
  "memory_id": "Candidate memory identifier",
  "query": "Query text",
  "memory": "Memory candidate text",
  "label": "0 or 1 (not relevant / relevant)",
  "dataset": "public",
  "query_timestamp": "Unix seconds (if available)",
  "candidate_timestamp": "Unix seconds, must be < query_timestamp"
}
```

### Optional Metadata
- `scope`: session | project | global
- `source_file`: Path to original data
- `evidence`: Human annotation notes

---

## Preprocessing & Cleaning

### Steps Applied
1. **Normalization**: Converted AlpsBench format to standardized JSONL schema
2. **Redaction**: Scanned for sensitive patterns (API keys, tokens, emails); none found in public benchmark
3. **Validation**: All samples passed contract validation (schema, temporal, split isolation)
4. **Deduplication**: No exact duplicates found

### Rejected Samples
- **Count**: 0
- **Reasons**: N/A (AlpsBench data is pre-cleaned)

---

## Splits

### Split Policy
- **Unit**: `(source, session_id)` — entire sessions stay together
- **Method**: Chronological within AlpsBench official splits
- **Random Seed**: N/A (using official splits)

| Split | Sessions | Queries | Pairs | Positives | Negatives |
|-------|----------|---------|-------|-----------|-----------|
| Train | [count] | [count] | 25,123 | [count] | [count] |
| Validation | [count] | 312 | 30,069 | [count] | [count] |
| Test | N/A | N/A | Reserved | N/A | N/A |

**Split Isolation Verified**: ✓ (no session appears in multiple splits)

---

## Label Distribution

### Overall
- **Positive (relevant)**: [count] ([percentage]%)
- **Negative (not relevant)**: [count] ([percentage]%)
- **Unknown/Uncertain**: 0 (all AlpsBench labels are binary)

### By Query
- **Queries with 0 positives**: Skipped from metrics (cannot compute NDCG)
- **Queries with 1 positive**: [count]
- **Queries with 2+ positives**: [count]

---

## Known Limitations

### Representation Gaps
1. **Task-specific**: AlpsBench is a retrieval benchmark; real runtime patterns (user corrections, multi-turn context) are not represented
2. **Small validation set**: 312 queries may not cover all scope/language/length slices
3. **No scope metadata**: Public benchmark does not include session/project/global scope labels
4. **No weak negative verification**: Negative labels are "not marked relevant"; unclear if truly irrelevant or simply not annotated

### Temporal & Distribution Drift
- Dataset is static; production queries may drift in topic, style, or taxonomy
- No runtime feedback loop (e.g., citation, correction) is captured

### Sensitive Data
- **Secret Scan Result**: ✓ Passed (0 matches for API keys, tokens, emails, private keys, connection strings)
- **PII Scan Result**: ✓ Passed (no emails, names, addresses detected)

---

## Ethical & Privacy Considerations

### Data Origin
- Public benchmark, no private user data

### Privacy Compliance
- No user consent required (public dataset)
- No personal identifiers

### Potential Harms
- Model trained on this data may not generalize to private sessions
- Over-reliance on benchmark performance may miss real-world failure modes

---

## Versioning & Updates

**Current Version**: 1.0.0  
**Change Log**:
- v1.0.0 (2026-11-03): Initial release, AlpsBench public splits only

**Planned Updates**:
- v2.0.0: Add private shadow traces (weak labels, lower weight)
- v2.1.0: Add human-verified gold annotations for independent model selection

**Deprecation Policy**: Datasets used in production models must remain available for audit; deprecated datasets are archived, not deleted.

---

## Reproducibility

### Manifest Hash
- **Train manifest**: [SHA-256 of file paths + line counts + split assignments]
- **Validation manifest**: [SHA-256]

### Reconstruction
```bash
# Example commands to rebuild from source
python scripts/normalize_conversations.py \
  --source alpsbench_public \
  --output memory_dataset/normalized/alpsbench_v1.jsonl

python scripts/build_session_splits.py \
  --input memory_dataset/normalized/alpsbench_v1.jsonl \
  --output-dir memory_dataset/splits \
  --seed 42
```

---

## Contact & Maintenance

**Maintainer**: [Name/Team]  
**Contact**: [Email or issue tracker]  
**Last Reviewed**: [YYYY-MM-DD]

---

## Checklist

Before using this dataset in training:

- [ ] All required fields present and validated
- [ ] Session split isolation verified (no leakage)
- [ ] Temporal constraints satisfied (candidates < query)
- [ ] Secret scan passed (zero high-risk matches)
- [ ] Evaluation-only data excluded from training split
- [ ] Manifest hash recorded
- [ ] Known limitations documented
- [ ] License verified for intended use

---

## References

- AlpsBench: [Citation and URL if applicable]
- MNEMOSYNE-MNS Model Training Protocol: `docs/model-training-protocol-v1.md`
- Dataset validation script: `scripts/validate_dataset.py`
