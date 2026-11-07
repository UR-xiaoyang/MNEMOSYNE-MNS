# Shadow Mode Trace Collection Schema

## Overview

Shadow mode runs the RANK model alongside (or instead of) the current retrieval system, collecting real `retrieved → cited → corrected` traces for production validation and continuous learning.

## Export Pipeline (P2)

### Building Training Data

Export shadow traces to JSONL training samples with explicit authorization:

```bash
python scripts/build_shadow_training_data.py \
  --source-db <EXPLICIT_SOURCE_DB> \
  --output-dir <OUTPUT_DIR> \
  --include-text  # Optional: include redacted text (default: IDs only)
```

**Privacy Boundaries:**
- Source DB path must be explicitly provided (no default discovery)
- Opens DB in read-only mode (`mode=ro`)
- Output contains privacy-safe IDs only by default
- `--include-text` flag runs redaction before including text
- Manifest and reports contain NO absolute source paths
- CLI stdout does not print source or output paths

**Label Generation:**

Every retrieval candidate generates one training sample with a label based on available evidence:

- **Positive**: Citation events generate `label="positive", label_source="citation"`
- **Corrected**: Correction events generate `label="corrected", label_source="correction"`
- **Unknown**: Candidates without citation or feedback evidence generate `label="unknown", label_source="unknown", unknown_reason="no_citation_or_feedback_evidence"`

Uncited candidates are NOT absent—they are explicitly labeled as `unknown` to preserve the candidate pool for future analysis. Unknown ≠ irrelevant; it means insufficient evidence.

**Sample Schema (Required Fields):**
- `trace_id`: Stable composite ID (hash of retrieval_event_id:memory_id:rank)
- `retrieval_event_id`: Links to retrieval event
- `session_id`, `turn_id`: Session context
- `query_timestamp`, `candidate_timestamp`: Temporal ordering
- `memory_id`: Candidate memory identifier
- `rank`, `score`: Ranking position and model score
- `model_id`, `scope`: Model and scope metadata
- `injected`: Whether candidate was injected into context
- `label`: One of "positive", "corrected", "unknown"
- `label_source`: Evidence source ("citation", "correction", "unknown")
- `label_confidence`: Float or None for unknown
- `evidence_ids`: List of event IDs supporting this label
- `sample_weight`: Training weight (0.5 for unknown, 1.0 for others)
- `unknown_reason`: Present only when label="unknown"

**Temporal Audit:**
- Future candidates (candidate_timestamp > query_timestamp) are rejected
- Rejections written to `rejections.jsonl` with reason codes
- Audit manifest records rejection counts by reason

**Text Redaction (--include-text):**
- Applies secret detection to query and memory text
- Redacted fields: `query_text_redacted`, `memory_text_redacted`
- Records: `redaction_applied=true`, detected secret types
- Post-export secret scan recommended before publication

**Output Structure:**
```
training_data/
  shadow_training_samples.jsonl  # One sample per retrieval candidate
  rejections.jsonl                # Future candidates, invalid JSON, etc.
  manifest.json                   # Statistics, no source paths
```

### Evaluating Quality

Generate quality report from exported data:

```bash
python scripts/evaluate_shadow_traces.py \
  --data-dir <OUTPUT_DIR> \
  --min-samples 10 \
  --output report.json
```

Returns `insufficient_evidence` when sample count < min-samples threshold (default 10). CLI stdout does not print output file paths.

**Report Fields:**
- `status`: `"success"` or `"insufficient_evidence"`
- `total_samples`: Total sample count
- `label_counts`: Distribution by label (positive, corrected, unknown)
- `label_source_counts`: Distribution by evidence source (citation, correction, unknown)
- `reason`: Explanation when status is insufficient_evidence

When status is insufficient_evidence, do not compute percentages or make training recommendations—report honestly that more data is needed.

## Trace Event Schema

### 1. Retrieval Event

```python
{
  "event_type": "retrieval",
  "event_id": str,           # UUID
  "timestamp": float,        # Unix timestamp
  "session_id": str,
  "turn_id": str,
  
  "query": {
    "text": str,
    "scope": {
      "level": str,          # "session" | "project" | "global"
      "session_id": str | None,
      "project_id": str | None,
    },
    "destination": str,      # "local" | "cloud_anthropic" | etc
    "path_hint": str | None, # Classification path from routing
    "requires_high_authority": bool,
  },
  
  "candidates": [
    {
      "memory_id": str,
      "rank": int,           # Position after ranking (1-indexed)
      "score": float,        # Ranker output score
      "features": {
        "semantic": dict,    # E5 cosine, etc
        "lexical": dict,     # TF-IDF, containment, etc
        "native": dict,      # Scope, authority, validity, etc
      },
      "filter_passed": bool, # Did it pass hard filters?
      "injected": bool,      # Was it actually injected into context?
    }
  ],
  
  "metadata": {
    "ranker_model_id": str,  # e.g., "RANK-HYBRID-01"
    "ranker_version": str,
    "retrieval_latency_ms": float,
    "candidate_pool_size": int,
    "injected_count": int,
  }
}
```

### 2. Citation Event

Recorded when the model's response references a memory.

```python
{
  "event_type": "citation",
  "event_id": str,
  "timestamp": float,
  "session_id": str,
  "turn_id": str,
  "retrieval_event_id": str,  # Links back to retrieval
  
  "memory_id": str,
  "citation": {
    "span": str,             # Text span in response that used this memory
    "confidence": float,     # Model's self-assessed confidence (if available)
    "explicit_reference": bool,  # Did model explicitly cite "according to..."
  },
  
  "context_position": int,    # Position in injected context (1-indexed)
  "original_rank": int,       # Rank from retrieval event
}
```

### 3. Correction Event

Recorded when the user corrects the model after a cited memory.

```python
{
  "event_type": "correction",
  "event_id": str,
  "timestamp": float,
  "session_id": str,
  "turn_id": str,
  "citation_event_id": str,   # Links back to citation
  
  "memory_id": str,
  "correction_type": str,     # "contradiction" | "outdated" | "incomplete" | "misapplied" | "scope_error"
  
  "user_feedback": {
    "text": str | None,       # User's correction message
    "corrected_content": str | None,  # What it should have been
    "action": str,            # "supersede" | "deprecate" | "update_scope" | "mark_stale"
  },
  
  "metadata": {
    "turns_since_citation": int,  # How many turns before user corrected
    "automatic_detection": bool,  # Was correction triggered by model detecting conflict?
  }
}
```

### 4. Ignored Event

Recorded when a highly-ranked memory was NOT cited despite injection.

```python
{
  "event_type": "ignored",
  "event_id": str,
  "timestamp": float,
  "session_id": str,
  "turn_id": str,
  "retrieval_event_id": str,
  
  "memory_id": str,
  "rank": int,
  "injected": bool,
  "context_position": int | None,
  
  "possible_reasons": list[str],  # Inferred: ["redundant", "not_applicable", "low_confidence"]
}
```

## Aggregated Metrics

From collected traces, compute production metrics:

```python
def compute_production_metrics(traces: list[Event]) -> dict:
    return {
        # Retrieval quality
        "citation_rate": cited_count / injected_count,
        "top1_citation_rate": cited_at_rank1 / queries,
        "top5_citation_rate": cited_in_top5 / queries,
        "mrr_citation": mean(1 / rank for cited memories),
        
        # Correction signals
        "correction_rate": corrections / citations,
        "correction_rate_by_rank": {rank: rate for rank in [1,2,3,5,10]},
        "time_to_correction_median": median(turns_since_citation),
        
        # Filtering effectiveness
        "hard_filter_recall": cited_count / (cited + wrongly_filtered),
        "hard_filter_precision": useful_filtered / all_filtered,
        
        # Latency
        "p50_retrieval_latency_ms": percentile(latencies, 50),
        "p99_retrieval_latency_ms": percentile(latencies, 99),
    }
```

## Storage

### SQLite Schema (for local shadow mode)

```sql
CREATE TABLE retrieval_events (
    event_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_json TEXT NOT NULL,  -- Full query structure
    candidates_json TEXT NOT NULL,  -- Array of candidates
    metadata_json TEXT NOT NULL,
    INDEX idx_session (session_id),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE citation_events (
    event_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    retrieval_event_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    citation_json TEXT NOT NULL,
    context_position INTEGER,
    original_rank INTEGER,
    FOREIGN KEY (retrieval_event_id) REFERENCES retrieval_events(event_id),
    INDEX idx_memory (memory_id),
    INDEX idx_retrieval (retrieval_event_id)
);

CREATE TABLE correction_events (
    event_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    citation_event_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    correction_type TEXT NOT NULL,
    feedback_json TEXT NOT NULL,
    FOREIGN KEY (citation_event_id) REFERENCES citation_events(event_id),
    INDEX idx_memory (memory_id),
    INDEX idx_citation (citation_event_id)
);

CREATE TABLE ignored_events (
    event_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    retrieval_event_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    injected BOOLEAN NOT NULL,
    FOREIGN KEY (retrieval_event_id) REFERENCES retrieval_events(event_id),
    INDEX idx_memory (memory_id)
);
```

## Trace Collection Workflow

```text
User query arrives
  ↓
1. RETRIEVAL: Run RANK model, log retrieval_event
  ↓
2. Hard filters + budget assembly
  ↓
3. Context injection → Model inference
  ↓
4. CITATION: Parse response, detect citations, log citation_events
  ↓
5. (Next turn) User corrects → log correction_event
  ↓
6. (End of turn) Detect ignored memories → log ignored_events
```

## Privacy & Security

- **Local-only storage**: Shadow mode traces stay on user's machine
- **No cloud upload**: Traces contain full query/memory text and must not leave local environment
- **Opt-in**: User must explicitly enable shadow mode
- **Redaction**: Credential/PII memories are redacted before logging (content replaced with `[REDACTED:{memory_id}]`)
- **Export**: User can export anonymized traces for training (query/memory ids only, no text)

## Usage Example

```python
from shadow_mode import ShadowModeCollector

collector = ShadowModeCollector(db_path="~/.hermes/shadow_traces.db")

# During retrieval
retrieval_event = collector.log_retrieval(
    query=query,
    candidates=ranked_candidates,
    injected_memory_ids=injected_ids,
    metadata={"ranker_model_id": "RANK-HYBRID-01"}
)

# During response generation
for memory_id in detect_citations_in_response(response):
    collector.log_citation(
        retrieval_event_id=retrieval_event.event_id,
        memory_id=memory_id,
        span=extract_citation_span(response, memory_id),
        context_position=get_context_position(memory_id)
    )

# On user correction
collector.log_correction(
    citation_event_id=citation_event_id,
    memory_id=memory_id,
    correction_type="outdated",
    user_feedback={"text": user_message, "action": "supersede"}
)

# After turn completes
for memory_id in injected_but_not_cited:
    collector.log_ignored(
        retrieval_event_id=retrieval_event.event_id,
        memory_id=memory_id,
        rank=original_rank
    )
```

## Evaluation Pipeline

```bash
# After collecting traces for N sessions
python3 scripts/evaluate_shadow_traces.py \
  --traces ~/.hermes/shadow_traces.db \
  --model-id RANK-HYBRID-01 \
  --output shadow_eval_report.json

# Compare against baseline
python3 scripts/compare_shadow_baselines.py \
  --baseline-traces baseline_traces.db \
  --candidate-traces candidate_traces.db \
  --output comparison.json
```

## Next Steps

1. Implement `ShadowModeCollector` class
2. Add citation detection heuristics
3. Integrate with Hermes agent runtime
4. Run shadow mode for 100+ sessions
5. Evaluate production metrics
6. Retrain RANK model on real traces

## References

- AlpsBench provides static query-memory-label triplets
- Shadow mode provides dynamic `retrieved → cited → corrected` sequences
- Real production metrics trump validation set metrics
