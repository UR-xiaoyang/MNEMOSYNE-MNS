#!/usr/bin/env python3
"""
Shadow trace training data export pipeline.

Converts shadow mode SQLite traces into traceable JSONL training samples.
Enforces explicit authorization and privacy boundaries from P1.
P2: Complete schema, unknown labels, temporal audit, text redaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from redaction_utils import detect_and_redact_secrets


def export_shadow_traces(
    source_db_path: str | Path | None,
    output_dir: str | Path,
    include_text: bool = False,
) -> dict[str, Any]:
    """
    Export shadow traces to training data format.
    
    Args:
        source_db_path: Explicit path to shadow DB. Required.
        output_dir: Where to write output files (must be Git-ignored).
        include_text: If True, include redacted text. Default False (IDs only).
    
    Returns:
        Export manifest with counts and output paths.
    
    Raises:
        ValueError: If source_db_path is not explicitly provided.
    """
    if source_db_path is None:
        raise ValueError(
            "Explicit source DB path required. "
            "Pass --source-db <path> via CLI or source_db_path= in code. "
            "No default discovery is performed."
        )
    
    source_db_path = Path(source_db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open in read-only mode
    conn = sqlite3.connect(f"file:{source_db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    
    samples = []
    rejections = []
    
    # Build citation/correction lookup for evidence
    cited_memories = {}  # (retrieval_event_id, memory_id) -> citation_event_id
    corrected_memories = {}  # citation_event_id -> correction details
    
    try:
        # Load citations
        try:
            cursor = conn.execute("""
                SELECT 
                    event_id,
                    retrieval_event_id,
                    memory_id,
                    timestamp
                FROM citation_events
            """)
            
            for row in cursor:
                key = (row["retrieval_event_id"], row["memory_id"])
                cited_memories[key] = row["event_id"]
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e):
                raise
        
        # Load corrections
        try:
            cursor = conn.execute("""
                SELECT 
                    citation_event_id,
                    correction_type,
                    event_id as correction_event_id
                FROM correction_events
            """)
            
            for row in cursor:
                corrected_memories[row["citation_event_id"]] = {
                    "correction_type": row["correction_type"],
                    "correction_event_id": row["correction_event_id"],
                }
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e):
                raise
        
        # Load memory metadata for temporal checks
        memory_metadata = {}  # memory_id -> {created_at, scope}
        try:
            cursor = conn.execute("""
                SELECT memory_id, created_at, scope
                FROM memory_metadata
            """)
            for row in cursor:
                memory_metadata[row["memory_id"]] = {
                    "created_at": row["created_at"],
                    "scope": row["scope"],
                }
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e):
                raise
        
        # Load memory content if text requested
        memory_content = {}
        if include_text:
            try:
                cursor = conn.execute("""
                    SELECT memory_id, content_text
                    FROM memory_content
                """)
                for row in cursor:
                    memory_content[row["memory_id"]] = row["content_text"]
            except sqlite3.OperationalError as e:
                if "no such table" not in str(e):
                    raise
        
        # Process retrieval events - generate sample per candidate
        try:
            cursor = conn.execute("""
                SELECT 
                    event_id,
                    timestamp,
                    session_id,
                    turn_id,
                    query_text,
                    query_json,
                    candidates_json,
                    metadata_json
                FROM retrieval_events
            """)
            
            for row in cursor:
                retrieval_id = row["event_id"]
                query_timestamp = row["timestamp"]
                session_id = row["session_id"]
                turn_id = row["turn_id"]
                
                # Parse metadata
                try:
                    metadata = json.loads(row["metadata_json"])
                    candidates = json.loads(row["candidates_json"])
                except json.JSONDecodeError:
                    rejections.append({
                        "retrieval_event_id": retrieval_id,
                        "reason": "invalid_json",
                        "rejected_at": time.time(),
                    })
                    continue
                
                model_id = metadata.get("model_id", "unknown")
                scope = metadata.get("scope", "unknown")
                
                # Process each candidate
                for candidate in candidates:
                    memory_id = candidate.get("memory_id")
                    
                    # Validate required fields
                    if not memory_id:
                        rejections.append({
                            "retrieval_event_id": retrieval_id,
                            "reason": "missing_memory_id",
                            "rejected_at": time.time(),
                        })
                        continue
                    
                    rank = candidate.get("rank")
                    score = candidate.get("score")
                    
                    if rank is None or score is None:
                        rejections.append({
                            "retrieval_event_id": retrieval_id,
                            "memory_id": memory_id,
                            "reason": "missing_field_rank_or_score",
                            "rejected_at": time.time(),
                        })
                        continue
                    
                    injected = candidate.get("injected", False)
                    
                    # Temporal audit: check if candidate timestamp <= query timestamp
                    if memory_id in memory_metadata:
                        candidate_timestamp = memory_metadata[memory_id]["created_at"]
                        
                        if candidate_timestamp > query_timestamp:
                            rejections.append({
                                "retrieval_event_id": retrieval_id,
                                "memory_id": memory_id,
                                "reason": "future_candidate",
                                "query_timestamp": query_timestamp,
                                "candidate_timestamp": candidate_timestamp,
                                "rejected_at": time.time(),
                            })
                            continue
                    else:
                        # No metadata available, use query timestamp as proxy
                        candidate_timestamp = query_timestamp
                    
                    # Generate trace_id (stable composite ID)
                    trace_id = hashlib.sha256(
                        f"{retrieval_id}:{memory_id}:{rank}".encode()
                    ).hexdigest()[:16]
                    
                    # Determine label based on evidence
                    citation_key = (retrieval_id, memory_id)
                    citation_event_id = cited_memories.get(citation_key)
                    
                    if citation_event_id:
                        # Check if corrected
                        correction = corrected_memories.get(citation_event_id)
                        
                        if correction:
                            label = "corrected"
                            label_source = "correction"
                            label_confidence = 0.95
                            evidence_ids = [citation_event_id, correction["correction_event_id"]]
                            sample_weight = 1.0
                            
                            sample = {
                                "trace_id": trace_id,
                                "retrieval_event_id": retrieval_id,
                                "citation_event_id": citation_event_id,
                                "correction_event_id": correction["correction_event_id"],
                                "session_id": session_id,
                                "turn_id": turn_id,
                                "query_timestamp": query_timestamp,
                                "candidate_timestamp": candidate_timestamp,
                                "memory_id": memory_id,
                                "rank": rank,
                                "score": score,
                                "model_id": model_id,
                                "scope": scope,
                                "injected": injected,
                                "label": label,
                                "label_source": label_source,
                                "label_confidence": label_confidence,
                                "evidence_ids": evidence_ids,
                                "sample_weight": sample_weight,
                                "correction_type": correction["correction_type"],
                            }
                        else:
                            # Cited but not corrected
                            label = "positive"
                            label_source = "citation"
                            label_confidence = 0.9
                            evidence_ids = [citation_event_id]
                            sample_weight = 1.0
                            
                            sample = {
                                "trace_id": trace_id,
                                "retrieval_event_id": retrieval_id,
                                "citation_event_id": citation_event_id,
                                "session_id": session_id,
                                "turn_id": turn_id,
                                "query_timestamp": query_timestamp,
                                "candidate_timestamp": candidate_timestamp,
                                "memory_id": memory_id,
                                "rank": rank,
                                "score": score,
                                "model_id": model_id,
                                "scope": scope,
                                "injected": injected,
                                "label": label,
                                "label_source": label_source,
                                "label_confidence": label_confidence,
                                "evidence_ids": evidence_ids,
                                "sample_weight": sample_weight,
                            }
                    else:
                        # No citation - label as unknown
                        label = "unknown"
                        label_source = "unknown"
                        label_confidence = None
                        evidence_ids = []
                        sample_weight = 0.5  # Lower weight for unknown
                        
                        sample = {
                            "trace_id": trace_id,
                            "retrieval_event_id": retrieval_id,
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "query_timestamp": query_timestamp,
                            "candidate_timestamp": candidate_timestamp,
                            "memory_id": memory_id,
                            "rank": rank,
                            "score": score,
                            "model_id": model_id,
                            "scope": scope,
                            "injected": injected,
                            "label": label,
                            "label_source": label_source,
                            "label_confidence": label_confidence,
                            "evidence_ids": evidence_ids,
                            "sample_weight": sample_weight,
                            "unknown_reason": "no_citation_or_feedback_evidence",
                        }
                    
                    # Add text if requested (with redaction)
                    if include_text:
                        query_text = row["query_text"]
                        mem_text = memory_content.get(memory_id, "")
                        
                        # Apply redaction
                        query_redacted, query_secrets = detect_and_redact_secrets(query_text)
                        mem_redacted, mem_secrets = detect_and_redact_secrets(mem_text)
                        
                        sample["query_text_redacted"] = query_redacted
                        sample["memory_text_redacted"] = mem_redacted
                        sample["redaction_applied"] = True
                        sample["query_secrets_detected"] = query_secrets
                        sample["memory_secrets_detected"] = mem_secrets
                    
                    samples.append(sample)
        
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e):
                raise
    
    finally:
        conn.close()
    
    # Write samples to JSONL
    samples_path = output_dir / "shadow_training_samples.jsonl"
    with open(samples_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Write rejections to JSONL
    rejections_path = output_dir / "rejections.jsonl"
    with open(rejections_path, "w") as f:
        for rejection in rejections:
            f.write(json.dumps(rejection, ensure_ascii=False) + "\n")
    
    # Generate stratified manifest
    label_counts = Counter(s["label"] for s in samples)
    label_source_counts = Counter(s["label_source"] for s in samples)
    model_counts = Counter(s["model_id"] for s in samples)
    scope_counts = Counter(s["scope"] for s in samples)
    
    unknown_count = sum(1 for s in samples if s["label"] == "unknown")
    rejected_count = len(rejections)
    
    # Check for secrets if text was included
    redaction_summary = {}
    if include_text:
        all_query_secrets = []
        all_mem_secrets = []
        for s in samples:
            all_query_secrets.extend(s.get("query_secrets_detected", []))
            all_mem_secrets.extend(s.get("memory_secrets_detected", []))
        
        redaction_summary = {
            "redaction_applied": True,
            "secrets_detected": len(all_query_secrets) + len(all_mem_secrets) > 0,
            "query_secret_types": list(set(all_query_secrets)),
            "memory_secret_types": list(set(all_mem_secrets)),
        }
    
    manifest = {
        "timestamp": time.time(),
        "total_samples": len(samples),
        "label_counts": dict(label_counts),
        "label_source_counts": dict(label_source_counts),
        "model_counts": dict(model_counts),
        "scope_counts": dict(scope_counts),
        "unknown_count": unknown_count,
        "rejected_count": rejected_count,
        "output_files": {
            "samples": "shadow_training_samples.jsonl",
            "rejections": "rejections.jsonl",
            "manifest": "manifest.json",
        },
        "privacy_note": "No absolute source paths included. Source DB accessed read-only.",
    }
    
    if redaction_summary:
        manifest.update(redaction_summary)
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return {
        "status": "success",
        "output_dir": str(output_dir),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "total_samples": len(samples),
    }


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Export shadow traces to training data format"
    )
    parser.add_argument(
        "--source-db",
        required=True,
        help="Explicit path to shadow DB (required, no defaults)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for training data (should be Git-ignored)",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include redacted text in output (default: IDs only)",
    )
    
    args = parser.parse_args()
    
    try:
        result = export_shadow_traces(
            source_db_path=args.source_db,
            output_dir=args.output_dir,
            include_text=args.include_text,
        )
        
        # CLI stdout must not contain paths
        print(f"✓ Export complete")
        print(f"  Samples: {result['total_samples']}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}", file=sys.stderr)
        sys.exit(1)
