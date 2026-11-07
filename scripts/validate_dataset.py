"""
Dataset contract validator for MNEMOSYNE-MNS training data.

Enforces:
- Required schema fields (trace_id, session_id, timestamps)
- Temporal ordering (candidates must precede query)
- Session split isolation (no session in multiple splits)
- Sensitive data redaction (API keys, tokens, emails)
- Evaluation-only data exclusion from training
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# Sensitive data patterns
SENSITIVE_PATTERNS = {
    "api_key": re.compile(r"sk-[a-zA-Z0-9]{10,}", re.IGNORECASE),
    "bearer_token": re.compile(r"Bearer\s+[a-zA-Z0-9\-_\.]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "private_key": re.compile(r"-----BEGIN (RSA |DSA )?PRIVATE KEY-----"),
    "connection_string": re.compile(r"(mongodb|postgres|mysql)://[^\s]+:[^\s]+@"),
}


def validate_dataset_contract(
    samples: list[dict],
    dataset_type: str = "ranking",
    split: str | None = None,
) -> dict[str, Any]:
    """
    Validate dataset contract.

    Args:
        samples: List of dataset samples
        dataset_type: Type of dataset (ranking, route, write, etc.)
        split: Optional split name (train, validation, test)

    Returns:
        Dict with:
        - valid: bool
        - errors: dict mapping error type to count
        - error_details: list of specific violations
    """
    errors: dict[str, int] = defaultdict(int)
    error_details: list[dict] = []

    for i, sample in enumerate(samples):
        sample_id = f"sample_{i}"
        
        # Required fields check
        if "trace_id" not in sample:
            errors["missing_trace_id"] += 1
            error_details.append({
                "sample_id": sample_id,
                "error": "missing_trace_id",
                "message": "Sample missing required field: trace_id",
            })
        
        if "session_id" not in sample:
            errors["missing_session_id"] += 1
            error_details.append({
                "sample_id": sample_id,
                "error": "missing_session_id",
                "message": "Sample missing required field: session_id",
            })
        
        # Temporal ordering check (if timestamps present)
        if "query_timestamp" in sample and "candidate_timestamp" in sample:
            if sample["candidate_timestamp"] > sample["query_timestamp"]:
                errors["temporal_violation"] += 1
                error_details.append({
                    "sample_id": sample_id,
                    "error": "temporal_violation",
                    "message": f"Candidate timestamp {sample['candidate_timestamp']} after query {sample['query_timestamp']}",
                })
        
        # Sensitive data check
        content_fields = ["content", "query", "memory", "text"]
        for field in content_fields:
            if field in sample:
                content = str(sample[field])
                for pattern_name, pattern in SENSITIVE_PATTERNS.items():
                    if pattern.search(content):
                        errors["sensitive_data"] += 1
                        error_details.append({
                            "sample_id": sample_id,
                            "error": "sensitive_data",
                            "message": f"Unredacted {pattern_name} detected in field '{field}'",
                        })
                        break  # One error per sample
        
        # Evaluation boundary check
        if split == "train" and sample.get("dataset") == "evaluation_only":
            errors["evaluation_data_in_training"] += 1
            error_details.append({
                "sample_id": sample_id,
                "error": "evaluation_data_in_training",
                "message": "Evaluation-only data found in training split",
            })

    return {
        "valid": len(errors) == 0,
        "errors": dict(errors),
        "error_details": error_details[:100],  # Limit to first 100 errors
        "total_samples": len(samples),
        "total_violations": sum(errors.values()),
    }


def validate_split_isolation(splits: dict[str, list[dict]]) -> dict[str, Any]:
    """
    Validate that sessions don't leak across splits.

    Args:
        splits: Dict mapping split name to list of samples

    Returns:
        Dict with:
        - valid: bool
        - errors: dict with error counts
        - leaked_sessions: list of session IDs in multiple splits
    """
    session_to_splits: dict[str, set[str]] = defaultdict(set)
    
    for split_name, samples in splits.items():
        for sample in samples:
            if "session_id" in sample:
                session_to_splits[sample["session_id"]].add(split_name)
    
    # Find sessions in multiple splits
    leaked_sessions = [
        session_id for session_id, split_set in session_to_splits.items()
        if len(split_set) > 1
    ]
    
    errors = {}
    if leaked_sessions:
        errors["split_leakage"] = len(leaked_sessions)
    
    return {
        "valid": len(leaked_sessions) == 0,
        "errors": errors,
        "leaked_sessions": leaked_sessions[:50],  # Limit to first 50
        "total_sessions": len(session_to_splits),
    }


def main() -> None:
    """CLI for dataset validation."""
    parser = argparse.ArgumentParser(description="Validate dataset contract")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--dataset-type", default="ranking", help="Dataset type")
    parser.add_argument("--split", help="Split name (train/validation/test)")
    parser.add_argument("--output", type=Path, help="Output report JSON")
    parser.add_argument("--split-isolation", nargs="+", type=Path, help="Multiple JSONL files to check split isolation")
    args = parser.parse_args()

    if args.split_isolation:
        # Validate split isolation across multiple files
        splits = {}
        for path in args.split_isolation:
            split_name = path.stem.split("_")[-1]  # e.g., "train" from "data_train.jsonl"
            with path.open(encoding="utf-8") as f:
                splits[split_name] = [json.loads(line) for line in f]
        
        result = validate_split_isolation(splits)
    else:
        # Validate single file
        with args.input.open(encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
        
        result = validate_dataset_contract(
            samples,
            dataset_type=args.dataset_type,
            split=args.split,
        )

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ Validation report saved: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Exit with error if validation failed
    if not result["valid"]:
        print(f"\n❌ Validation FAILED: {result.get('total_violations', len(result.get('errors', {})))} violations")
        exit(1)
    else:
        print(f"\n✓ Validation PASSED: {result.get('total_samples', 0)} samples")


if __name__ == "__main__":
    main()
