#!/usr/bin/env python3
"""
Shadow trace evaluation script.

Generates quality reports from exported shadow training data.
Returns insufficient_evidence when sample size is inadequate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def evaluate_shadow_traces(
    data_dir: str | Path,
    min_samples: int = 10,
) -> dict[str, Any]:
    """
    Evaluate shadow trace training data quality.
    
    Args:
        data_dir: Directory containing shadow_training_samples.jsonl
        min_samples: Minimum samples required for meaningful evaluation
    
    Returns:
        Evaluation report with statistics or insufficient_evidence status.
    """
    data_dir = Path(data_dir)
    samples_path = data_dir / "shadow_training_samples.jsonl"
    
    if not samples_path.exists():
        return {
            "status": "insufficient_evidence",
            "reason": "No samples file found",
        }
    
    # Load samples
    samples = []
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    # Check for insufficient evidence
    if len(samples) < min_samples:
        return {
            "status": "insufficient_evidence",
            "reason": f"Only {len(samples)} samples found, minimum {min_samples} required",
            "total_samples": len(samples),
        }
    
    # Compute statistics
    label_counts = Counter(s["label"] for s in samples)
    label_source_counts = Counter(s["label_source"] for s in samples)
    
    report = {
        "status": "success",
        "total_samples": len(samples),
        "label_counts": dict(label_counts),
        "label_source_counts": dict(label_source_counts),
        "privacy_note": "Report contains no source paths or sensitive data",
    }
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate shadow trace training data")
    parser.add_argument("--data-dir", required=True, help="Directory with training samples")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum samples for evaluation")
    parser.add_argument("--output", help="Output JSON path (default: stdout)")
    
    args = parser.parse_args()
    
    result = evaluate_shadow_traces(
        data_dir=args.data_dir,
        min_samples=args.min_samples,
    )
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("✓ Report written successfully")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
