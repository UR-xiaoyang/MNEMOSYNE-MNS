#!/usr/bin/env python3
"""
Iterative rank training loop with weak supervision.

P3: Build iteration manifests from shadow samples with temporal splits,
unknown label handling, and automatic data gates.

No manual gold labels required - uses citation/correction evidence only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_iteration_manifest(
    input_dir: str | Path | None,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    """
    Build iteration manifest from shadow training samples.
    
    Splits samples temporally (not randomly) to prevent leakage.
    Separates citation/correction labels, assigns zero weight to unknown.
    
    Args:
        input_dir: Explicit path to shadow samples directory (required).
        output_dir: Where to write split files and manifest.
        train_ratio: Fraction for training (chronologically earliest).
        val_ratio: Fraction for validation (middle chronological window).
        
    Returns:
        Manifest dict with counts, file paths, and label statistics.
        
    Raises:
        ValueError: If input_dir is None (explicit authorization required).
    """
    if input_dir is None:
        raise ValueError(
            "Explicit input directory required. "
            "Pass --input-dir <path> via CLI or input_dir= in code. "
            "No default discovery is performed."
        )
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all samples
    samples = []
    samples_file = input_dir / "samples.jsonl"
    
    if not samples_file.exists():
        raise FileNotFoundError(f"No samples.jsonl found in {input_dir}")
    
    with open(samples_file) as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    
    if not samples:
        return {
            "status": "insufficient_evidence",
            "reason": "No samples found",
            "train_samples": 0,
            "val_samples": 0,
            "holdout_samples": 0,
        }
    
    # Sort by query timestamp (temporal ordering)
    samples.sort(key=lambda x: x["query_timestamp"])
    
    # Calculate split indices
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    holdout_samples = samples[val_end:]
    
    # Process samples: set sample_weight=0 for unknown labels
    def process_sample(sample: dict) -> dict:
        """Add sample_weight based on label."""
        sample = sample.copy()
        if sample.get("label") == "unknown":
            sample["sample_weight"] = 0.0
        elif "sample_weight" not in sample:
            sample["sample_weight"] = 1.0
        return sample
    
    train_samples = [process_sample(s) for s in train_samples]
    val_samples = [process_sample(s) for s in val_samples]
    holdout_samples = [process_sample(s) for s in holdout_samples]
    
    # Write split files
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    holdout_file = output_dir / "holdout.jsonl"
    
    with open(train_file, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(val_file, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(holdout_file, "w") as f:
        for sample in holdout_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Collect statistics (no text content in manifest)
    all_labels = [s["label"] for s in samples]
    label_counts = Counter(all_labels)
    
    citation_count = label_counts.get("positive", 0)
    correction_count = label_counts.get("corrected", 0)
    unknown_count = label_counts.get("unknown", 0)
    
    unknown_ratio = unknown_count / len(samples) if samples else 0.0
    
    # Build manifest (metadata only, no paths, no text)
    manifest = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "holdout_samples": len(holdout_samples),
        "train_file": str(train_file.name),  # Filename only, not full path
        "val_file": str(val_file.name),
        "holdout_file": str(holdout_file.name),
        "citation_count": citation_count,
        "correction_count": correction_count,
        "unknown_count": unknown_count,
        "unknown_ratio": unknown_ratio,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "temporal_split": True,
    }
    
    # Write manifest
    manifest_file = output_dir / "iteration_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def check_data_gate(
    manifest: dict[str, Any],
    min_samples: int = 100,
    max_unknown_ratio: float = 0.9,
) -> dict[str, Any]:
    """
    Check if data meets minimum quality thresholds.
    
    Args:
        manifest: Iteration manifest from build_iteration_manifest.
        min_samples: Minimum total samples required.
        max_unknown_ratio: Maximum acceptable ratio of unknown labels.
        
    Returns:
        Gate result with status: "pass", "pass_with_warning", or "insufficient_evidence".
    """
    train_samples = manifest.get("train_samples", 0)
    val_samples = manifest.get("val_samples", 0)
    total_samples = train_samples + val_samples
    unknown_ratio = manifest.get("unknown_ratio", 1.0)
    
    # Insufficient samples
    if total_samples < min_samples:
        return {
            "status": "insufficient_evidence",
            "reason": f"Only {total_samples} samples found, minimum {min_samples} required",
            "total_samples": total_samples,
        }
    
    # High unknown ratio warning
    warnings = []
    if unknown_ratio > max_unknown_ratio:
        warnings.append(f"unknown_ratio={unknown_ratio:.2f} exceeds threshold {max_unknown_ratio}")
    
    if warnings:
        return {
            "status": "pass_with_warning",
            "warnings": warnings,
            "total_samples": total_samples,
        }
    
    return {
        "status": "pass",
        "total_samples": total_samples,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build iteration manifest from shadow training samples"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing samples.jsonl (explicit authorization required)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write split files and manifest",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples for data gate (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Build manifest
    print(f"Building iteration manifest from {args.input_dir}")
    manifest = build_iteration_manifest(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    
    print(f"\nManifest created:")
    print(f"  Total samples: {manifest['total_samples']}")
    print(f"  Train: {manifest['train_samples']}")
    print(f"  Val: {manifest['val_samples']}")
    print(f"  Holdout: {manifest['holdout_samples']}")
    print(f"  Citation: {manifest['citation_count']}")
    print(f"  Correction: {manifest['correction_count']}")
    print(f"  Unknown: {manifest['unknown_count']} ({manifest['unknown_ratio']:.1%})")
    
    # Check data gate
    gate_result = check_data_gate(manifest, min_samples=args.min_samples)
    print(f"\nData gate: {gate_result['status']}")
    if gate_result["status"] == "insufficient_evidence":
        print(f"  {gate_result['reason']}")
    elif "warnings" in gate_result:
        for warning in gate_result["warnings"]:
            print(f"  Warning: {warning}")


if __name__ == "__main__":
    main()
