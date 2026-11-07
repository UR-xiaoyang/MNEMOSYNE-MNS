#!/usr/bin/env python3
"""
Active disagreement sampler for weak-supervised iterative training.

P3: Sample high-value examples based on baseline/candidate disagreement,
low confidence, and hard constraint boundaries.

No human labeling required - outputs sample IDs and reasons only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def categorize_disagreement(sample: dict[str, Any]) -> str:
    """
    Categorize the type of disagreement.
    
    Args:
        sample: Dict with baseline_score, candidate_score, baseline_rank, candidate_rank.
        
    Returns:
        Reason string describing the disagreement type.
    """
    baseline_score = sample.get("baseline_score", 0.0)
    candidate_score = sample.get("candidate_score", 0.0)
    baseline_rank = sample.get("baseline_rank", 999)
    candidate_rank = sample.get("candidate_rank", 999)
    
    score_diff = abs(baseline_score - candidate_score)
    rank_diff = abs(baseline_rank - candidate_rank)
    
    # Low confidence: both models near decision boundary
    if max(baseline_score, candidate_score) < 0.55:
        return "low_confidence"
    
    # Large rank flip
    if rank_diff >= 5:
        return "large_rank_flip"
    
    # Score disagreement
    if score_diff > 0.3:
        return "score_disagreement"
    
    return "minor_disagreement"


def sample_disagreements(
    baseline_predictions: str | Path,
    candidate_predictions: str | Path,
    n_samples: int = 100,
) -> list[dict[str, Any]]:
    """
    Sample high-value disagreement cases between baseline and candidate.
    
    Args:
        baseline_predictions: Path to baseline predictions JSONL.
        candidate_predictions: Path to candidate predictions JSONL.
        n_samples: Maximum number of samples to return.
        
    Returns:
        List of disagreement samples with IDs and reasons (no text content).
    """
    # Load predictions
    baseline_preds = []
    with open(baseline_predictions) as f:
        for line in f:
            pred = json.loads(line)
            baseline_preds.append(pred)
    
    candidate_preds = []
    with open(candidate_predictions) as f:
        for line in f:
            pred = json.loads(line)
            candidate_preds.append(pred)
    
    # Index by (query_id, pair_id)
    baseline_by_key = {}
    for pred in baseline_preds:
        key = (pred["query_id"], pred["pair_id"])
        baseline_by_key[key] = pred
    
    candidate_by_key = {}
    for pred in candidate_preds:
        key = (pred["query_id"], pred["pair_id"])
        candidate_by_key[key] = pred
    
    # Find disagreements
    disagreements = []
    
    for key in baseline_by_key:
        if key not in candidate_by_key:
            continue
        
        baseline_pred = baseline_by_key[key]
        candidate_pred = candidate_by_key[key]
        
        query_id = baseline_pred["query_id"]
        pair_id = baseline_pred["pair_id"]
        
        baseline_score = baseline_pred.get("score", 0.0)
        candidate_score = candidate_pred.get("score", 0.0)
        baseline_rank = baseline_pred.get("rank", 999)
        candidate_rank = candidate_pred.get("rank", 999)
        
        # Compute disagreement
        score_diff = abs(baseline_score - candidate_score)
        rank_diff = abs(baseline_rank - candidate_rank)
        
        if score_diff >= 0.1 or rank_diff > 0:
            # Build sample (IDs only, no text)
            sample = {
                "query_id": query_id,
                "pair_id": pair_id,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "baseline_rank": baseline_rank,
                "candidate_rank": candidate_rank,
                "score_diff": score_diff,
                "rank_diff": rank_diff,
            }
            
            reason = categorize_disagreement(sample)
            sample["reason"] = reason
            
            # Remove text fields if they exist (privacy)
            sample.pop("query_text", None)
            sample.pop("memory_text", None)
            sample.pop("query_text_redacted", None)
            sample.pop("memory_text_redacted", None)
            
            disagreements.append(sample)
    
    # Sort by disagreement magnitude (rank_diff primary, score_diff secondary)
    disagreements.sort(key=lambda x: (x["rank_diff"], x["score_diff"]), reverse=True)
    
    # Take top n_samples
    return disagreements[:n_samples]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sample high-value disagreement cases for active learning"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline predictions JSONL file",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate predictions JSONL file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for sampled disagreements",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Maximum samples to select (default: 100)",
    )
    
    args = parser.parse_args()
    
    print(f"Sampling disagreements between baseline and candidate...")
    print(f"  Baseline: {Path(args.baseline).name}")
    print(f"  Candidate: {Path(args.candidate).name}")
    print(f"  Max samples: {args.n_samples}")
    
    samples = sample_disagreements(
        baseline_predictions=args.baseline,
        candidate_predictions=args.candidate,
        n_samples=args.n_samples,
    )
    
    print(f"\nFound {len(samples)} disagreement samples")
    
    # Categorize reasons
    reason_counts = defaultdict(int)
    for sample in samples:
        reason_counts[sample["reason"]] += 1
    
    print("\nDisagreement types:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"\nSamples written to: {output_path.name}")


if __name__ == "__main__":
    main()
