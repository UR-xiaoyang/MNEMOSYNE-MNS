#!/usr/bin/env python3
"""
Offline replay and comparison of baseline vs candidate rankers.

P3: Compare ranking models on identical query pools with paired bootstrap.
Uses existing evaluate_ranking.py infrastructure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_ranking import compute_ranking_metrics, paired_bootstrap_comparison


def validate_predictions(predictions: list[dict]) -> None:
    """
    Validate prediction format and consistency.
    
    Args:
        predictions: List of prediction dicts.
        
    Raises:
        ValueError: If predictions are invalid.
    """
    seen_pairs = set()
    
    for pred in predictions:
        query_id = pred.get("query_id")
        pair_id = pred.get("pair_id")
        
        if not query_id or not pair_id:
            raise ValueError("Missing query_id or pair_id in prediction")
        
        # Check pair_id uniqueness
        key = (query_id, pair_id)
        if key in seen_pairs:
            raise ValueError(f"Duplicate pair_id {pair_id} for query {query_id}")
        seen_pairs.add(key)
        
        # Check pair_id prefix matches query_id
        if "::" in pair_id:
            prefix = pair_id.split("::")[0]
            if prefix != query_id:
                raise ValueError(
                    f"pair_id prefix '{prefix}' does not match query_id '{query_id}'"
                )


def load_predictions(filepath: str | Path) -> list[dict]:
    """Load predictions from JSONL file."""
    filepath = Path(filepath)
    predictions = []
    
    with open(filepath) as f:
        for line in f:
            pred = json.loads(line)
            predictions.append(pred)
    
    return predictions


def replay_and_compare(
    baseline_predictions: str | Path,
    candidate_predictions: str | Path,
    n_bootstrap: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Replay baseline and candidate on same query pool and compare.
    
    Args:
        baseline_predictions: Path to baseline predictions JSONL.
        candidate_predictions: Path to candidate predictions JSONL.
        n_bootstrap: Number of bootstrap resamples (default: 3000).
        seed: Random seed for bootstrap (default: 42).
        
    Returns:
        Comparison result with metrics and confidence intervals.
        
    Raises:
        ValueError: If query pools don't match or predictions are invalid.
    """
    # Load predictions
    baseline_preds = load_predictions(baseline_predictions)
    candidate_preds = load_predictions(candidate_predictions)
    
    # Validate each
    validate_predictions(baseline_preds)
    validate_predictions(candidate_preds)
    
    # Extract query sets
    baseline_queries = {pred["query_id"] for pred in baseline_preds}
    candidate_queries = {pred["query_id"] for pred in candidate_preds}
    
    # Verify identical query pools
    if baseline_queries != candidate_queries:
        missing_in_candidate = baseline_queries - candidate_queries
        missing_in_baseline = candidate_queries - baseline_queries
        
        error_parts = []
        if missing_in_candidate:
            error_parts.append(f"Missing in candidate: {missing_in_candidate}")
        if missing_in_baseline:
            error_parts.append(f"Missing in baseline: {missing_in_baseline}")
        
        raise ValueError(
            f"Query pools don't match. {' '.join(error_parts)}"
        )
    
    # Verify identical candidate sets per query
    baseline_by_query = {}
    candidate_by_query = {}
    
    for pred in baseline_preds:
        query_id = pred["query_id"]
        if query_id not in baseline_by_query:
            baseline_by_query[query_id] = set()
        baseline_by_query[query_id].add(pred["pair_id"])
    
    for pred in candidate_preds:
        query_id = pred["query_id"]
        if query_id not in candidate_by_query:
            candidate_by_query[query_id] = set()
        candidate_by_query[query_id].add(pred["pair_id"])
    
    for query_id in baseline_queries:
        baseline_pairs = baseline_by_query[query_id]
        candidate_pairs = candidate_by_query[query_id]
        
        if baseline_pairs != candidate_pairs:
            raise ValueError(
                f"Query {query_id} has different candidate sets: "
                f"baseline={len(baseline_pairs)}, candidate={len(candidate_pairs)}"
            )
    
    # Compute metrics for each model
    baseline_metrics = compute_ranking_metrics(baseline_preds, score_key="score")
    candidate_metrics = compute_ranking_metrics(candidate_preds, score_key="score")
    
    # Paired bootstrap comparison
    comparison = paired_bootstrap_comparison(
        predictions_a=baseline_preds,
        predictions_b=candidate_preds,
        score_key_a="score",
        score_key_b="score",
        n_resamples=n_bootstrap,
        seed=seed,
    )
    
    # Build result (no file paths)
    result = {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "comparison": comparison,
        "n_queries": len(baseline_queries),
        "n_bootstrap": n_bootstrap,
    }
    
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Offline replay and comparison of baseline vs candidate rankers"
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
        help="Output JSON file for comparison results",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=3000,
        help="Bootstrap resamples (default: 3000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    print(f"Replaying baseline vs candidate...")
    print(f"  Baseline: {Path(args.baseline).name}")
    print(f"  Candidate: {Path(args.candidate).name}")
    
    result = replay_and_compare(
        baseline_predictions=args.baseline,
        candidate_predictions=args.candidate,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    
    print(f"\nBaseline metrics:")
    for metric, value in result["baseline_metrics"].items():
        if metric != "queries":
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nCandidate metrics:")
    for metric, value in result["candidate_metrics"].items():
        if metric != "queries":
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nPaired comparison (candidate - baseline):")
    for metric, stats in result["comparison"].items():
        delta = stats["mean_difference"]
        ci_low = stats["ci_lower"]
        ci_high = stats["ci_upper"]
        print(f"  {metric}: {delta:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults written to: {output_path.name}")


if __name__ == "__main__":
    main()
