"""
Evaluate ranking models with query-grouped metrics and paired bootstrap comparison.

Provides:
- Query-grouped Recall@k, MRR, NDCG@k computation
- Paired bootstrap comparison with confidence intervals
- CLI for evaluation from prediction files
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_ranking_metrics(predictions: list[dict], score_key: str = "score") -> dict:
    """
    Compute query-grouped ranking metrics.

    Args:
        predictions: List of dicts with keys: pair_id, query_id, label, <score_key>
        score_key: Key name for the ranking score (default: "score")

    Returns:
        Dict with queries count and averaged metrics:
        Recall@1, Recall@5, Recall@10, MRR, NDCG@5, NDCG@10
    """
    # Group by query_id
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for pred in predictions:
        query_id = pred["query_id"]
        score = float(pred[score_key])
        label = int(pred["label"])
        groups[query_id].append((score, label))

    # Compute per-query metrics
    metric_values = {
        "Recall@1": [],
        "Recall@5": [],
        "Recall@10": [],
        "MRR": [],
        "NDCG@5": [],
        "NDCG@10": [],
    }

    for candidates in groups.values():
        # Sort by score descending
        ranked = sorted(candidates, key=lambda x: x[0], reverse=True)
        positive_count = sum(label for _, label in ranked)

        # Skip queries with no positives
        if positive_count == 0:
            continue

        # Recall@k
        metric_values["Recall@1"].append(float(any(label for _, label in ranked[:1])))
        metric_values["Recall@5"].append(float(any(label for _, label in ranked[:5])))
        metric_values["Recall@10"].append(float(any(label for _, label in ranked[:10])))

        # MRR
        first_positive = next(i + 1 for i, (_, label) in enumerate(ranked) if label)
        metric_values["MRR"].append(1.0 / first_positive)

        # NDCG@k
        for k in (5, 10):
            dcg = sum(label / math.log2(i + 2) for i, (_, label) in enumerate(ranked[:k]))
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, positive_count)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metric_values[f"NDCG@{k}"].append(ndcg)

    # Average across queries (count queries with at least one positive)
    queries_with_positives = len(metric_values["Recall@1"])
    
    return {
        "queries": queries_with_positives,
        **{key: float(np.mean(value)) if value else 0.0 for key, value in metric_values.items()},
    }


def paired_bootstrap_comparison(
    predictions_a: list[dict],
    predictions_b: list[dict],
    score_key_a: str,
    score_key_b: str,
    n_resamples: int = 3000,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Paired bootstrap comparison between two models.

    Args:
        predictions_a: Predictions from model A
        predictions_b: Predictions from model B
        score_key_a: Score key for model A
        score_key_b: Score key for model B
        n_resamples: Number of bootstrap resamples (default: 3000)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping metric names to {delta, ci95_lower, ci95_upper, p_b_gt_a}
    """
    # Use local RNG to avoid polluting global state
    rng = np.random.default_rng(seed)

    # Validate required fields upfront
    for i, pred in enumerate(predictions_a):
        if "query_id" not in pred:
            raise ValueError(f"predictions_a[{i}] missing required field 'query_id'")
        if "pair_id" not in pred:
            raise ValueError(f"predictions_a[{i}] missing required field 'pair_id'")
        if "label" not in pred:
            raise ValueError(f"predictions_a[{i}] missing required field 'label'")
    
    for i, pred in enumerate(predictions_b):
        if "query_id" not in pred:
            raise ValueError(f"predictions_b[{i}] missing required field 'query_id'")
        if "pair_id" not in pred:
            raise ValueError(f"predictions_b[{i}] missing required field 'pair_id'")
        if "label" not in pred:
            raise ValueError(f"predictions_b[{i}] missing required field 'label'")

    # Validate (query_id, pair_id) uniqueness and no cross-query pair_ids
    def validate_uniqueness(predictions: list[dict], name: str):
        seen_pairs = set()  # (query_id, pair_id) tuples
        pair_to_queries = {}  # pair_id -> set of query_ids
        
        for pred in predictions:
            qid = pred["query_id"]
            pid = pred["pair_id"]
            
            # Check pair_id prefix consistency when using :: format
            if "::" in pid:
                prefix = pid.rsplit("::", 1)[0]
                if prefix != qid:
                    raise ValueError(
                        f"{name} has pair_id '{pid}' with prefix '{prefix}' that does not match "
                        f"query_id '{qid}'. When using '::' format, pair_id prefix must equal query_id."
                    )
            
            # Check (query_id, pair_id) uniqueness
            key = (qid, pid)
            if key in seen_pairs:
                raise ValueError(
                    f"{name} has duplicate pair_id '{pid}' within query '{qid}'. "
                    f"Each (query_id, pair_id) must be unique."
                )
            seen_pairs.add(key)
            
            # Track which queries each pair_id appears in
            if pid not in pair_to_queries:
                pair_to_queries[pid] = set()
            pair_to_queries[pid].add(qid)
        
        # Check no pair_id appears in multiple queries
        for pid, qids in pair_to_queries.items():
            if len(qids) > 1:
                raise ValueError(
                    f"{name} has pair_id '{pid}' appearing in multiple queries: {sorted(qids)}. "
                    f"Each pair_id must belong to exactly one query."
                )
    
    validate_uniqueness(predictions_a, "predictions_a")
    validate_uniqueness(predictions_b, "predictions_b")

    # Query ID sets must match exactly (strict paired comparison)
    query_ids_a = set(pred["query_id"] for pred in predictions_a)
    query_ids_b = set(pred["query_id"] for pred in predictions_b)
    
    if query_ids_a != query_ids_b:
        only_a = query_ids_a - query_ids_b
        only_b = query_ids_b - query_ids_a
        raise ValueError(
            f"Query ID sets must match exactly for paired comparison. "
            f"predictions_a has {len(query_ids_a)} queries, predictions_b has {len(query_ids_b)} queries. "
            f"Only in A: {sorted(only_a) if only_a else 'none'}, "
            f"Only in B: {sorted(only_b) if only_b else 'none'}."
        )
    
    common_query_ids = sorted(query_ids_a)

    # Group predictions by query and validate pairing
    groups_a: dict[str, list[dict]] = defaultdict(list)
    groups_b: dict[str, list[dict]] = defaultdict(list)

    for pred in predictions_a:
        if pred["query_id"] in common_query_ids:
            groups_a[pred["query_id"]].append(pred)

    for pred in predictions_b:
        if pred["query_id"] in common_query_ids:
            groups_b[pred["query_id"]].append(pred)
    
    # Validate that each common query has matching pair_id sets and labels
    for qid in common_query_ids:
        pair_ids_a = {p["pair_id"] for p in groups_a[qid]}
        pair_ids_b = {p["pair_id"] for p in groups_b[qid]}
        
        if pair_ids_a != pair_ids_b:
            raise ValueError(
                f"Query '{qid}' has mismatched pair_id sets: "
                f"A has {len(pair_ids_a)} pairs, B has {len(pair_ids_b)} pairs. "
                f"Paired bootstrap requires identical candidate sets."
            )
        
        # Check that labels match for each pair_id
        for pair_id in pair_ids_a:
            pred_a = next(p for p in groups_a[qid] if p["pair_id"] == pair_id)
            pred_b = next(p for p in groups_b[qid] if p["pair_id"] == pair_id)
            
            if pred_a["label"] != pred_b["label"]:
                raise ValueError(
                    f"Pair '{pair_id}' has mismatched labels: "
                    f"A has label={pred_a['label']}, B has label={pred_b['label']}. "
                    f"Paired bootstrap requires identical labels."
                )
        
        # Verify score keys exist
        for pred in groups_a[qid]:
            if score_key_a not in pred:
                raise ValueError(f"Score key '{score_key_a}' not found in prediction {pred['pair_id']}")
        
        for pred in groups_b[qid]:
            if score_key_b not in pred:
                raise ValueError(f"Score key '{score_key_b}' not found in prediction {pred['pair_id']}")
    
    # Convert to score-label tuples after validation
    groups_a_tuples: dict[str, list[tuple[float, int]]] = {
        qid: [(float(p[score_key_a]), int(p["label"])) for p in preds]
        for qid, preds in groups_a.items()
    }
    groups_b_tuples: dict[str, list[tuple[float, int]]] = {
        qid: [(float(p[score_key_b]), int(p["label"])) for p in preds]
        for qid, preds in groups_b.items()
    }

    # Compute per-query metrics for both models
    def query_metrics(groups_tuples, qid):
        """Compute metrics for a single query."""
        ranked = sorted(groups_tuples[qid], key=lambda x: x[0], reverse=True)
        pos_count = sum(label for _, label in ranked)

        if pos_count == 0:
            return np.array([0.0] * 6)

        r1 = float(any(label for _, label in ranked[:1]))
        r5 = float(any(label for _, label in ranked[:5]))
        r10 = float(any(label for _, label in ranked[:10]))
        first = next(i + 1 for i, (_, label) in enumerate(ranked) if label)
        mrr = 1.0 / first

        dcg5 = sum(label / math.log2(i + 2) for i, (_, label) in enumerate(ranked[:5]))
        idcg5 = sum(1.0 / math.log2(i + 2) for i in range(min(5, pos_count)))
        ndcg5 = dcg5 / idcg5 if idcg5 > 0 else 0.0

        dcg10 = sum(label / math.log2(i + 2) for i, (_, label) in enumerate(ranked[:10]))
        idcg10 = sum(1.0 / math.log2(i + 2) for i in range(min(10, pos_count)))
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0

        return np.array([r1, r5, r10, mrr, ndcg5, ndcg10])

    metrics_a = np.array([query_metrics(groups_a_tuples, qid) for qid in common_query_ids])
    metrics_b = np.array([query_metrics(groups_b_tuples, qid) for qid in common_query_ids])
    deltas = metrics_b - metrics_a

    # Bootstrap resampling with local RNG
    n = len(common_query_ids)
    metric_names = ["Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@5", "NDCG@10"]
    results = {}

    for i, name in enumerate(metric_names):
        delta_mean = deltas[:, i].mean()
        bootstrap_deltas = []

        for _ in range(n_resamples):
            sample_idx = rng.choice(n, size=n, replace=True)
            bootstrap_deltas.append(deltas[sample_idx, i].mean())

        bootstrap_deltas.sort()
        ci_low = bootstrap_deltas[int(0.025 * n_resamples)]
        ci_high = bootstrap_deltas[int(0.975 * n_resamples)]
        p_greater = sum(d > 0 for d in bootstrap_deltas) / n_resamples

        results[name] = {
            "delta": float(delta_mean),
            "ci95_lower": float(ci_low),
            "ci95_upper": float(ci_high),
            "p_b_gt_a": float(p_greater),
        }

    return results


def main() -> None:
    """CLI for ranking evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ranking model predictions")
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL predictions file")
    parser.add_argument("--score-key", default="score", help="Score key name (default: score)")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--bootstrap-resamples", type=int, default=3000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare-with", type=Path, help="Compare with another predictions file")
    parser.add_argument("--compare-score-key", help="Score key for comparison model")
    args = parser.parse_args()

    # Load predictions
    with args.predictions.open(encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    # Compute metrics
    metrics = compute_ranking_metrics(predictions, args.score_key)

    result = {
        "predictions_file": str(args.predictions),
        "score_key": args.score_key,
        "metrics": metrics,
    }

    # Optional: bootstrap comparison
    if args.compare_with:
        with args.compare_with.open(encoding="utf-8") as f:
            compare_predictions = [json.loads(line) for line in f]

        compare_score_key = args.compare_score_key or args.score_key
        bootstrap_result = paired_bootstrap_comparison(
            compare_predictions,
            predictions,
            compare_score_key,
            args.score_key,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

        result["bootstrap_comparison"] = {
            "baseline_file": str(args.compare_with),
            "baseline_score_key": compare_score_key,
            "n_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "metrics": bootstrap_result,
        }

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ Evaluation saved: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
