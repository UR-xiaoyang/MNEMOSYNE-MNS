"""
Evaluate and compare all trained RANK models.

Loads each model's validation predictions and computes:
- Per-model metrics
- Pairwise comparisons with bootstrap
- Feature importance analysis
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_predictions(path: Path) -> dict[str, dict]:
    """Load predictions from JSONL file."""
    predictions = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            pred = json.loads(line)
            predictions[pred["pair_id"]] = pred
    return predictions


def grouped_predictions(predictions: dict[str, dict]) -> dict[str, list[tuple[float, int, str]]]:
    """Group predictions by query_id."""
    groups: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for pair_id, pred in predictions.items():
        query_id = pair_id.rsplit("::", 1)[0]
        # Find score key (model-specific)
        score_keys = [k for k in pred.keys() if k not in ["pair_id", "query_id", "memory_id", "label"]]
        if not score_keys:
            continue
        # Use the last score key (usually the final model score)
        score_key = score_keys[-1]
        groups[query_id].append((pred[score_key], pred["label"], pair_id))
    return groups


def ranking_metrics(groups: dict[str, list[tuple[float, int, str]]]) -> dict:
    """Compute ranking metrics from grouped predictions."""
    values = {key: [] for key in ("Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@5", "NDCG@10")}
    
    for candidates in groups.values():
        ranked = sorted(candidates, key=lambda x: x[0], reverse=True)
        positive_count = sum(label for _, label, _ in ranked)
        if positive_count == 0:
            continue
        
        values["Recall@1"].append(float(any(label for _, label, _ in ranked[:1])))
        values["Recall@5"].append(float(any(label for _, label, _ in ranked[:5])))
        values["Recall@10"].append(float(any(label for _, label, _ in ranked[:10])))
        
        first = next(i + 1 for i, (_, label, _) in enumerate(ranked) if label)
        values["MRR"].append(1.0 / first)
        
        for k in (5, 10):
            dcg = sum(label / math.log2(i + 2) for i, (_, label, _) in enumerate(ranked[:k]))
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, positive_count)))
            values[f"NDCG@{k}"].append(dcg / ideal if ideal else 0.0)
    
    return {
        "queries": len(groups),
        **{key: float(np.mean(value)) if value else 0.0 for key, value in values.items()}
    }


def bootstrap_comparison(
    groups_a: dict[str, list[tuple[float, int, str]]],
    groups_b: dict[str, list[tuple[float, int, str]]],
    n_samples: int = 3000,
    seed: int = 42,
) -> dict[str, dict]:
    """Paired bootstrap comparison between two models."""
    np.random.seed(seed)
    
    # Compute per-query metrics for both models
    query_ids = sorted(set(groups_a.keys()) & set(groups_b.keys()))
    
    def query_metrics(groups, qid):
        ranked = sorted(groups[qid], key=lambda x: x[0], reverse=True)
        pos_count = sum(label for _, label, _ in ranked)
        if pos_count == 0:
            return [0.0] * 6
        
        r1 = float(any(label for _, label, _ in ranked[:1]))
        r5 = float(any(label for _, label, _ in ranked[:5]))
        r10 = float(any(label for _, label, _ in ranked[:10]))
        first = next(i + 1 for i, (_, label, _) in enumerate(ranked) if label)
        mrr = 1.0 / first
        
        dcg5 = sum(label / math.log2(i + 2) for i, (_, label, _) in enumerate(ranked[:5]))
        ideal5 = sum(1.0 / math.log2(i + 2) for i in range(min(5, pos_count)))
        ndcg5 = dcg5 / ideal5 if ideal5 else 0.0
        
        dcg10 = sum(label / math.log2(i + 2) for i, (_, label, _) in enumerate(ranked[:10]))
        ideal10 = sum(1.0 / math.log2(i + 2) for i in range(min(10, pos_count)))
        ndcg10 = dcg10 / ideal10 if ideal10 else 0.0
        
        return [r1, r5, r10, mrr, ndcg5, ndcg10]
    
    metrics_a = np.array([query_metrics(groups_a, qid) for qid in query_ids])
    metrics_b = np.array([query_metrics(groups_b, qid) for qid in query_ids])
    deltas = metrics_b - metrics_a
    
    # Bootstrap
    n = len(query_ids)
    metric_names = ["Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@5", "NDCG@10"]
    results = {}
    
    for i, name in enumerate(metric_names):
        delta_mean = deltas[:, i].mean()
        bootstrap_deltas = []
        
        for _ in range(n_samples):
            sample_idx = np.random.choice(n, size=n, replace=True)
            bootstrap_deltas.append(deltas[sample_idx, i].mean())
        
        bootstrap_deltas.sort()
        ci_low = bootstrap_deltas[int(0.025 * n_samples)]
        ci_high = bootstrap_deltas[int(0.975 * n_samples)]
        p_greater = sum(d > 0 for d in bootstrap_deltas) / n_samples
        
        results[name] = {
            "delta": float(delta_mean),
            "ci95": [float(ci_low), float(ci_high)],
            "p_b_gt_a": float(p_greater),
        }
    
    return results


def main() -> None:
    models = {
        "TFIDF": "runs/rank_baseline_v1/validation_predictions.jsonl",
        "Pairwise": "runs/rank_pairwise_v1/validation_predictions.jsonl",
        "Tree": "runs/rank_pairwise_tree_v1/validation_predictions.jsonl",
        "E5": "runs/rank_embedding_e5_v1/validation_predictions.jsonl",
        "Hybrid": "runs/rank_hybrid_v1/validation_predictions.jsonl",
    }
    
    print("=== MNEMOSYNE-MNS RANK Model Evaluation ===\n")
    
    # Load all predictions
    all_preds = {}
    all_groups = {}
    for name, path in models.items():
        if not Path(path).exists():
            print(f"⚠️  {name}: Not found ({path})")
            continue
        preds = load_predictions(Path(path))
        groups = grouped_predictions(preds)
        all_preds[name] = preds
        all_groups[name] = groups
        print(f"✓ {name}: {len(preds)} pairs, {len(groups)} queries")
    
    print("\n=== Validation Metrics ===\n")
    print(f"{'Model':<12} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'NDCG@5':>8} {'NDCG@10':>8}")
    print("-" * 72)
    
    metrics_all = {}
    for name in all_groups:
        metrics = ranking_metrics(all_groups[name])
        metrics_all[name] = metrics
        print(f"{name:<12} {metrics['Recall@1']:>8.4f} {metrics['Recall@5']:>8.4f} {metrics['Recall@10']:>8.4f} "
              f"{metrics['MRR']:>8.4f} {metrics['NDCG@5']:>8.4f} {metrics['NDCG@10']:>8.4f}")
    
    # Pairwise comparisons
    print("\n=== Pairwise Bootstrap Comparison (3000 samples) ===\n")
    
    if "Hybrid" in all_groups and "E5" in all_groups:
        print("Hybrid vs E5:")
        comp = bootstrap_comparison(all_groups["E5"], all_groups["Hybrid"])
        for metric, stats in comp.items():
            sig = "***" if stats["p_b_gt_a"] >= 0.99 else "**" if stats["p_b_gt_a"] >= 0.95 else "*" if stats["p_b_gt_a"] >= 0.90 else ""
            print(f"  {metric:<12} Δ={stats['delta']:+.4f}  95%CI=[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]  "
                  f"P(H>E)={stats['p_b_gt_a']:.3f} {sig}")
    
    if "E5" in all_groups and "Tree" in all_groups:
        print("\nE5 vs Tree:")
        comp = bootstrap_comparison(all_groups["Tree"], all_groups["E5"])
        for metric, stats in comp.items():
            sig = "***" if stats["p_b_gt_a"] >= 0.99 else "**" if stats["p_b_gt_a"] >= 0.95 else "*" if stats["p_b_gt_a"] >= 0.90 else ""
            print(f"  {metric:<12} Δ={stats['delta']:+.4f}  95%CI=[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]  "
                  f"P(E>T)={stats['p_b_gt_a']:.3f} {sig}")
    
    # Save report
    report = {
        "evaluation_date": "2026-10-29",
        "models_evaluated": list(all_groups.keys()),
        "validation_queries": metrics_all[next(iter(all_groups))]["queries"],
        "metrics": metrics_all,
        "comparisons": {
            "hybrid_vs_e5": bootstrap_comparison(all_groups["E5"], all_groups["Hybrid"]) if "Hybrid" in all_groups and "E5" in all_groups else None,
            "e5_vs_tree": bootstrap_comparison(all_groups["Tree"], all_groups["E5"]) if "E5" in all_groups and "Tree" in all_groups else None,
        },
    }
    
    output_path = Path("runs/model_evaluation_report.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Report saved: {output_path}")


if __name__ == "__main__":
    main()
