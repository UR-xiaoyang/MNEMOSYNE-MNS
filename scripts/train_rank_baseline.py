from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_DIR = Path("C:/Users/xiaoyang/memory_dataset/features_v1")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_metrics(rows: list[dict], scores: np.ndarray) -> dict:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        # pair_id is <query/sample id>::<candidate id>.
        qid = row["pair_id"].rsplit("::", 1)[0]
        groups[qid].append((float(score), int(row["label"])))

    recalls = {1: [], 5: [], 10: []}
    rr = []
    ndcg5 = []
    ndcg10 = []
    for items in groups.values():
        ranked = sorted(items, key=lambda x: x[0], reverse=True)
        positives = sum(label for _, label in ranked)
        if positives == 0:
            continue
        for k in recalls:
            recalls[k].append(float(any(label for _, label in ranked[:k])))
        first_rank = next((i + 1 for i, (_, label) in enumerate(ranked) if label), None)
        rr.append(1.0 / first_rank if first_rank else 0.0)
        for k, target in ((5, ndcg5), (10, ndcg10)):
            dcg = sum(label / math.log2(i + 2) for i, (_, label) in enumerate(ranked[:k]))
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(positives, k)))
            target.append(dcg / ideal if ideal else 0.0)

    result = {
        "queries": len(groups),
        "queries_with_positive": len(rr),
        "Recall@1": float(np.mean(recalls[1])) if recalls[1] else 0.0,
        "Recall@5": float(np.mean(recalls[5])) if recalls[5] else 0.0,
        "Recall@10": float(np.mean(recalls[10])) if recalls[10] else 0.0,
        "MRR": float(np.mean(rr)) if rr else 0.0,
        "NDCG@5": float(np.mean(ndcg5)) if ndcg5 else 0.0,
        "NDCG@10": float(np.mean(ndcg10)) if ndcg10 else 0.0,
    }
    labels = np.asarray([int(r["label"]) for r in rows])
    if len(np.unique(labels)) > 1:
        result["pair_ROC_AUC"] = float(roc_auc_score(labels, scores))
        result["pair_AP"] = float(average_precision_score(labels, scores))
    return result


def select_public(rows: list[dict], matrix: np.ndarray, extra: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    indices = [i for i, row in enumerate(rows) if row.get("dataset") == "public"]
    return [rows[i] for i in indices], matrix[indices], extra[indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rank_baseline_v1"))
    args = parser.parse_args()
    fd, od = args.feature_dir, args.output_dir
    od.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows_all = read_jsonl(fd / "rerank_train_pairs.jsonl")
    val_rows_all = read_jsonl(fd / "rerank_validation_pairs.jsonl")
    train_x_all = load_npz(fd / "rerank_train_lexical.npz")
    val_x_all = load_npz(fd / "rerank_validation_lexical.npz")
    train_extra_all = np.load(fd / "rerank_train_extra.npy")
    val_extra_all = np.load(fd / "rerank_validation_extra.npy")

    train_rows, train_x, train_extra = select_public(train_rows_all, train_x_all, train_extra_all)
    val_rows, val_x, val_extra = select_public(val_rows_all, val_x_all, val_extra_all)
    y_train = np.asarray([int(r["label"]) for r in train_rows])

    report = {
        "feature_dir": str(fd),
        "data_policy": "public AlpsBench only; private weak adaptation excluded from this first baseline",
        "train_pairs": len(train_rows),
        "validation_pairs": len(val_rows),
        "train_positive_pairs": int(y_train.sum()),
        "validation_positive_pairs": int(sum(int(r["label"]) for r in val_rows)),
        "models": {},
    }

    # Baseline A: direct TF-IDF cosine feature already computed without labels.
    t0 = time.time()
    cosine_scores = val_extra[:, 0]
    report["models"]["tfidf_cosine"] = {
        "type": "nonparametric_score",
        "validation": group_metrics(val_rows, cosine_scores),
        "train_seconds": 0.0,
    }

    # Baseline B: structured feature logistic model. Fit scaler on train only.
    t0 = time.time()
    scaler = StandardScaler()
    xtr_extra = scaler.fit_transform(train_extra)
    xva_extra = scaler.transform(val_extra)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(xtr_extra, y_train)
    structured_scores = model.predict_proba(xva_extra)[:, 1]
    joblib.dump({"model": model, "scaler": scaler}, od / "structured_logistic.joblib")
    report["models"]["structured_logistic"] = {
        "type": "logistic_regression_on_12_structured_features",
        "validation": group_metrics(val_rows, structured_scores),
        "train_seconds": round(time.time() - t0, 3),
    }

    # Baseline C: sparse linear model on query/memory TF-IDF interactions.
    # Keep this separate from structured model so a gain can be attributed.
    t0 = time.time()
    sparse_model = LogisticRegression(
        class_weight="balanced",
        max_iter=150,
        C=0.5,
        solver="liblinear",
        random_state=42,
    )
    sparse_model.fit(train_x, y_train)
    sparse_scores = sparse_model.predict_proba(val_x)[:, 1]
    joblib.dump(sparse_model, od / "sparse_logistic.joblib")
    report["models"]["sparse_logistic"] = {
        "type": "logistic_regression_on_sparse_tfidf_interactions",
        "feature_dim": int(train_x.shape[1]),
        "validation": group_metrics(val_rows, sparse_scores),
        "train_seconds": round(time.time() - t0, 3),
    }

    # Save per-query predictions for later error analysis.
    pred_path = od / "validation_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row, a, b, c in zip(val_rows, cosine_scores, structured_scores, sparse_scores):
            f.write(json.dumps({
                "pair_id": row["pair_id"],
                "query_id": row["pair_id"].rsplit("::", 1)[0],
                "memory_id": row.get("memory_id"),
                "label": row["label"],
                "tfidf_cosine": float(a),
                "structured_logistic": float(b),
                "sparse_logistic": float(c),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")

    report["elapsed_seconds"] = round(time.time() - started, 3)
    report["verification"] = {
        "public_only": True,
        "feature_target_separation": True,
        "grouped_query_metrics": True,
        "private_weak_data_used": False,
        "gpu_required": False,
    }
    (od / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
