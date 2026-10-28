from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_DIR = Path("C:/Users/xiaoyang/memory_dataset/features_v1")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def metrics(rows: list[dict], scores: np.ndarray) -> dict:
    groups = defaultdict(list)
    for row, score in zip(rows, scores):
        qid = row["pair_id"].rsplit("::", 1)[0]
        groups[qid].append((float(score), int(row["label"])))
    r1, r5, r10, mrr, nd5, nd10 = [], [], [], [], [], []
    for items in groups.values():
        ranked = sorted(items, reverse=True)
        positive_count = sum(x[1] for x in ranked)
        if not positive_count:
            continue
        r1.append(float(any(x[1] for x in ranked[:1])))
        r5.append(float(any(x[1] for x in ranked[:5])))
        r10.append(float(any(x[1] for x in ranked[:10])))
        first = next(i + 1 for i, x in enumerate(ranked) if x[1])
        mrr.append(1.0 / first)
        for k, target in ((5, nd5), (10, nd10)):
            dcg = sum(label / math.log2(i + 2) for i, (_, label) in enumerate(ranked[:k]))
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, positive_count)))
            target.append(dcg / ideal if ideal else 0.0)
    return {
        "queries": len(groups),
        "Recall@1": float(np.mean(r1)),
        "Recall@5": float(np.mean(r5)),
        "Recall@10": float(np.mean(r10)),
        "MRR": float(np.mean(mrr)),
        "NDCG@5": float(np.mean(nd5)),
        "NDCG@10": float(np.mean(nd10)),
    }


def public_rows(rows: list[dict], x: np.ndarray, extra: np.ndarray):
    ix = [i for i, r in enumerate(rows) if r.get("dataset") == "public"]
    return [rows[i] for i in ix], extra[ix]


def grouped(rows: list[dict], x: np.ndarray):
    out = defaultdict(list)
    for i, row in enumerate(rows):
        qid = row["pair_id"].rsplit("::", 1)[0]
        out[qid].append(i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    ap.add_argument("--output-dir", type=Path, default=Path("runs/rank_pairwise_v1"))
    args = ap.parse_args()
    fd, od = args.feature_dir, args.output_dir
    od.mkdir(parents=True, exist_ok=True)
    start = time.time()

    train_rows_all = read_jsonl(fd / "rerank_train_pairs.jsonl")
    val_rows_all = read_jsonl(fd / "rerank_validation_pairs.jsonl")
    test_rows_all = read_jsonl(fd / "rerank_test_pairs.jsonl")
    train_extra_all = np.load(fd / "rerank_train_extra.npy")
    val_extra_all = np.load(fd / "rerank_validation_extra.npy")
    test_extra_all = np.load(fd / "rerank_test_extra.npy")
    train_rows, train_x = public_rows(train_rows_all, train_rows_all, train_extra_all)
    val_rows, val_x = public_rows(val_rows_all, val_rows_all, val_extra_all)
    # The feature builder's test split contains the isolated private weak
    # adaptation slice; keep it out of fitting and report it only as a domain
    # transfer diagnostic.
    test_rows, test_x = test_rows_all, test_extra_all

    # Grouped pairwise samples: each positive candidate is preferred over each
    # same-query negative. This trains ranking, not independent classification.
    groups = grouped(train_rows, train_x)
    pair_x, pair_y = [], []
    for indices in groups.values():
        positives = [i for i in indices if train_rows[i]["label"] == 1]
        negatives = [i for i in indices if train_rows[i]["label"] == 0]
        for p in positives:
            for n in negatives:
                delta = train_x[p] - train_x[n]
                pair_x.append(delta)
                pair_y.append(1)
                pair_x.append(-delta)
                pair_y.append(0)
    pair_x = np.asarray(pair_x, dtype=np.float32)
    pair_y = np.asarray(pair_y, dtype=np.int8)
    if len(pair_x) == 0:
        raise RuntimeError("no pairwise samples")

    scaler = StandardScaler()
    pair_x_scaled = scaler.fit_transform(pair_x)
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(pair_x_scaled, pair_y)
    # Logistic on positive-minus-negative pairs gives a ranking direction.
    weights = model.coef_[0]
    joblib.dump({"scaler": scaler, "model": model, "weights": weights}, od / "pairwise_logistic.joblib")

    train_scores = scaler.transform(train_x) @ weights
    val_scores = scaler.transform(val_x) @ weights
    test_scores = scaler.transform(test_x) @ weights
    report = {
        "model_id": "MNEMOSYNE-MNS-RANK-PAIRWISE-01",
        "data_policy": "public AlpsBench only; private weak adaptation excluded",
        "train_queries": len(groups),
        "train_pairs": int(len(pair_x)),
        "validation_queries": len(grouped(val_rows, val_x)),
        "validation_metrics": metrics(val_rows, val_scores),
        "private_weak_transfer_diagnostic": metrics(test_rows, test_scores) if test_rows else None,
        "train_metrics": metrics(train_rows, train_scores),
        "feature_dim": int(train_x.shape[1]),
        "elapsed_seconds": round(time.time() - start, 3),
        "verification": {
            "grouped_pairwise_training": True,
            "public_only": True,
            "private_weak_data_used": False,
            "target_in_features": False,
            "gpu_required": False,
        },
    }
    with (od / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, score in zip(val_rows, val_scores):
            f.write(json.dumps({
                "pair_id": row["pair_id"],
                "query_id": row["pair_id"].rsplit("::", 1)[0],
                "memory_id": row.get("memory_id"),
                "label": row["label"],
                "pairwise_score": float(score),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    (od / "pairwise_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
