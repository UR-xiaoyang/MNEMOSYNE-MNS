from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_DIR = Path("C:/Users/xiaoyang/memory_dataset/features_v1")
OUTPUT_DIR = Path("runs/rank_pairwise_tree_v1")
FEATURE_NAMES = json.loads((FEATURE_DIR / "rerank_extra_features.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def public_indices(rows: list[dict]) -> list[int]:
    return [i for i, row in enumerate(rows) if row.get("dataset") == "public"]


def grouped(rows: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["pair_id"].rsplit("::", 1)[0]].append(i)
    return groups


def ranking_metrics(rows: list[dict], scores: np.ndarray) -> dict:
    groups = defaultdict(list)
    for row, score in zip(rows, scores):
        groups[row["pair_id"].rsplit("::", 1)[0]].append((float(score), int(row["label"])))
    values = {key: [] for key in ("Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@5", "NDCG@10")}
    for candidates in groups.values():
        ranked = sorted(candidates, key=lambda x: x[0], reverse=True)
        positive_count = sum(label for _, label in ranked)
        if positive_count == 0:
            continue
        values["Recall@1"].append(float(any(label for _, label in ranked[:1])))
        values["Recall@5"].append(float(any(label for _, label in ranked[:5])))
        values["Recall@10"].append(float(any(label for _, label in ranked[:10])))
        first = next(i + 1 for i, (_, label) in enumerate(ranked) if label)
        values["MRR"].append(1.0 / first)
        for k in (5, 10):
            dcg = sum(label / math.log2(i + 2) for i, (_, label) in enumerate(ranked[:k]))
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, positive_count)))
            values[f"NDCG@{k}"].append(dcg / ideal if ideal else 0.0)
    return {key: float(np.mean(value)) if value else 0.0 for key, value in values.items()} | {"queries": len(groups)}


def make_pairwise(rows: list[dict], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = grouped(rows)
    deltas, labels = [], []
    for indices in groups.values():
        positives = [i for i in indices if rows[i]["label"] == 1]
        negatives = [i for i in indices if rows[i]["label"] == 0]
        for pos in positives:
            for neg in negatives:
                delta = features[pos] - features[neg]
                forward = np.concatenate([delta, np.abs(delta), features[pos] * features[neg]])
                reverse_delta = features[neg] - features[pos]
                reverse = np.concatenate([reverse_delta, np.abs(reverse_delta), features[neg] * features[pos]])
                deltas.extend((forward, reverse))
                labels.extend((1, 0))
    return np.asarray(deltas, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all = read_jsonl(FEATURE_DIR / "rerank_train_pairs.jsonl")
    val_all = read_jsonl(FEATURE_DIR / "rerank_validation_pairs.jsonl")
    train_extra_all = np.load(FEATURE_DIR / "rerank_train_extra.npy")
    val_extra_all = np.load(FEATURE_DIR / "rerank_validation_extra.npy")
    train_ix, val_ix = public_indices(train_all), public_indices(val_all)
    train_rows = [train_all[i] for i in train_ix]
    val_rows = [val_all[i] for i in val_ix]
    train_features = train_extra_all[train_ix]
    val_features = val_extra_all[val_ix]

    started = time.time()
    pair_x, pair_y = make_pairwise(train_rows, train_features)
    scaler = StandardScaler()
    pair_x_scaled = scaler.fit_transform(pair_x)
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        min_samples_leaf=30,
        random_state=42,
    )
    model.fit(pair_x_scaled, pair_y)

    def borda_scores(rows: list[dict], features: np.ndarray) -> np.ndarray:
        """Rank candidates by mean pairwise win probability within each query."""
        result = np.zeros(len(rows), dtype=np.float32)
        for indices in grouped(rows).values():
            if len(indices) == 1:
                result[indices[0]] = 0.5
                continue
            left, right = [], []
            pairs = []
            for i in indices:
                for j in indices:
                    if i != j:
                        left.append(features[i])
                        right.append(features[j])
                        pairs.append((i, j))
            left = np.asarray(left, dtype=np.float32)
            right = np.asarray(right, dtype=np.float32)
            pair_features = np.concatenate([left - right, np.abs(left - right), left * right], axis=1)
            win_probs = model.predict_proba(scaler.transform(pair_features))[:, 1]
            total = defaultdict(float)
            count = defaultdict(int)
            for (i, _), probability in zip(pairs, win_probs):
                total[i] += float(probability)
                count[i] += 1
            for i in indices:
                result[i] = total[i] / count[i]
        return result

    val_scores = borda_scores(val_rows, val_features)
    train_scores = borda_scores(train_rows, train_features)

    report = {
        "model_id": "MNEMOSYNE-MNS-RANK-TREE-01",
        "data_policy": "AlpsBench public train/validation only; private weak data excluded",
        "train_queries": len(grouped(train_rows)),
        "validation_queries": len(grouped(val_rows)),
        "train_pairs": int(len(pair_x)),
        "feature_dim_before_expansion": int(train_features.shape[1]),
        "feature_dim_after_expansion": int(pair_x.shape[1]),
        "train_metrics": ranking_metrics(train_rows, train_scores),
        "validation_metrics": ranking_metrics(val_rows, val_scores),
        "elapsed_seconds": round(time.time() - started, 3),
        "verification": {
            "grouped_pairwise_training": True,
            "public_only": True,
            "private_weak_data_used": False,
            "gpu_required": False,
        },
    }
    joblib.dump({"scaler": scaler, "model": model, "feature_names": FEATURE_NAMES}, OUTPUT_DIR / "tree_pairwise.joblib")
    (OUTPUT_DIR / "tree_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, score in zip(val_rows, val_scores):
            f.write(json.dumps({"pair_id": row["pair_id"], "query_id": row["pair_id"].rsplit("::", 1)[0], "memory_id": row.get("memory_id"), "label": row["label"], "tree_score": float(score)}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
