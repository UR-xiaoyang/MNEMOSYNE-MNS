from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_DIR = Path("C:/Users/xiaoyang/memory_dataset/features_v1")
OUTPUT_DIR = Path("runs/rank_pairwise_ablation_v1")
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
                deltas.extend((delta, -delta))
                labels.extend((1, 0))
    return np.asarray(deltas, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def train_variant(name: str, keep: list[str], train_rows: list[dict], train_features: np.ndarray, val_rows: list[dict], val_features: np.ndarray) -> dict:
    indices = [FEATURE_NAMES.index(feature) for feature in keep]
    train_selected = train_features[:, indices]
    val_selected = val_features[:, indices]
    pair_x, pair_y = make_pairwise(train_rows, train_selected)
    scaler = StandardScaler()
    pair_scaled = scaler.fit_transform(pair_x)
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(pair_scaled, pair_y)
    weights = model.coef_[0]
    val_scores = scaler.transform(val_selected) @ weights
    result = {
        "variant": name,
        "features": keep,
        "feature_count": len(keep),
        "train_pairs": int(len(pair_x)),
        "validation": ranking_metrics(val_rows, val_scores),
    }
    joblib.dump({"scaler": scaler, "model": model, "weights": weights, "features": keep}, OUTPUT_DIR / f"{name}.joblib")
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows_all = read_jsonl(FEATURE_DIR / "rerank_train_pairs.jsonl")
    val_rows_all = read_jsonl(FEATURE_DIR / "rerank_validation_pairs.jsonl")
    train_ix = public_indices(train_rows_all)
    val_ix = public_indices(val_rows_all)
    train_rows = [train_rows_all[i] for i in train_ix]
    val_rows = [val_rows_all[i] for i in val_ix]
    train_features_all = np.load(FEATURE_DIR / "rerank_train_extra.npy")
    val_features_all = np.load(FEATURE_DIR / "rerank_validation_extra.npy")
    train_features = train_features_all[train_ix]
    val_features = val_features_all[val_ix]

    variants = {
        "all": FEATURE_NAMES,
        "similarity_only": ["tfidf_cosine", "lexical_jaccard", "lexical_containment"],
        "similarity_length": ["tfidf_cosine", "lexical_jaccard", "lexical_containment", "query_len_log", "memory_len_log"],
        "similarity_cues": ["tfidf_cosine", "lexical_jaccard", "lexical_containment", "query_has_reference", "query_has_negation", "memory_has_negation", "memory_has_update"],
        "similarity_provenance": ["tfidf_cosine", "lexical_jaccard", "lexical_containment", "same_source_public", "weak_proxy", "sample_weight"],
        "no_tfidf": [name for name in FEATURE_NAMES if name != "tfidf_cosine"],
    }
    started = time.time()
    results = [train_variant(name, features, train_rows, train_features, val_rows, val_features) for name, features in variants.items()]
    results.sort(key=lambda x: x["validation"]["NDCG@10"], reverse=True)
    report = {
        "model_family": "MNEMOSYNE-MNS-RANK-PAIRWISE-01 feature ablation",
        "data_policy": "AlpsBench public train/validation only; private weak data excluded",
        "train_queries": len(grouped(train_rows)),
        "validation_queries": len(grouped(val_rows)),
        "feature_names": FEATURE_NAMES,
        "results": results,
        "elapsed_seconds": round(time.time() - started, 3),
        "interpretation_policy": "Ablation ranks variants for diagnosis; no final model selection from one validation split alone.",
    }
    (OUTPUT_DIR / "ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
