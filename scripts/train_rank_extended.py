"""
Extended hybrid ranker with native features for MNEMOSYNE-MNS.

Trains pairwise logistic ranker with:
- 1 E5 semantic cosine
- 12 structured features (lexical, length, cues)
- 14 native features (scope, authority, validity, usage, path)
= 27 total features
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

# Add scripts to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from memory_schema import Query, Scope, extract_native_features, synthesize_native_metadata_for_alpsbench

DEFAULT_FEATURE_DIR = Path("C:/Users/xiaoyang/memory_dataset/features_v1")
DEFAULT_E5_MODEL = "intfloat/multilingual-e5-small"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def public_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("dataset") == "public"]


def grouped(rows: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        out[row["pair_id"].rsplit("::", 1)[0]].append(i)
    return out


def ranking_metrics(rows: list[dict], scores: np.ndarray) -> dict:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
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


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


def encode(texts: list[str], prefix: str, tokenizer, model, device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    result: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [prefix + text for text in texts[start : start + batch_size]]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            pooled = mean_pooling(model(**encoded), encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            result.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(result, axis=0) if result else np.empty((0, 0), dtype=np.float32)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--e5-model", default=DEFAULT_E5_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rank_extended_v1"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_all = read_jsonl(args.feature_dir / "rerank_train_pairs.jsonl")
    val_all = read_jsonl(args.feature_dir / "rerank_validation_pairs.jsonl")
    train_extra_all = np.load(args.feature_dir / "rerank_train_extra.npy")
    val_extra_all = np.load(args.feature_dir / "rerank_validation_extra.npy")
    train_rows = public_rows(train_all)
    val_rows = public_rows(val_all)
    train_ix = [i for i, row in enumerate(train_all) if row in train_rows]
    val_ix = [i for i, row in enumerate(val_all) if row in val_rows]
    train_extra = train_extra_all[train_ix]
    val_extra = val_extra_all[val_ix]

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.e5_model)
    model = AutoModel.from_pretrained(args.e5_model).to(device)
    
    print("Encoding training...", flush=True)
    train_query_emb = encode([r["query"] for r in train_rows], "query: ", tokenizer, model, device, args.batch_size, args.max_length)
    train_memory_emb = encode([r["memory"] for r in train_rows], "passage: ", tokenizer, model, device, args.batch_size, args.max_length)
    train_e5_cosine = np.sum(train_query_emb * train_memory_emb, axis=1, keepdims=True)
    
    print("Encoding validation...", flush=True)
    val_query_emb = encode([r["query"] for r in val_rows], "query: ", tokenizer, model, device, args.batch_size, args.max_length)
    val_memory_emb = encode([r["memory"] for r in val_rows], "passage: ", tokenizer, model, device, args.batch_size, args.max_length)
    val_e5_cosine = np.sum(val_query_emb * val_memory_emb, axis=1, keepdims=True)
    
    print("Synthesizing native metadata...", flush=True)
    train_native = []
    for i, row in enumerate(train_rows):
        memory_entity = synthesize_native_metadata_for_alpsbench(row["memory"], i, "train")
        query = Query(text=row["query"], scope=Scope(level="global"), destination="local", path="imported/alpsbench")
        native_feat = extract_native_features(query, memory_entity)
        train_native.append(native_feat)
    train_native = np.asarray(train_native, dtype=np.float32)
    
    val_native = []
    for i, row in enumerate(val_rows):
        memory_entity = synthesize_native_metadata_for_alpsbench(row["memory"], i, "val")
        query = Query(text=row["query"], scope=Scope(level="global"), destination="local", path="imported/alpsbench")
        native_feat = extract_native_features(query, memory_entity)
        val_native.append(native_feat)
    val_native = np.asarray(val_native, dtype=np.float32)
    
    print("Combining features: E5(1) + structured(12) + native(14) = 27...", flush=True)
    train_combined = np.concatenate([train_e5_cosine, train_extra, train_native], axis=1).astype(np.float32)
    val_combined = np.concatenate([val_e5_cosine, val_extra, val_native], axis=1).astype(np.float32)
    
    print("Training pairwise ranker...", flush=True)
    pair_x, pair_y = make_pairwise(train_rows, train_combined)
    scaler = StandardScaler()
    pair_scaled = scaler.fit_transform(pair_x)
    ranker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    ranker.fit(pair_scaled, pair_y)
    
    train_scores = scaler.transform(train_combined) @ ranker.coef_[0]
    val_scores = scaler.transform(val_combined) @ ranker.coef_[0]
    
    feature_names_json = json.loads((args.feature_dir / "rerank_extra_features.json").read_text(encoding="utf-8"))
    feature_names = ["e5_cosine"] + feature_names_json + [
        "scope_exact", "scope_visible", "authority_priority", "authority_reliable", "authority_is_user",
        "validity_active", "validity_stale", "validity_conflicted", "validity_days_verified",
        "usage_citation_rate", "usage_correction_rate", "usage_recency",
        "path_prefix", "path_exact"
    ]
    
    report = {
        "model_id": "MNEMOSYNE-MNS-RANK-EXTENDED-01",
        "components": {
            "semantic_encoder": args.e5_model,
            "semantic_mode": "frozen_cpu_inference",
            "structured_features": 12,
            "native_features": 14,
            "ranker": "logistic_pairwise",
        },
        "data_policy": "AlpsBench public train/validation; native metadata synthesized",
        "train_queries": len(grouped(train_rows)),
        "validation_queries": len(grouped(val_rows)),
        "train_pairs": int(len(pair_x)),
        "feature_dim": int(train_combined.shape[1]),
        "feature_names": feature_names,
        "train_metrics": ranking_metrics(train_rows, train_scores),
        "validation_metrics": ranking_metrics(val_rows, val_scores),
        "elapsed_seconds": round(time.time() - started, 3),
        "verification": {
            "grouped_pairwise_training": True,
            "public_only": True,
            "native_metadata_synthesized": True,
            "gpu_required": False,
        },
    }
    
    joblib.dump({
        "scaler": scaler,
        "ranker": ranker,
        "feature_names": feature_names,
        "e5_model": args.e5_model,
    }, args.output_dir / "extended_ranker.joblib")
    
    with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, score in zip(val_rows, val_scores):
            f.write(json.dumps({
                "pair_id": row["pair_id"],
                "query_id": row["pair_id"].rsplit("::", 1)[0],
                "memory_id": row.get("memory_id"),
                "label": row["label"],
                "extended_score": float(score),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    
    (args.output_dir / "extended_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
