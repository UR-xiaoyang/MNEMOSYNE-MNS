from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_FEATURE_DIR = Path("C:/Users/xiaoyang/memory_dataset/features_v1")
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


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


def encode(
    texts: list[str],
    prefix: str,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
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
            print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(result, axis=0) if result else np.empty((0, 0), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rank_embedding_e5_v1"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows = public_rows(read_jsonl(args.feature_dir / "rerank_validation_pairs.jsonl"))
    query_texts = []
    memory_texts = []
    query_id_to_index: dict[str, int] = {}
    memory_to_index: dict[str, int] = {}
    pair_query_index, pair_memory_index = [], []
    for row in rows:
        qid = row["pair_id"].rsplit("::", 1)[0]
        query = row["query"]
        memory = row["memory"]
        if qid not in query_id_to_index:
            query_id_to_index[qid] = len(query_texts)
            query_texts.append(query)
        if memory not in memory_to_index:
            memory_to_index[memory] = len(memory_texts)
            memory_texts.append(memory)
        pair_query_index.append(query_id_to_index[qid])
        pair_memory_index.append(memory_to_index[memory])

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    query_embeddings = encode(query_texts, "query: ", tokenizer, model, device, args.batch_size, args.max_length)
    memory_embeddings = encode(memory_texts, "passage: ", tokenizer, model, device, args.batch_size, args.max_length)
    scores = np.sum(query_embeddings[np.asarray(pair_query_index)] * memory_embeddings[np.asarray(pair_memory_index)], axis=1)

    report = {
        "model_id": "MNEMOSYNE-MNS-RANK-E5-FROZEN-01",
        "encoder": args.model,
        "encoder_mode": "frozen_cpu_inference",
        "data_policy": "AlpsBench public validation only; no fitting and no private weak data",
        "queries": len(query_texts),
        "unique_candidate_memories": len(memory_texts),
        "pair_count": len(rows),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "metrics": ranking_metrics(rows, scores),
        "elapsed_seconds": round(time.time() - started, 3),
        "verification": {
            "public_only": True,
            "private_weak_data_used": False,
            "weights_fine_tuned": False,
            "gpu_required": False,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        },
    }
    np.save(args.output_dir / "query_embeddings.npy", query_embeddings)
    np.save(args.output_dir / "memory_embeddings.npy", memory_embeddings)
    with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, score in zip(rows, scores):
            f.write(json.dumps({
                "pair_id": row["pair_id"],
                "query_id": row["pair_id"].rsplit("::", 1)[0],
                "memory_id": row.get("memory_id"),
                "label": row["label"],
                "e5_cosine": float(score),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    (args.output_dir / "embedding_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
