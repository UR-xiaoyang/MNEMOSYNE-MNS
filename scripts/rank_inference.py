from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


class RankingBaseline:
    """Unified inference interface for MNEMOSYNE-MNS ranking baselines."""
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
        self.model_id = self.metadata["model_id"]
        
        if self.model_id == "MNEMOSYNE-MNS-RANK-E5-FROZEN-01":
            self.encoder_name = self.metadata["encoder"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            self.encoder = AutoModel.from_pretrained(self.encoder_name).to(self.device)
            self.encoder.eval()
            
        elif self.model_id == "MNEMOSYNE-MNS-RANK-HYBRID-01":
            # joblib.load reads locally trained model artifacts, not untrusted external data
            artifact = joblib.load(model_path / "hybrid_ranker.joblib")
            self.scaler = artifact["scaler"]
            self.ranker = artifact["ranker"]
            self.feature_names = artifact["feature_names"]
            self.encoder_name = artifact["e5_model"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            self.encoder = AutoModel.from_pretrained(self.encoder_name).to(self.device)
            self.encoder.eval()
            
        elif self.model_id == "MNEMOSYNE-MNS-RANK-TREE-01":
            # joblib.load reads locally trained model artifacts, not untrusted external data
            artifact = joblib.load(model_path / "tree_pairwise.joblib")
            self.scaler = artifact["scaler"]
            self.tree_model = artifact["model"]
            self.feature_names = artifact["feature_names"]
            
        else:
            raise ValueError(f"Unknown model_id: {self.model_id}")
    
    def encode_text(self, texts: list[str], prefix: str, batch_size: int = 64, max_length: int = 256) -> np.ndarray:
        """Encode texts using E5 with proper prefix."""
        result: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [prefix + text for text in texts[start : start + batch_size]]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                pooled = mean_pooling(self.encoder(**encoded), encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                result.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(result, axis=0) if result else np.empty((0, 0), dtype=np.float32)
    
    def score(self, queries: list[str], memories: list[str], extra_features: np.ndarray | None = None) -> np.ndarray:
        """
        Score query-memory pairs.
        
        Args:
            queries: List of query texts
            memories: List of memory texts (same length as queries)
            extra_features: Optional [N x 12] structured features for hybrid/tree models
        
        Returns:
            scores: [N] ranking scores (higher = more relevant)
        """
        if len(queries) != len(memories):
            raise ValueError(f"Length mismatch: {len(queries)} queries vs {len(memories)} memories")
        
        if self.model_id == "MNEMOSYNE-MNS-RANK-E5-FROZEN-01":
            query_emb = self.encode_text(queries, "query: ")
            memory_emb = self.encode_text(memories, "passage: ")
            return np.sum(query_emb * memory_emb, axis=1)
        
        elif self.model_id == "MNEMOSYNE-MNS-RANK-HYBRID-01":
            if extra_features is None:
                raise ValueError("Hybrid model requires extra_features")
            query_emb = self.encode_text(queries, "query: ")
            memory_emb = self.encode_text(memories, "passage: ")
            e5_cosine = np.sum(query_emb * memory_emb, axis=1, keepdims=True)
            combined = np.concatenate([e5_cosine, extra_features], axis=1).astype(np.float32)
            return self.scaler.transform(combined) @ self.ranker.coef_[0]
        
        elif self.model_id == "MNEMOSYNE-MNS-RANK-TREE-01":
            if extra_features is None:
                raise ValueError("Tree model requires extra_features")
            expanded = np.concatenate([extra_features, np.abs(extra_features), extra_features * extra_features], axis=1)
            return self.tree_model.predict_proba(self.scaler.transform(expanded))[:, 1]
        
        else:
            raise ValueError(f"Unsupported model_id: {self.model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified inference for MNEMOSYNE-MNS ranking baselines")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--query", type=str, help="Single query text for demo")
    parser.add_argument("--memory", type=str, help="Single memory text for demo")
    args = parser.parse_args()
    
    baseline = RankingBaseline(args.model_path)
    print(f"Loaded {baseline.model_id}")
    
    if args.query and args.memory:
        score = baseline.score([args.query], [args.memory])
        print(f"Query: {args.query}")
        print(f"Memory: {args.memory}")
        print(f"Score: {score[0]:.6f}")


if __name__ == "__main__":
    main()
