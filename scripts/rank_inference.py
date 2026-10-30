from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from structured_features import build_structured_features
except ModuleNotFoundError:  # Package import: ``from scripts.rank_inference ...``
    from .structured_features import build_structured_features


def _load_neural_dependencies():
    import torch
    from transformers import AutoModel, AutoTokenizer

    return torch, AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_path(model_path: str | Path) -> Path:
    """Resolve model paths independently of the caller's working directory."""
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def mean_pooling(model_output, attention_mask, torch_module):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / torch_module.clamp(mask.sum(1), min=1e-9)


class RankingBaseline:
    """Unified inference interface for MNEMOSYNE-MNS ranking baselines."""
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = resolve_model_path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        self.torch, auto_model, auto_tokenizer = _load_neural_dependencies()
        self.auto_model = auto_model
        self.auto_tokenizer = auto_tokenizer
        self.device = self.torch.device(device)
        self.metadata = json.loads((self.model_path / "metadata.json").read_text(encoding="utf-8"))
        self.model_id = self.metadata["model_id"]
        
        if self.model_id == "MNEMOSYNE-MNS-RANK-E5-FROZEN-01":
            self.encoder_name = os.environ.get("MNEMOSYNE_E5_MODEL", self.metadata["encoder"])
            self.tokenizer = self.auto_tokenizer.from_pretrained(self.encoder_name, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
            self.encoder = self.auto_model.from_pretrained(self.encoder_name, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1").to(self.device)
            self.encoder.eval()
            
        elif self.model_id == "MNEMOSYNE-MNS-RANK-HYBRID-01":
            artifact_path = self.model_path / "hybrid_ranker.joblib"
            if artifact_path.exists():
                # joblib.load reads locally trained model artifacts, not untrusted external data
                artifact = joblib.load(artifact_path)
                self.scaler = artifact["scaler"]
                self.ranker = artifact["ranker"]
                self.feature_names = artifact["feature_names"]
                self.encoder_name = artifact["e5_model"]
                self.vectorizer = artifact.get("vectorizer")
            else:
                scaler_payload = json.loads((self.model_path / "scaler.json").read_text(encoding="utf-8"))
                ranker_payload = json.loads((self.model_path / "ranker.json").read_text(encoding="utf-8"))
                self.scaler_mean = np.asarray(scaler_payload["mean"], dtype=np.float32)
                self.scaler_scale = np.asarray(scaler_payload["scale"], dtype=np.float32)
                self.ranker_coef = np.asarray(ranker_payload["coef"], dtype=np.float32)
                self.ranker_intercept = float(ranker_payload["intercept"])
                self.feature_names = ranker_payload["feature_names"]
                self.encoder_name = ranker_payload.get("e5_model", self.metadata["components"]["semantic_encoder"])
                self.vectorizer = None
            self.encoder_name = os.environ.get("MNEMOSYNE_E5_MODEL", self.encoder_name)
            offline = os.environ.get("HF_HUB_OFFLINE") == "1"
            self.tokenizer = self.auto_tokenizer.from_pretrained(self.encoder_name, local_files_only=offline)
            self.encoder = self.auto_model.from_pretrained(self.encoder_name, local_files_only=offline).to(self.device)
            self.encoder.eval()
            
        elif self.model_id == "MNEMOSYNE-MNS-RANK-TREE-01":
            # joblib.load reads locally trained model artifacts, not untrusted external data
            artifact = joblib.load(self.model_path / "tree_pairwise.joblib")
            self.scaler = artifact["scaler"]
            self.tree_model = artifact["model"]
            self.feature_names = artifact["feature_names"]
            
        else:
            raise ValueError(f"Unknown model_id: {self.model_id}")
    
    def encode_text(self, texts: list[str], prefix: str, batch_size: int = 64, max_length: int = 256) -> np.ndarray:
        """Encode texts using E5 with proper prefix."""
        result: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [prefix + text for text in texts[start : start + batch_size]]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                pooled = mean_pooling(self.encoder(**encoded), encoded["attention_mask"], self.torch)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                result.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(result, axis=0) if result else np.empty((0, 0), dtype=np.float32)
    
    def score(self, queries: list[str], memories: list[str], extra_features: np.ndarray | None = None) -> np.ndarray:
        """
        Score query-memory pairs. Hybrid models compute structured features
        automatically when callers do not provide them.
        """
        if len(queries) != len(memories):
            raise ValueError(f"Length mismatch: {len(queries)} queries vs {len(memories)} memories")
        
        if self.model_id == "MNEMOSYNE-MNS-RANK-E5-FROZEN-01":
            query_emb = self.encode_text(queries, "query: ")
            memory_emb = self.encode_text(memories, "passage: ")
            return np.sum(query_emb * memory_emb, axis=1)
        
        elif self.model_id == "MNEMOSYNE-MNS-RANK-HYBRID-01":
            if extra_features is None:
                extra_features, _ = build_structured_features(
                    queries,
                    memories,
                    vectorizer=self.vectorizer,
                )
            if extra_features.shape != (len(queries), 12):
                raise ValueError(
                    f"Hybrid model requires [N, 12] structured features; got {extra_features.shape}"
                )
            query_emb = self.encode_text(queries, "query: ")
            memory_emb = self.encode_text(memories, "passage: ")
            e5_cosine = np.sum(query_emb * memory_emb, axis=1, keepdims=True)
            combined = np.concatenate([e5_cosine, extra_features], axis=1).astype(np.float32)
            if hasattr(self, "ranker"):
                return self.scaler.transform(combined) @ self.ranker.coef_[0]
            scaled = (combined - self.scaler_mean) / np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
            return scaled @ self.ranker_coef + self.ranker_intercept
        
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
