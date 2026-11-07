"""
构建 FAISS 索引用于语义召回

使用 E5 embeddings（intfloat/multilingual-e5-small）编码所有记忆内容
构建 IndexFlatIP（内积索引，384维）并保存到 deploy/faiss_index.bin
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from rank_inference import RankingBaseline, resolve_model_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_index(
    db_path: str | Path,
    output_path: str | Path,
    e5_model_path: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """
    构建 FAISS 索引
    
    Args:
        db_path: SQLite 数据库路径
        output_path: FAISS 索引输出路径
        e5_model_path: E5 模型路径
        batch_size: 编码批次大小
        device: 计算设备 "cpu" 或 "cuda"
    
    Returns:
        统计信息字典
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required for FAISS indexing. "
            "Install it with: pip install faiss-cpu"
        )
    
    # Load memories from database
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT memory_id, content FROM memories ORDER BY updated_at DESC").fetchall()
    conn.close()
    
    if not rows:
        raise ValueError("No memories found in database")
    
    memory_ids = [row["memory_id"] for row in rows]
    contents = [row["content"] for row in rows]
    
    print(f"Loaded {len(memory_ids)} memories from {db_path}")
    
    # Initialize E5 encoder
    # We use the E5 model through RankingBaseline's encode_text method
    # Create a minimal metadata.json to satisfy RankingBaseline
    temp_metadata_path = Path(e5_model_path) / "metadata.json"
    if not temp_metadata_path.exists():
        # Create temporary metadata for E5-only encoding
        metadata = {
            "model_id": "MNEMOSYNE-MNS-RANK-E5-FROZEN-01",
            "encoder": str(e5_model_path),
        }
        temp_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        created_metadata = True
    else:
        created_metadata = False
    
    try:
        encoder = RankingBaseline(e5_model_path, device=device)
        print(f"Loaded E5 encoder: {encoder.encoder_name}")
        
        # Encode all memories with "passage: " prefix
        print(f"Encoding {len(contents)} memories...")
        embeddings = encoder.encode_text(contents, prefix="passage: ", batch_size=batch_size)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Build FAISS index (IndexFlatIP for inner product)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        print(f"Built FAISS index with {index.ntotal} vectors, dimension={dimension}")
        
        # Save index and memory_id mapping
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(output_path))
        
        # Save memory_id mapping
        mapping_path = output_path.with_suffix(".pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump(memory_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved FAISS index to {output_path}")
        print(f"Saved memory_id mapping to {mapping_path}")
        
        return {
            "total_memories": len(memory_ids),
            "dimension": dimension,
            "index_path": str(output_path),
            "mapping_path": str(mapping_path),
        }
    
    finally:
        if created_metadata:
            temp_metadata_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for semantic retrieval")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "memory.db",
        help="Path to SQLite database (default: memory.db in project root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "deploy/faiss_index.bin",
        help="Output path for FAISS index (default: deploy/faiss_index.bin)",
    )
    parser.add_argument(
        "--e5-model",
        type=Path,
        default=PROJECT_ROOT / "deploy/e5_model",
        help="Path to E5 model directory (default: deploy/e5_model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for encoding (default: cpu)",
    )
    
    args = parser.parse_args()
    
    stats = build_index(
        db_path=args.db_path,
        output_path=args.output,
        e5_model_path=args.e5_model,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    print("\nIndex build completed:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
