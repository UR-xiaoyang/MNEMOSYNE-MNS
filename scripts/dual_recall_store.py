"""
双路召回存储层：FTS + FAISS 语义搜索

扩展 MemoryStore，提供：
- FTS 词汇匹配（top_k=50）
- FAISS ANN 语义匹配（top_k=50）
- 合并去重（top_k=100）
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np

from memory_schema import MemoryEntity, Scope, Authority, Validity, Retrieval, Usage, Egress, Provenance


class DualRecallStore:
    """双路召回存储层：FTS + FAISS"""
    
    def __init__(
        self,
        db_path: str | Path,
        faiss_index_path: str | Path | None = None,
        e5_model_path: str | Path | None = None,
        device: str = "cpu",
    ):
        """
        初始化双路召回存储
        
        Args:
            db_path: SQLite 数据库路径
            faiss_index_path: FAISS 索引路径（如果为 None，仅使用 FTS）
            e5_model_path: E5 模型路径（用于查询编码）
            device: 计算设备 "cpu" 或 "cuda"
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                version INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED, content,
                tokenize='unicode61'
            );
            """
        )
        self.connection.commit()
        
        # Initialize FAISS components
        self.faiss_enabled = False
        self.faiss_index = None
        self.memory_id_map = None
        self.encoder = None
        
        if faiss_index_path is not None:
            self._init_faiss(faiss_index_path, e5_model_path, device)
    
    def _init_faiss(
        self,
        faiss_index_path: str | Path,
        e5_model_path: str | Path | None,
        device: str,
    ) -> None:
        """初始化 FAISS 索引和编码器"""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS retrieval. "
                "Install it with: pip install faiss-cpu"
            )
        
        faiss_index_path = Path(faiss_index_path).expanduser()
        if not faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        
        # Load memory_id mapping
        # Safe: pickle file is created locally by build_faiss_index.py and not from untrusted sources
        mapping_path = faiss_index_path.with_suffix(".pkl")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Memory ID mapping not found: {mapping_path}")
        
        with open(mapping_path, "rb") as f:
            self.memory_id_map = pickle.load(f)
        
        if len(self.memory_id_map) != self.faiss_index.ntotal:
            raise ValueError(
                f"Memory ID mapping size mismatch: {len(self.memory_id_map)} vs {self.faiss_index.ntotal}"
            )
        
        # Load E5 encoder
        if e5_model_path is not None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from rank_inference import RankingBaseline
            
            # Create temporary metadata for E5-only encoding
            e5_model_path = Path(e5_model_path).expanduser()
            temp_metadata_path = e5_model_path / "metadata.json"
            created_metadata = False
            
            if not temp_metadata_path.exists():
                metadata = {
                    "model_id": "MNEMOSYNE-MNS-RANK-E5-FROZEN-01",
                    "encoder": str(e5_model_path),
                }
                temp_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                created_metadata = True
            
            try:
                self.encoder = RankingBaseline(e5_model_path, device=device)
            finally:
                if created_metadata:
                    temp_metadata_path.unlink()
        
        self.faiss_enabled = True
        print(f"FAISS enabled: {self.faiss_index.ntotal} vectors, dimension={self.faiss_index.d}")
    
    def upsert(self, memory: MemoryEntity) -> None:
        """插入或更新记忆"""
        payload = json.dumps(_to_payload(memory), ensure_ascii=False, separators=(",", ":"))
        self.connection.execute(
            """
            INSERT INTO memories(memory_id, content, payload_json, updated_at, version)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                content=excluded.content,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at,
                version=excluded.version
            """,
            (memory.memory_id, memory.content, payload, memory.updated_at, memory.version),
        )
        self.connection.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory.memory_id,))
        self.connection.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)", (memory.memory_id, memory.content))
        self.connection.commit()
    
    def get(self, memory_id: str) -> MemoryEntity | None:
        """获取单个记忆"""
        row = self.connection.execute("SELECT payload_json FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        return _from_payload(json.loads(row["payload_json"])) if row else None
    
    def search_candidates(
        self,
        query: str,
        *,
        scope_level: str,
        destination: str,
        limit: int = 100,
        strict_validity: bool = True,
        session_id: str | None = None,
        project_id: str | None = None,
        fts_top_k: int = 50,
        faiss_top_k: int = 50,
    ) -> list[MemoryEntity]:
        """
        双路召回搜索候选记忆
        
        Args:
            query: 查询文本
            scope_level: 作用域级别
            destination: 目标位置
            limit: 最终返回的最大候选数量
            strict_validity: 是否严格过滤有效性
            session_id: 会话 ID
            project_id: 项目 ID
            fts_top_k: FTS 召回的 top-K
            faiss_top_k: FAISS 召回的 top-K
        
        Returns:
            候选记忆列表（已去重）
        """
        candidates_dict = {}  # memory_id -> MemoryEntity，用于去重
        
        # Route 1: FTS 词汇召回
        fts_candidates = self._search_fts(query, fts_top_k)
        for memory in fts_candidates:
            candidates_dict[memory.memory_id] = memory
        
        # Route 2: FAISS 语义召回
        if self.faiss_enabled and self.encoder is not None:
            faiss_candidates = self._search_faiss(query, faiss_top_k)
            for memory in faiss_candidates:
                candidates_dict[memory.memory_id] = memory
        
        # 合并去重后的候选记忆
        all_candidates = list(candidates_dict.values())
        
        # 回退策略：如果没有候选，使用全局召回
        if not all_candidates:
            all_candidates = self._fallback_search(limit * 10)
        
        # 硬过滤
        filtered = [
            memory for memory in all_candidates
            if memory.hard_filter(
                scope_level,
                destination,
                strict_validity,
                session_id=session_id,
                project_id=project_id,
            )
        ]
        
        return filtered[:limit]
    
    def _search_fts(self, query: str, top_k: int) -> list[MemoryEntity]:
        """FTS 词汇召回"""
        fetch_limit = max(top_k * 2, 100)
        rows = self.connection.execute(
            """
            SELECT m.payload_json
            FROM memories_fts f
            JOIN memories m ON m.memory_id = f.memory_id
            WHERE memories_fts MATCH ?
            ORDER BY m.updated_at DESC
            LIMIT ?
            """,
            (_fts_query(query), fetch_limit),
        ).fetchall()
        
        candidates = [_from_payload(json.loads(row["payload_json"])) for row in rows]
        
        # 如果 FTS 没有结果，使用 LIKE 回退
        if not candidates:
            like_query = f"%{query.strip()}%"
            rows = self.connection.execute(
                "SELECT payload_json FROM memories WHERE content LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (like_query, fetch_limit),
            ).fetchall()
            candidates = [_from_payload(json.loads(row["payload_json"])) for row in rows]
        
        return candidates[:top_k]
    
    def _search_faiss(self, query: str, top_k: int) -> list[MemoryEntity]:
        """FAISS 语义召回"""
        if not self.faiss_enabled or self.encoder is None:
            return []
        
        import faiss
        
        # Encode query with "query: " prefix
        query_embedding = self.encoder.encode_text([query], prefix="query: ", batch_size=1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # Retrieve memories by memory_id
        candidates = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.memory_id_map):
                memory_id = self.memory_id_map[idx]
                memory = self.get(memory_id)
                if memory is not None:
                    candidates.append(memory)
        
        return candidates
    
    def _fallback_search(self, limit: int) -> list[MemoryEntity]:
        """回退策略：返回最近更新的记忆"""
        rows = self.connection.execute(
            "SELECT payload_json FROM memories ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_from_payload(json.loads(row["payload_json"])) for row in rows]
    
    def supersede(self, old_memory_id: str, replacement: MemoryEntity) -> MemoryEntity:
        """原子替换记忆"""
        old = self.get(old_memory_id)
        if old is None:
            raise KeyError(f"Unknown memory_id: {old_memory_id}")
        old.validity.state = "superseded"
        old.validity.superseded_by = replacement.memory_id
        old.version += 1
        replacement.version = max(replacement.version, old.version)
        self.upsert(old)
        replacement.provenance.parent_memory_id = old_memory_id
        self.upsert(replacement)
        return replacement
    
    def list_session_checkpoints(self, session_id: str) -> list[MemoryEntity]:
        """列出会话检查点"""
        prefix = f"session:{session_id}:checkpoint:%"
        rows = self.connection.execute(
            "SELECT payload_json FROM memories WHERE memory_id LIKE ? ORDER BY updated_at DESC",
            (prefix,),
        ).fetchall()
        return [_from_payload(json.loads(row["payload_json"])) for row in rows]
    
    def close(self) -> None:
        """关闭连接"""
        self.connection.close()


def _fts_query(query: str) -> str:
    """构建 FTS5 查询字符串"""
    tokens = [token.replace('"', "") for token in query.split() if token.strip()]
    return " OR ".join(f'"{token}"' for token in tokens) or '""'


def _to_payload(memory: MemoryEntity) -> dict:
    """将 MemoryEntity 转换为 JSON payload"""
    return {
        "memory_id": memory.memory_id,
        "content": memory.content,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "version": memory.version,
        "scope": vars(memory.scope),
        "authority": {**vars(memory.authority), "class": memory.authority.class_},
        "validity": vars(memory.validity),
        "retrieval": vars(memory.retrieval),
        "usage": vars(memory.usage),
        "egress": vars(memory.egress),
        "provenance": vars(memory.provenance),
    }


def _from_payload(payload: dict) -> MemoryEntity:
    """从 JSON payload 恢复 MemoryEntity"""
    authority = dict(payload["authority"])
    authority["class_"] = authority.pop("class")
    return MemoryEntity(
        memory_id=payload["memory_id"], content=payload["content"],
        created_at=payload["created_at"], updated_at=payload["updated_at"], version=payload["version"],
        scope=Scope(**payload["scope"]), authority=Authority(**authority),
        validity=Validity(**payload["validity"]), retrieval=Retrieval(**payload["retrieval"]),
        usage=Usage(**payload["usage"]), egress=Egress(**payload["egress"]),
        provenance=Provenance(**payload["provenance"]),
    )
