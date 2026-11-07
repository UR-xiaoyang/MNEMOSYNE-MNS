"""Experimental cross-conversation global memory storage.

This is an opt-in experimental feature that provides cross-conversation memory
WITHOUT reading or modifying Hermes' built-in global memory system.

Uses a dedicated SQLite table with explicit approved field validation.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any

from memory_store import MemoryStore


class ExperimentalGlobalMemory:
    """Manage experimental cross-conversation global memory entries.
    
    This operates on a dedicated table separate from regular memories.
    All writes require explicit approval (approved=true).
    Only approved=1 entries are returned on reads.
    """
    
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._ensure_table()
        
    def _ensure_table(self) -> None:
        """Create experimental_global_memories table if it doesn't exist."""
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experimental_global_memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    approved INTEGER NOT NULL DEFAULT 0,
                    source_session_id TEXT,
                    status TEXT DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    CHECK (approved IN (0, 1))
                )
            """)
            conn.commit()
        finally:
            conn.close()
        
    def write(
        self,
        memory_id: str,
        content: str,
        *,
        approved: bool,
        source_session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write an experimental global memory entry.
        
        Args:
            memory_id: Stable unique identifier
            content: Memory content
            approved: Must be True; False or missing raises ValueError
            source_session_id: Source session for lineage tracking
            metadata: Optional additional metadata
            
        Returns:
            Dict with memory_id, version, created_at, updated_at
            
        Raises:
            ValueError: If approved is not True, or memory_id is invalid
        """
        if not approved:
            raise ValueError("approved must be True for experimental global memory writes")
        
        if not memory_id or not memory_id.strip():
            raise ValueError("memory_id is required")
            
        if not content or not content.strip():
            raise ValueError("content is required")
        
        # Use experimental namespace prefix
        if not memory_id.startswith("experimental_global:"):
            memory_id = f"experimental_global:{memory_id}"
        
        conn = sqlite3.connect(self.store.db_path)
        try:
            # Check for duplicates
            cursor = conn.execute(
                "SELECT memory_id FROM experimental_global_memories WHERE memory_id = ?",
                (memory_id,)
            )
            if cursor.fetchone() is not None:
                raise ValueError(f"memory_id {memory_id} already exists")
            
            now = time.time()
            conn.execute(
                """INSERT INTO experimental_global_memories 
                   (memory_id, content, approved, source_session_id, status, created_at, updated_at)
                   VALUES (?, ?, 1, ?, 'active', ?, ?)""",
                (memory_id, content, source_session_id, now, now)
            )
            conn.commit()
            
            return {
                "memory_id": memory_id,
                "version": 1,
                "created_at": now,
                "updated_at": now,
                "approved": True,
            }
        finally:
            conn.close()
    
    def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all approved experimental global memories.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of memory dicts with id, content, created_at, updated_at
        """
        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """SELECT memory_id, content, source_session_id, created_at, updated_at
                   FROM experimental_global_memories
                   WHERE approved = 1 AND status = 'active'
                   ORDER BY created_at DESC, memory_id ASC
                   LIMIT ?""",
                (limit,)
            )
            
            return [
                {
                    "memory_id": row["memory_id"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "source_session_id": row["source_session_id"] or "",
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
    
    def retrieve(
        self,
        query: str,
        *,
        max_tokens: int = 2000,
        tokens_per_memory: int = 200,
    ) -> list[dict[str, Any]]:
        """Retrieve approved experimental global memories relevant to a query.
        
        Args:
            query: Search query
            max_tokens: Token budget
            tokens_per_memory: Estimated tokens per memory
            
        Returns:
            List of memories with id, content, rank, score
        """
        # Get all approved entries
        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """SELECT memory_id, content, created_at
                   FROM experimental_global_memories
                   WHERE approved = 1 AND status = 'active'
                   ORDER BY created_at DESC, memory_id ASC
                   LIMIT 50"""
            )
            rows = cursor.fetchall()
        finally:
            conn.close()
        
        if not rows:
            return []
        
        # Simple relevance scoring: recent entries first
        # In production, this would use semantic search
        ranked = [
            {
                "memory_id": row["memory_id"],
                "content": row["content"],
                "rank": idx + 1,
                "score": float(len(rows) - idx),
                "filter_passed": True,
                "injected": True,
                "metadata": {
                    "scope": "global",
                    "experimental": True,
                },
            }
            for idx, row in enumerate(rows)
        ]
        
        # Apply budget
        from context_assembly import assemble_context
        
        assembled = assemble_context(
            ranked,
            max_tokens=max_tokens,
            tokens_per_memory=tokens_per_memory,
        )
        
        return assembled
