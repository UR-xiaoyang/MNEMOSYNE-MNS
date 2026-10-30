"""Local SQLite memory store with scope, egress and validity filtering."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memory_schema import Authority, Egress, MemoryEntity, Provenance, Retrieval, Scope, Usage, Validity


class MemoryStore:
    def __init__(self, db_path: str | Path):
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

    def upsert(self, memory: MemoryEntity) -> None:
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
            """ ,
            (memory.memory_id, memory.content, payload, memory.updated_at, memory.version),
        )
        self.connection.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory.memory_id,))
        self.connection.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)", (memory.memory_id, memory.content))
        self.connection.commit()

    def get(self, memory_id: str) -> MemoryEntity | None:
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
    ) -> list[MemoryEntity]:
        fetch_limit = max(limit * 10, 100)
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
        if not candidates:
            like_query = f"%{query.strip()}%"
            rows = self.connection.execute(
                "SELECT payload_json FROM memories WHERE content LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (like_query, fetch_limit),
            ).fetchall()
            candidates = [_from_payload(json.loads(row["payload_json"])) for row in rows]
        if not candidates:
            rows = self.connection.execute(
                "SELECT payload_json FROM memories ORDER BY updated_at DESC LIMIT ?", (fetch_limit,)
            ).fetchall()
            candidates = [_from_payload(json.loads(row["payload_json"])) for row in rows]
        return [
            memory for memory in candidates
            if memory.hard_filter(scope_level, destination, strict_validity)
        ][:limit]

    def supersede(self, old_memory_id: str, replacement: MemoryEntity) -> MemoryEntity:
        """Atomically mark the old entity superseded and persist its replacement."""
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

    def close(self) -> None:
        self.connection.close()



def _fts_query(query: str) -> str:
    tokens = [token.replace('"', "") for token in query.split() if token.strip()]
    return " OR ".join(f'"{token}"' for token in tokens) or '""'


def _to_payload(memory: MemoryEntity) -> dict:
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
