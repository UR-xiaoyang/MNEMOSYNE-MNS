"""Session-scoped working context stored as first-class MNEMOSYNE entities."""

from __future__ import annotations

import hashlib
from typing import Any

from context_assembly import assemble_context
from memory_schema import memory_from_api
from memory_store import MemoryStore


class ContextMemoryService:
    """Persist and retrieve working context without promoting it to long-term memory."""

    def __init__(self, store: MemoryStore, pipeline: Any | None = None) -> None:
        self.store = store
        self.pipeline = pipeline

    def record_turn(
        self,
        *,
        session_id: str,
        project_id: str = "",
        turn_id: str = "",
        user_text: str,
        assistant_text: str,
        destination: str = "local",
    ) -> str:
        if not session_id:
            raise ValueError("session_id is required for working context")
        identity = turn_id or hashlib.sha256(
            f"{session_id}\0{user_text}\0{assistant_text}".encode("utf-8")
        ).hexdigest()[:16]
        memory_id = f"session:{session_id}:{identity}"
        existing = self.store.get(memory_id)
        if existing is not None:
            return memory_id

        memory = memory_from_api(
            memory_id,
            self._format_turn(user_text, assistant_text),
            scope_level="session",
            destination=destination,
            session_id=session_id,
            project_id=project_id or None,
        )
        self.store.upsert(memory)
        return memory_id

    def record_checkpoint(
        self,
        *,
        session_id: str,
        turn_id: str,
        project_id: str = "",
        transcript: str,
        destination: str = "local",
    ) -> str:
        if not session_id:
            raise ValueError("session_id is required for a checkpoint")
        checkpoint_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:16]
        memory_id = f"session:{session_id}:checkpoint:{turn_id}:{checkpoint_hash}"
        if self.store.get(memory_id) is not None:
            return memory_id
        memory = memory_from_api(
            memory_id,
            f"[Session checkpoint]\n{transcript.strip()}",
            scope_level="session",
            destination=destination,
            session_id=session_id,
            project_id=project_id or None,
        )
        existing_checkpoints = self.store.list_session_checkpoints(session_id)
        for previous in existing_checkpoints:
            previous.validity.state = "superseded"
            previous.validity.superseded_by = memory_id
            previous.version += 1
            self.store.upsert(previous)
        self.store.upsert(memory)
        return memory_id

    def retrieve_context(
        self,
        query: str,
        *,
        session_id: str,
        project_id: str = "",
        scope_level: str = "session",
        destination: str = "local",
        top_k: int = 10,
        max_tokens: int = 4000,
        tokens_per_memory: int = 200,
    ) -> dict[str, Any]:
        candidates = self.store.search_candidates(
            query,
            scope_level=scope_level,
            destination=destination,
            limit=top_k,
            session_id=session_id,
            project_id=project_id or None,
        )
        ranked = [
            {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "rank": index,
                "score": float(len(candidates) - index + 1),
                "filter_passed": True,
                "injected": True,
                "metadata": {
                    "scope": memory.scope.level,
                    "authority_class": memory.authority.class_,
                    "authority_priority": memory.authority.priority,
                    "validity_state": memory.validity.state,
                },
            }
            for index, memory in enumerate(candidates, start=1)
        ]
        assembled = assemble_context(
            ranked,
            max_tokens=max_tokens,
            tokens_per_memory=tokens_per_memory,
        )
        return {
            "results": assembled,
            "ranked": ranked,
            "token_cost": sum(item.get("assembly_token_cost", tokens_per_memory) for item in assembled),
        }

    def retrieve_recent(
        self,
        *,
        session_id: str,
        project_id: str = "",
        max_tokens: int = 4000,
        tokens_per_memory: int = 200,
    ) -> dict[str, Any]:
        """Retrieve recent conversation context for a session in chronological order.
        
        Args:
            session_id: Required session identifier
            project_id: Optional project filter
            max_tokens: Token budget for context
            tokens_per_memory: Estimated tokens per turn
            
        Returns:
            Dict with results (ordered list of turns), token_cost
            
        Raises:
            ValueError: If session_id is missing
        """
        if not session_id:
            raise ValueError("session_id is required for context retrieval")
        
        # Get all session turns ordered by time (most recent first from DB)
        candidates = self.store.search_candidates(
            query="",  # Empty query gets all by recency
            scope_level="session",
            destination="local",
            limit=100,
            strict_validity=True,
            session_id=session_id,
            project_id=project_id or None,
        )
        
        # Filter to only turn entries (not checkpoints)
        turns = [
            m for m in candidates
            if m.memory_id.startswith(f"session:{session_id}:")
            and ":checkpoint:" not in m.memory_id
        ]
        
        # Reverse to get chronological order (oldest first)
        turns.reverse()
        
        # Convert to result format
        results = []
        used_tokens = 0
        
        for turn in turns:
            # Estimate tokens
            import re
            content = turn.content
            token_count = max(1, len(re.findall(r"\w+|[^\w\s]", content, re.UNICODE)))
            
            if used_tokens + token_count > max_tokens:
                break
            
            results.append({
                "memory_id": turn.memory_id,
                "content": content,
                "created_at": turn.created_at,
                "token_cost": token_count,
            })
            used_tokens += token_count
        
        return {
            "results": results,
            "token_cost": used_tokens,
        }

    @staticmethod
    def _format_turn(user_text: str, assistant_text: str) -> str:
        return f"[Session context]\nUser: {user_text.strip()}\nAssistant: {assistant_text.strip()}"
