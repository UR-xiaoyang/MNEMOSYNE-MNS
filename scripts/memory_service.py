"""Application service orchestrating route, store, rank, write and update."""

from __future__ import annotations

from typing import Any

from context_assembly import assemble_context
from memory_schema import memory_from_api
from memory_store import MemoryStore
from route_baseline import RouteDecision, route_query
from write_gate import proposal_payload


class MemoryService:
    def __init__(self, pipeline: Any, store: MemoryStore):
        self.pipeline = pipeline
        self.store = store

    def write(self, memory_id: str, content: str, *, scope_level: str, destination: str):
        memory = memory_from_api(memory_id, content, scope_level=scope_level, destination=destination)
        existing = self.store.get(memory_id)
        if existing is not None:
            memory.version = existing.version + 1
        self.store.upsert(memory)
        return memory

    def supersede(self, old_id: str, new_id: str, content: str, *, scope_level: str, destination: str):
        replacement = memory_from_api(new_id, content, scope_level=scope_level, destination=destination)
        return self.store.supersede(old_id, replacement)

    def propose_write(self, text: str, *, scope_level: str) -> list[dict]:
        return proposal_payload(text, scope_level=scope_level)

    def retrieve_store(
        self,
        query: str,
        *,
        scope_level: str,
        destination: str,
        top_k: int = 10,
        max_tokens: int = 4000,
        tokens_per_memory: int = 200,
    ) -> dict:
        decision: RouteDecision = route_query(query)
        candidates = self.store.search_candidates(
            query,
            scope_level=scope_level,
            destination=destination,
            limit=max(top_k * 10, 100),
        )
        ranked = self.pipeline.retrieve_and_rank(
            query_text=query,
            candidate_memories=candidates,
            scope_level=scope_level,
            destination=destination,
            path_hint=decision.primary_path,
            requires_high_authority=decision.requires_high_authority,
            top_k=top_k,
        ) if candidates else []
        return {
            "decision": decision,
            "ranked": ranked,
            "assembled": assemble_context(
                ranked,
                max_tokens=max_tokens,
                tokens_per_memory=tokens_per_memory,
            ),
        }
