"""Local Python adapter for inserting MNEMOSYNE memory into agent turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from citation_detection import detect_citations_in_response
from memory_service import MemoryService
from shadow_mode import ShadowModeCollector
from shadow_runtime import record_citation, record_retrieval


@dataclass(frozen=True)
class MemoryAdapterConfig:
    """Runtime policy for a host agent's memory lifecycle."""

    shadow_enabled: bool = False
    top_k: int = 20
    max_context_tokens: int = 4000
    tokens_per_memory: int = 200


@dataclass
class TurnContext:
    """Memory material prepared before the host agent generates a response."""

    query: str
    session_id: str
    turn_id: str
    scope_level: str
    destination: str
    ranked: list[dict[str, Any]]
    assembled: list[dict[str, Any]]
    context_text: str
    retrieval_event_id: str | None = None


@dataclass
class TurnTrace:
    """Citation observations produced after a host agent response."""

    context: TurnContext
    response: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_event_ids: dict[str, str] = field(default_factory=dict)


class HermesMemoryAdapter:
    """Bridge a host agent turn to route/retrieve/rank/assemble and Shadow Mode.

    This adapter never commits a memory. Writes and supersedes remain explicit
    ``MemoryService`` operations so a generated response cannot mutate resident
    memory implicitly.
    """

    def __init__(
        self,
        service: MemoryService,
        shadow_collector: ShadowModeCollector | None = None,
        config: MemoryAdapterConfig | None = None,
    ) -> None:
        self.service = service
        self.shadow_collector = shadow_collector
        self.config = config or MemoryAdapterConfig()

    def before_turn(
        self,
        query: str,
        *,
        session_id: str,
        turn_id: str,
        scope_level: str = "global",
        destination: str = "local",
    ) -> TurnContext:
        """Retrieve and assemble memory context before response generation."""
        outcome = self.service.retrieve_store(
            query,
            scope_level=scope_level,
            destination=destination,
            top_k=self.config.top_k,
            max_tokens=self.config.max_context_tokens,
            tokens_per_memory=self.config.tokens_per_memory,
        )
        decision = outcome["decision"]
        ranked = list(outcome.get("ranked", []))
        assembled = list(outcome.get("assembled", []))
        route_path = getattr(decision, "primary_path", "")
        metadata = {
            "ranker_model_id": self._ranker_model_id(),
            "scope_level": scope_level,
            "destination": destination,
            "route_path": route_path,
            "injected_count": len(assembled),
        }
        candidates = [self._shadow_candidate(item) for item in ranked]
        retrieval_event_id = record_retrieval(
            self.shadow_collector,
            enabled=self.config.shadow_enabled,
            session_id=session_id,
            turn_id=turn_id,
            query={
                "text": query,
                "scope_level": scope_level,
                "destination": destination,
                "route_path": route_path,
            },
            candidates=candidates,
            metadata=metadata,
        )
        return TurnContext(
            query=query,
            session_id=session_id,
            turn_id=turn_id,
            scope_level=scope_level,
            destination=destination,
            ranked=ranked,
            assembled=assembled,
            context_text=self._format_context(assembled),
            retrieval_event_id=retrieval_event_id,
        )

    def after_turn(self, context: TurnContext, response: str) -> TurnTrace:
        """Detect citations and optionally persist citation/ignored observations."""
        citations = detect_citations_in_response(response, context.assembled)
        citation_event_ids: dict[str, str] = {}
        positions = {
            item.get("memory_id"): position
            for position, item in enumerate(context.assembled, start=1)
        }
        ranks = {
            item.get("memory_id"): int(item.get("rank", 0))
            for item in context.ranked
        }
        for citation in citations:
            memory_id = citation["memory_id"]
            event_id = record_citation(
                self.shadow_collector,
                enabled=self.config.shadow_enabled,
                session_id=context.session_id,
                turn_id=context.turn_id,
                retrieval_event_id=context.retrieval_event_id,
                memory_id=memory_id,
                citation=citation,
                original_rank=ranks.get(memory_id, 0),
                context_position=positions.get(memory_id),
            )
            if event_id is not None:
                citation_event_ids[memory_id] = event_id

        if self.config.shadow_enabled and self.shadow_collector is not None:
            cited_ids = {citation["memory_id"] for citation in citations}
            for item in context.assembled:
                memory_id = item.get("memory_id")
                if memory_id not in cited_ids and context.retrieval_event_id is not None:
                    self.shadow_collector.log_ignored(
                        session_id=context.session_id,
                        turn_id=context.turn_id,
                        retrieval_event_id=context.retrieval_event_id,
                        memory_id=str(memory_id),
                        rank=int(item.get("rank", 0)),
                        injected=True,
                    )

        return TurnTrace(
            context=context,
            response=response,
            citations=citations,
            citation_event_ids=citation_event_ids,
        )

    def record_correction(
        self,
        trace: TurnTrace,
        *,
        memory_id: str,
        correction_type: str,
        feedback: dict[str, Any],
    ) -> str | None:
        """Persist a correction only for a cited memory in enabled Shadow Mode."""
        if not self.config.shadow_enabled or self.shadow_collector is None:
            return None
        citation_event_id = trace.citation_event_ids.get(memory_id)
        if citation_event_id is None:
            return None
        event = self.shadow_collector.log_correction(
            session_id=trace.context.session_id,
            turn_id=trace.context.turn_id,
            citation_event_id=citation_event_id,
            memory_id=memory_id,
            correction_type=correction_type,
            feedback=feedback,
        )
        return event.event_id

    def _ranker_model_id(self) -> str:
        ranker = getattr(getattr(self.service, "pipeline", None), "ranker", None)
        return str(getattr(ranker, "model_id", "unknown"))

    @staticmethod
    def _shadow_candidate(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "memory_id": item.get("memory_id"),
            "rank": item.get("rank"),
            "score": item.get("score"),
            "injected": item.get("injected", False),
        }

    @staticmethod
    def _format_context(assembled: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            str(item.get("content", "")).strip()
            for item in assembled
            if str(item.get("content", "")).strip()
        )
