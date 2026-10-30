"""Small integration helpers for optional local shadow tracing."""

from __future__ import annotations

from typing import Any

from shadow_mode import ShadowModeCollector


def record_retrieval(
    collector: ShadowModeCollector | None,
    *,
    enabled: bool,
    session_id: str,
    turn_id: str,
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str | None:
    """Record a retrieval only when explicitly enabled."""
    if not enabled or collector is None:
        return None
    event = collector.log_retrieval(
        session_id=session_id,
        turn_id=turn_id,
        query=query,
        candidates=candidates,
        metadata=metadata,
    )
    return event.event_id

def record_citation(
    collector: ShadowModeCollector | None,
    *,
    enabled: bool,
    session_id: str,
    turn_id: str,
    retrieval_event_id: str | None,
    memory_id: str,
    citation: dict[str, Any],
    original_rank: int,
    context_position: int | None = None,
) -> str | None:
    if not enabled or collector is None or retrieval_event_id is None:
        return None
    event = collector.log_citation(
        session_id=session_id,
        turn_id=turn_id,
        retrieval_event_id=retrieval_event_id,
        memory_id=memory_id,
        citation=citation,
        context_position=context_position,
        original_rank=original_rank,
    )
    return event.event_id

def record_correction(
    collector: ShadowModeCollector | None,
    *,
    enabled: bool,
    session_id: str,
    turn_id: str,
    citation_event_id: str | None,
    memory_id: str,
    correction_type: str,
    feedback: dict[str, Any],
) -> str | None:
    if not enabled or collector is None or citation_event_id is None:
        return None
    event = collector.log_correction(
        session_id=session_id,
        turn_id=turn_id,
        citation_event_id=citation_event_id,
        memory_id=memory_id,
        correction_type=correction_type,
        feedback=feedback,
    )
    return event.event_id
