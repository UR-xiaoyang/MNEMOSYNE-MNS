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
