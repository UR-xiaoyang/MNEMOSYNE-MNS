"""Budgeted, conflict-aware context admission."""

from __future__ import annotations

import re
from typing import Any


def _estimate_tokens(content: str, fallback: int) -> int:
    # Conservative approximation for mixed Chinese/English text.
    if not content:
        return fallback
    return max(1, len(re.findall(r"\w+|[^\w\s]", content, re.UNICODE)))


def _near_duplicate(left: str, right: str, threshold: float = 0.85) -> bool:
    left_tokens = set(re.findall(r"\w+|[\u4e00-\u9fff]", left.lower(), re.UNICODE))
    right_tokens = set(re.findall(r"\w+|[\u4e00-\u9fff]", right.lower(), re.UNICODE))
    if not left_tokens or not right_tokens:
        return left.strip() == right.strip()
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= threshold


def assemble_context(
    ranked_results: list[dict[str, Any]],
    *,
    max_tokens: int = 4000,
    tokens_per_memory: int = 200,
    allow_conflicted: bool = False,
) -> list[dict[str, Any]]:
    """Admit ranked candidates under validity, conflict, dedup and budget rules."""
    if max_tokens < 0 or tokens_per_memory <= 0:
        raise ValueError("max_tokens must be non-negative and tokens_per_memory must be positive")

    by_id = {item.get("memory_id"): item for item in ranked_results}
    linked_conflicts: set[str] = set()
    for item in ranked_results:
        conflicts = item.get("metadata", {}).get("conflict_with", []) or []
        if conflicts:
            if item.get("memory_id") is not None:
                linked_conflicts.add(str(item["memory_id"]))
            linked_conflicts.update(str(conflict) for conflict in conflicts)

    selected: list[dict[str, Any]] = []
    used_tokens = 0
    for item in sorted(ranked_results, key=lambda value: (value.get("rank", 10**9), -value.get("score", 0.0))):
        state = item.get("metadata", {}).get("validity_state", "active")
        memory_id = item.get("memory_id")
        if state in {"superseded", "deprecated"}:
            item["injected"] = False
            continue
        if state == "conflicted" and not allow_conflicted:
            item["injected"] = False
            continue
        if memory_id in linked_conflicts:
            item["injected"] = False
            continue
        if any(_near_duplicate(item.get("content", ""), existing.get("content", "")) for existing in selected):
            item["injected"] = False
            continue

        cost = int(item.get("token_cost", _estimate_tokens(item.get("content", ""), tokens_per_memory)))
        if used_tokens + cost > max_tokens:
            item["injected"] = False
            continue
        item["injected"] = True
        item["assembly_token_cost"] = cost
        selected.append(item)
        used_tokens += cost

    return selected
