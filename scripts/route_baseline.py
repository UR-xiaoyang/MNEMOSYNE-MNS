"""Auditable deterministic ROUTE baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    primary_path: str
    candidate_paths: list[str]
    confidence_proxy: float
    force_slow_fallback: bool
    fallback_reason: str | None
    query_kind: str
    requires_high_authority: bool


RULES = {
    "user/constraint": (re.compile(r"(?:禁止|不要|不能|必须|不得|never|do not|must)", re.I), "constraint"),
    "user/preference": (re.compile(r"(?:偏好|喜欢|习惯|prefer|preference|默认使用|回复使用)", re.I), "preference"),
    "engineering/code": (re.compile(r"(?:git|python|代码|配置|服务器|API|docker|训练|编译|ssh|proxy|代理)", re.I), "engineering"),
    "project/decision": (re.compile(r"(?:决定|方案|架构|项目|最终选择|decision|architecture|plan)", re.I), "decision"),
    "task/state": (re.compile(r"(?:继续|上次|之前|刚才|当前|进展|状态|continue|previous|resume|status)", re.I), "reference"),
}
REFERENCE_RE = re.compile(r"(?:继续|上次|之前|刚才|前面|那个|历史|continue|previous|earlier|resume|that)", re.I)


def route_query(text: str) -> RouteDecision:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return RouteDecision("unclassified", ["unclassified"], 0.0, True, "low_route_confidence", "unknown", False)

    hits: list[tuple[str, str, int]] = []
    for path, (pattern, kind) in RULES.items():
        count = len(pattern.findall(cleaned))
        if count:
            hits.append((path, kind, count))

    hits.sort(key=lambda item: (-item[2], item[0]))
    if not hits:
        return RouteDecision("unclassified", ["unclassified"], 0.2, True, "low_route_confidence", "unknown", False)

    primary, kind, count = hits[0]
    candidates = [path for path, _, _ in hits]
    confidence = min(0.95, 0.45 + 0.15 * count + 0.05 * max(0, len(hits) - 1))
    is_reference = bool(REFERENCE_RE.search(cleaned))
    if is_reference and "task/state" not in candidates:
        candidates.append("task/state")
    force = is_reference or confidence < 0.5
    reason = "reference_or_history" if is_reference else ("low_route_confidence" if force else None)
    return RouteDecision(
        primary_path=primary,
        candidate_paths=candidates,
        confidence_proxy=confidence,
        force_slow_fallback=force,
        fallback_reason=reason,
        query_kind=kind,
        requires_high_authority=kind == "constraint",
    )
