"""Conservative conflict detection for memory entities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from memory_schema import MemoryEntity

NEGATION_RE = re.compile(r"(?:不要|禁止|不能|不得|永远不|never|do not|don't|avoid)", re.I)
CONTENT_RE = re.compile(r"(?:使用|采用|use|using)\s+(.+)", re.I)


@dataclass(frozen=True)
class ConflictProposal:
    left_id: str
    right_id: str
    reason: str
    confidence: float


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _opposite_negation(left: str, right: str) -> bool:
    left_negated = bool(NEGATION_RE.search(left))
    right_negated = bool(NEGATION_RE.search(right))
    if left_negated == right_negated:
        return False
    positive = right if left_negated else left
    negative = left if left_negated else right
    positive_tail = CONTENT_RE.search(positive)
    negative_tail = CONTENT_RE.search(negative)
    if positive_tail and negative_tail:
        return _normalized(positive_tail.group(1)) == _normalized(negative_tail.group(1))
    positive_words = set(re.findall(r"\w+|[\u4e00-\u9fff]", positive.lower(), re.UNICODE))
    negative_words = set(re.findall(r"\w+|[\u4e00-\u9fff]", negative.lower(), re.UNICODE))
    return len(positive_words & negative_words) >= 2


def detect_conflict(left: MemoryEntity, right: MemoryEntity) -> ConflictProposal | None:
    if _normalized(left.content) == _normalized(right.content):
        return None
    if _opposite_negation(left.content, right.content):
        return ConflictProposal(left.memory_id, right.memory_id, "opposite_negation", 0.9)
    return None


def find_conflicts(memories: list[MemoryEntity]) -> list[ConflictProposal]:
    proposals: list[ConflictProposal] = []
    for index, left in enumerate(memories):
        for right in memories[index + 1 :]:
            proposal = detect_conflict(left, right)
            if proposal is not None:
                proposals.append(proposal)
    return proposals
