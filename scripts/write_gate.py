"""Conservative WRITE expert baseline: propose, never silently commit."""

from __future__ import annotations

import re
from dataclasses import dataclass


SENSITIVE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?:api[_ -]?key|token|password|secret|密钥|密码)\s*[:=：]\s*\S+", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
QUESTION_RE = re.compile(r"(?:？|\?|如何|怎么|能否|是否|请问|why|how|can you|could you)", re.I)
REQUEST_RE = re.compile(r"(?:请|帮我|执行|运行|开始|继续|do|run|execute|please)", re.I)
PREFERENCE_RE = re.compile(r"(?:偏好|喜欢|习惯|prefer|preference|always use|默认使用|回复使用)", re.I)
CONSTRAINT_RE = re.compile(r"(?:不要|禁止|不能|必须|永远不|不得|never|do not|must|禁止)", re.I)
TRANSIENT_RE = re.compile(r"(?:现在|当前|临时|这次|正在|today|currently|temporary|this run|running)", re.I)


@dataclass
class WriteProposal:
    content: str
    kind: str
    scope_level: str
    confidence: float
    admissible: bool
    rejection_reason: str | None = None
    source: str = "user_visible_text"


def _sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def extract_write_proposals(text: str, *, scope_level: str = "session") -> list[WriteProposal]:
    """Extract conservative proposals from one visible user message.

    This v1 intentionally rejects questions/requests and transient state. A
    later learned model may rank proposals, but hard safety gates remain here.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    if _sensitive(cleaned):
        return [WriteProposal(cleaned, "sensitive", "session", 1.0, False, "sensitive_content")]
    if QUESTION_RE.search(cleaned):
        return []
    if TRANSIENT_RE.search(cleaned) and not PREFERENCE_RE.search(cleaned) and not CONSTRAINT_RE.search(cleaned):
        return [WriteProposal(cleaned, "task_state", "session", 0.72, False, "transient_state")]
    if REQUEST_RE.search(cleaned) and not (PREFERENCE_RE.search(cleaned) or CONSTRAINT_RE.search(cleaned)):
        return []
    if CONSTRAINT_RE.search(cleaned):
        return [WriteProposal(cleaned, "constraint", scope_level, 0.9, True)]
    if PREFERENCE_RE.search(cleaned):
        return [WriteProposal(cleaned, "preference", scope_level, 0.88, True)]
    return []


def proposal_payload(text: str, *, scope_level: str = "session") -> list[dict]:
    return [
        {
            "content": proposal.content,
            "kind": proposal.kind,
            "scope_level": proposal.scope_level,
            "confidence": proposal.confidence,
            "admissible": proposal.admissible,
            "rejection_reason": proposal.rejection_reason,
            "source": proposal.source,
        }
        for proposal in extract_write_proposals(text, scope_level=scope_level)
    ]
