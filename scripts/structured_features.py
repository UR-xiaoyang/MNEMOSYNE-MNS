"""Deterministic structured features used by the trained hybrid ranker."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}|[\u4e00-\u9fff]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"\b[\w./:-]{2,}\b", re.UNICODE)
NEGATION_RE = re.compile(
    r"(?i)(不要|别再|不能|禁止|不采用|取消|否决|失败|错误|避免|不再|do not|don't|never|avoid|reject|cancel|failed|error)"
)
UPDATE_RE = re.compile(
    r"(?i)(改成|改为|换成|替换|以后|从现在开始|最终决定|更新|变更|replace|change to|switch to|from now on|update|supersede)"
)
REFERENCE_RE = re.compile(
    r"(?i)(继续|接着|上次|之前|刚才|前面|那个|再改|再做|回到|按照.*方案|根据.*方案|历史|上回|我们讨论|刚刚|前文|continue|previous|earlier|last time|that plan|as before|resume)"
)

FEATURE_NAMES = [
    "tfidf_cosine",
    "lexical_jaccard",
    "lexical_containment",
    "query_len_log",
    "memory_len_log",
    "query_has_reference",
    "query_has_negation",
    "memory_has_negation",
    "memory_has_update",
    "same_source_public",
    "weak_proxy",
    "sample_weight",
]


def _tokens(text: str) -> set[str]:
    raw = TOKEN_RE.findall(text.lower())
    result = {item for item in raw if len(item) > 1}
    for sequence in CJK_RE.findall(text):
        result.update(sequence[i : i + 2] for i in range(max(0, len(sequence) - 1)))
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _containment(left: set[str], right: set[str]) -> float:
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def _char_wb_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> dict[str, int]:
    """Approximate the training vectorizer's char_wb representation.

    The portable runtime uses this only as a deterministic fallback when the
    fitted sklearn vectorizer is not supplied. The production path should
    pass the fitted vectorizer to ``build_structured_features``.
    """
    padded = f" {text.lower()} "
    counts: dict[str, int] = {}
    for n in range(min_n, max_n + 1):
        for start in range(0, len(padded) - n + 1):
            gram = padded[start : start + n]
            if " " not in gram[1:-1]:
                counts[gram] = counts.get(gram, 0) + 1
    return counts


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _fallback_cosine(query: str, memory: str) -> float:
    q_counts = _char_wb_ngrams(query)
    m_counts = _char_wb_ngrams(memory)
    vocabulary = set(q_counts) | set(m_counts)
    if not vocabulary:
        return 0.0
    q = np.asarray([q_counts.get(key, 0.0) for key in vocabulary], dtype=np.float32)
    m = np.asarray([m_counts.get(key, 0.0) for key in vocabulary], dtype=np.float32)
    return _cosine(q, m)


def build_structured_features(
    queries: Sequence[str],
    memories: Sequence[str],
    *,
    vectorizer=None,
    same_source_public: Sequence[float] | None = None,
    weak_proxy: Sequence[float] | None = None,
    sample_weight: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the exact 12-column structured feature contract.

    ``vectorizer`` should be the fitted ``TfidfVectorizer`` from training. If
    omitted, a deterministic local character n-gram cosine is used so the API
    remains usable without sklearn artifacts; the learned ranker still expects
    the same column order.
    """
    if len(queries) != len(memories):
        raise ValueError("queries and memories must have equal lengths")
    n = len(queries)
    defaults = {
        "same_source_public": np.ones(n, dtype=np.float32),
        "weak_proxy": np.zeros(n, dtype=np.float32),
        "sample_weight": np.ones(n, dtype=np.float32),
    }
    supplied = {
        "same_source_public": same_source_public,
        "weak_proxy": weak_proxy,
        "sample_weight": sample_weight,
    }
    for name, values in supplied.items():
        if values is not None:
            if len(values) != n:
                raise ValueError(f"{name} must have length {n}")
            defaults[name] = np.asarray(values, dtype=np.float32)

    if vectorizer is not None and n:
        query_matrix = vectorizer.transform(list(queries))
        memory_matrix = vectorizer.transform(list(memories))
    else:
        query_matrix = memory_matrix = None

    rows: list[list[float]] = []
    for index, (query, memory) in enumerate(zip(queries, memories)):
        query_tokens = _tokens(query)
        memory_tokens = _tokens(memory)
        if vectorizer is not None:
            tfidf_cosine = _cosine(
                query_matrix[index].toarray().ravel(),
                memory_matrix[index].toarray().ravel(),
            )
        else:
            tfidf_cosine = _fallback_cosine(query, memory)
        rows.append(
            [
                tfidf_cosine,
                _jaccard(query_tokens, memory_tokens),
                _containment(query_tokens, memory_tokens),
                math.log1p(len(query)),
                math.log1p(len(memory)),
                float(bool(REFERENCE_RE.search(query))),
                float(bool(NEGATION_RE.search(query))),
                float(bool(NEGATION_RE.search(memory))),
                float(bool(UPDATE_RE.search(memory))),
                float(defaults["same_source_public"][index]),
                float(defaults["weak_proxy"][index]),
                float(defaults["sample_weight"][index]),
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(n, len(FEATURE_NAMES)), FEATURE_NAMES.copy()
