"""
Citation detection for MNEMOSYNE-MNS shadow mode.

Detects when the model's response references an injected memory.
Uses heuristics: exact substring match, key phrase overlap, and explicit references.
"""

from __future__ import annotations

import re
from typing import Any


def detect_citations_in_response(
    response: str,
    injected_memories: list[dict[str, Any]],
    min_overlap_words: int = 3,
    min_overlap_ratio: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Detect which injected memories were cited in the model's response.
    
    Args:
        response: Model's response text
        injected_memories: List of dicts with keys: memory_id, content, rank
        min_overlap_words: Minimum overlapping words to consider a citation
        min_overlap_ratio: Minimum ratio of memory words present in response
    
    Returns:
        List of citation dicts with keys: memory_id, span, confidence, explicit_reference
    """
    citations = []
    response_lower = response.lower()
    
    # Explicit reference patterns
    explicit_patterns = [
        r"(?:根据|按照|依据|参考|如前所述|之前提到|上面说|刚才说)[\s\S]{0,30}",
        r"(?:according to|as mentioned|previously|earlier|as stated)[\s\S]{0,30}",
    ]
    
    for memory in injected_memories:
        memory_id = memory["memory_id"]
        content = memory["content"]
        content_lower = content.lower()
        
        # Method 1: Exact substring match (high confidence)
        if content_lower in response_lower:
            start = response_lower.index(content_lower)
            span = response[start : start + len(content)]
            citations.append({
                "memory_id": memory_id,
                "span": span,
                "confidence": 0.95,
                "explicit_reference": False,
                "method": "exact_match",
            })
            continue
        
        # Method 2: Key phrase overlap
        memory_words = set(re.findall(r'\w+', content_lower))
        # Filter out stopwords (simple Chinese + English)
        stopwords = {
            "的", "了", "在", "是", "和", "与", "或", "为", "等", "中", "用", "对", "与",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"
        }
        memory_words = {w for w in memory_words if len(w) > 1 and w not in stopwords}
        
        if not memory_words:
            continue
        
        response_words = set(re.findall(r'\w+', response_lower))
        overlap = memory_words & response_words
        
        if len(overlap) >= min_overlap_words or len(overlap) / len(memory_words) >= min_overlap_ratio:
            # Find approximate span in response
            span = extract_citation_span(response, overlap)
            
            # Check if near explicit reference
            explicit = False
            for pattern in explicit_patterns:
                if re.search(pattern, response_lower):
                    explicit = True
                    break
            
            confidence = min(0.9, 0.5 + 0.4 * (len(overlap) / len(memory_words)))
            
            citations.append({
                "memory_id": memory_id,
                "span": span,
                "confidence": confidence,
                "explicit_reference": explicit,
                "method": "key_phrase_overlap",
                "overlap_count": len(overlap),
            })
    
    # Deduplicate: if multiple memories cited same span, keep highest confidence
    citations = deduplicate_citations(citations)
    
    return citations


def extract_citation_span(response: str, overlap_words: set[str], max_span_len: int = 150) -> str:
    """
    Extract the approximate span in response where overlapping words appear.
    Returns the first contiguous region with the most overlap words.
    """
    response_lower = response.lower()
    words = re.findall(r'\w+', response_lower)
    
    if not words:
        return ""
    
    # Find window with most overlap
    best_start, best_end, best_count = 0, 0, 0
    
    for start in range(len(words)):
        count = 0
        for end in range(start, min(start + 30, len(words))):
            if words[end] in overlap_words:
                count += 1
        if count > best_count:
            best_count = count
            best_start = start
            best_end = min(start + 30, len(words))
    
    # Convert word indices back to character span
    # Simple approximation: find first and last overlap word positions
    span_words = words[best_start:best_end]
    pattern = r'\b' + r'\b.*?\b'.join(re.escape(w) for w in span_words[:5]) + r'\b'
    match = re.search(pattern, response, re.IGNORECASE)
    
    if match:
        span = match.group(0)
        if len(span) > max_span_len:
            span = span[:max_span_len] + "..."
        return span
    
    return ""


def deduplicate_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate citations that reference overlapping spans.
    Keep the citation with highest confidence.
    """
    if len(citations) <= 1:
        return citations
    
    # Sort by confidence descending
    citations = sorted(citations, key=lambda x: x["confidence"], reverse=True)
    
    result = []
    used_spans = set()
    
    for cit in citations:
        span_lower = cit["span"].lower()[:50]  # Use prefix for overlap check
        
        # Check if this span overlaps with already selected citations
        overlap = False
        for used in used_spans:
            if span_lower in used or used in span_lower:
                overlap = True
                break
        
        if not overlap:
            result.append(cit)
            used_spans.add(span_lower)
    
    return result


def detect_ignored_memories(
    injected_memories: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Identify memories that were injected but not cited.
    
    Returns:
        List of ignored memory dicts with keys: memory_id, rank, injected
    """
    cited_ids = {c["memory_id"] for c in citations}
    
    ignored = []
    for memory in injected_memories:
        if memory["memory_id"] not in cited_ids:
            ignored.append({
                "memory_id": memory["memory_id"],
                "rank": memory.get("rank", 0),
                "injected": True,
            })
    
    return ignored
