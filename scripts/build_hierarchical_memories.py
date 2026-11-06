#!/usr/bin/env python3
"""
Generate hierarchical memory representations (L1/L2/L3) for NCM-style retrieval.

L3 (Full): Original memory content
L2 (Summary): Extractive/abstractive summary (50-100 tokens)
L1 (Keywords): Key phrases extracted (5-10 tokens)
"""
import json
from pathlib import Path
from typing import List, Dict
import re

def extract_keywords(text: str, max_keywords: int = 5) -> str:
    """
    Simple keyword extraction (TF-IDF would be better, but this is a baseline).
    
    Args:
        text: Full memory content
        max_keywords: Maximum number of keywords to extract
    
    Returns:
        Comma-separated keywords
    """
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stopwords and count frequency
    word_freq = {}
    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    
    return ', '.join([kw for kw, _ in top_keywords])

def generate_extractive_summary(text: str, max_sentences: int = 3) -> str:
    """
    Simple extractive summary: select first N sentences.
    
    In production, use:
    - BART/T5 for abstractive summarization
    - TextRank for better extractive summarization
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    summary_sentences = sentences[:min(max_sentences, len(sentences))]
    return '. '.join(summary_sentences) + '.'

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)."""
    return len(text) // 4

def build_hierarchical_memories(input_file: str, output_file: str):
    """
    Convert flat memory dataset to hierarchical (L1/L2/L3) format.
    
    Args:
        input_file: Path to original memory JSONL
        output_file: Path to hierarchical memory JSONL
    """
    print(f"Processing {input_file}...")
    
    hierarchical_memories = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} memories...")
            
            memory = json.loads(line)
            
            # Extract original content
            full_content = memory.get('memory_text', memory.get('content', ''))
            
            if not full_content:
                continue  # Skip empty memories
            
            # Generate hierarchical representations
            l3_full = full_content
            l2_summary = generate_extractive_summary(full_content, max_sentences=3)
            l1_keywords = extract_keywords(full_content, max_keywords=5)
            
            # Build hierarchical memory
            hierarchical_memory = {
                **memory,  # Keep original fields
                'content_l3_full': l3_full,
                'content_l2_summary': l2_summary,
                'content_l1_keywords': l1_keywords,
                'token_estimates': {
                    'l3': estimate_tokens(l3_full),
                    'l2': estimate_tokens(l2_summary),
                    'l1': estimate_tokens(l1_keywords)
                }
            }
            
            hierarchical_memories.append(hierarchical_memory)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for mem in hierarchical_memories:
            f.write(json.dumps(mem, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved {len(hierarchical_memories)} hierarchical memories to {output_file}")
    
    # Statistics
    total_l3 = sum(m['token_estimates']['l3'] for m in hierarchical_memories)
    total_l2 = sum(m['token_estimates']['l2'] for m in hierarchical_memories)
    total_l1 = sum(m['token_estimates']['l1'] for m in hierarchical_memories)
    
    print(f"\nToken Statistics:")
    print(f"  L3 (Full):    {total_l3:,} tokens ({total_l3 / len(hierarchical_memories):.1f} avg)")
    print(f"  L2 (Summary): {total_l2:,} tokens ({total_l2 / len(hierarchical_memories):.1f} avg)")
    print(f"  L1 (Keywords): {total_l1:,} tokens ({total_l1 / len(hierarchical_memories):.1f} avg)")
    print(f"  Compression: L2 = {total_l2/total_l3*100:.1f}% of L3, L1 = {total_l1/total_l3*100:.1f}% of L3")

if __name__ == "__main__":
    # Example usage
    build_hierarchical_memories(
        input_file="data/unified/train_v2.jsonl",
        output_file="data/hierarchical/train_l123.jsonl"
    )
    
    print("\n✓ Hierarchical memory generation complete!")
    print("\nNext steps:")
    print("1. Train RANK model on hierarchical data")
    print("2. Implement BudgetManager expert")
    print("3. A/B test: hierarchical vs flat memory")
