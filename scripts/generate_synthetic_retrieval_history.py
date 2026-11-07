#!/usr/bin/env python3
"""
Generate synthetic retrieval history for Bayesian uncertainty modeling.

Problem: Shadow traces lack repeated retrievals (0 memories with ≥5 retrievals)
Solution: Generate realistic synthetic retrieval histories based on:
1. Unified v2 dataset (35K query-memory pairs with quality labels)
2. Realistic retrieval patterns (Zipf distribution)
3. Citation probability based on memory quality

Output: Enhanced shadow traces with retrieval counts for Bayesian modeling
"""
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def load_unified_dataset(path: str) -> List[dict]:
    """Load unified v2 training data."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def simulate_retrieval_history(
    memories: List[dict],
    n_queries: int = 5000,
    alpha: float = 1.5
) -> Dict[str, Dict]:
    """
    Simulate realistic retrieval history using Zipf distribution.
    
    Args:
        memories: List of memory records with quality labels
        n_queries: Total number of simulated queries
        alpha: Zipf exponent (higher = more concentrated on top memories)
    
    Returns:
        Dict[memory_id] = {n_retrievals, n_citations, quality_label}
    """
    
    # Extract unique memories
    unique_memories = {}
    for mem in memories:
        mem_id = mem.get('memory_id')
        if mem_id and mem_id not in unique_memories:
            # Determine ground truth quality
            label = mem.get('label', 'unknown')
            quality_score = {
                'citation': 0.8,      # High quality
                'correction': 0.7,    # Medium-high
                'relevant': 0.6,      # Medium
                'unknown': 0.3        # Low quality
            }.get(label, 0.3)
            
            unique_memories[mem_id] = {
                'memory_text': mem.get('memory_text', ''),
                'quality_label': label,
                'quality_score': quality_score,
                'n_retrievals': 0,
                'n_citations': 0,
            }
    
    memory_ids = list(unique_memories.keys())
    n_memories = len(memory_ids)
    
    print(f"Simulating {n_queries} queries over {n_memories} unique memories...")
    
    # Generate Zipf-distributed retrieval frequencies
    # Top memories get retrieved more often (realistic pattern)
    ranks = np.arange(1, n_memories + 1)
    probabilities = 1 / (ranks ** alpha)
    probabilities /= probabilities.sum()
    
    # Simulate queries
    for _ in range(n_queries):
        # Select memory according to Zipf distribution
        selected_idx = np.random.choice(n_memories, p=probabilities)
        mem_id = memory_ids[selected_idx]
        
        # Increment retrieval count
        unique_memories[mem_id]['n_retrievals'] += 1
        
        # Simulate citation decision (based on quality score)
        quality = unique_memories[mem_id]['quality_score']
        if random.random() < quality:
            unique_memories[mem_id]['n_citations'] += 1
    
    # Filter memories with at least 1 retrieval
    retrieved_memories = {
        mem_id: stats for mem_id, stats in unique_memories.items()
        if stats['n_retrievals'] > 0
    }
    
    return retrieved_memories

def export_synthetic_shadow_traces(
    retrieval_history: Dict[str, Dict],
    output_path: str
):
    """Export synthetic retrieval history in shadow trace format."""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for mem_id, stats in retrieval_history.items():
            # Create one record per memory with aggregated statistics
            memory_text = str(stats.get('memory_text') or '')
            truncated_text = memory_text[:100] + '...' if len(memory_text) > 100 else memory_text
            
            record = {
                'memory_id': mem_id,
                'memory_text': truncated_text,
                'n_retrievals': stats['n_retrievals'],
                'n_citations': stats['n_citations'],
                'quality_label': stats['quality_label'],
                'quality_score': stats['quality_score'],
                'citation_rate': stats['n_citations'] / stats['n_retrievals'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✓ Exported {len(retrieval_history)} memories to {output_file}")

def analyze_distribution(retrieval_history: Dict[str, Dict]):
    """Analyze synthetic retrieval distribution."""
    
    retrievals = [stats['n_retrievals'] for stats in retrieval_history.values()]
    citations = [stats['n_citations'] for stats in retrieval_history.values()]
    
    print("\n" + "="*60)
    print("Synthetic Retrieval History Statistics")
    print("="*60)
    print(f"Total memories: {len(retrieval_history)}")
    print(f"Total retrievals: {sum(retrievals)}")
    print(f"Total citations: {sum(citations)}")
    print(f"Global citation rate: {sum(citations)/sum(retrievals):.2%}")
    print()
    print("Retrieval count distribution:")
    print(f"  Min: {min(retrievals)}")
    print(f"  Median: {np.median(retrievals):.1f}")
    print(f"  Mean: {np.mean(retrievals):.1f}")
    print(f"  Max: {max(retrievals)}")
    print()
    
    # Distribution by bucket
    buckets = [
        (1, 4, "1-4 retrievals"),
        (5, 9, "5-9 retrievals"),
        (10, 19, "10-19 retrievals"),
        (20, 49, "20-49 retrievals"),
        (50, float('inf'), "50+ retrievals"),
    ]
    
    print("Distribution by retrieval count:")
    for low, high, label in buckets:
        count = sum(1 for r in retrievals if low <= r <= high)
        percentage = count / len(retrievals) * 100
        print(f"  {label:<20}: {count:>6} ({percentage:>5.1f}%)")
    
    print("\n" + "="*60)

def run_generation(
    input_path: str = "data/unified/train_v2.jsonl",
    output_path: str = "data/synthetic_retrieval/retrieval_history.jsonl",
    n_queries: int = 10000,
    alpha: float = 1.5
):
    """Run synthetic retrieval history generation."""
    
    print("="*60)
    print("Synthetic Retrieval History Generator")
    print("="*60)
    
    # Load source data
    print(f"\n1. Loading unified dataset from {input_path}...")
    memories = load_unified_dataset(input_path)
    print(f"   ✓ Loaded {len(memories)} query-memory pairs")
    
    # Simulate retrieval history
    print(f"\n2. Simulating {n_queries} queries (Zipf α={alpha})...")
    retrieval_history = simulate_retrieval_history(memories, n_queries, alpha)
    
    # Analyze distribution
    analyze_distribution(retrieval_history)
    
    # Export
    print(f"\n3. Exporting to {output_path}...")
    export_synthetic_shadow_traces(retrieval_history, output_path)
    
    # Statistics for Bayesian modeling
    qualified_memories = sum(
        1 for stats in retrieval_history.values()
        if stats['n_retrievals'] >= 5
    )
    print(f"\n✓ {qualified_memories} memories with ≥5 retrievals (suitable for Bayesian modeling)")
    
    return retrieval_history

if __name__ == "__main__":
    import sys
    
    # Check if unified dataset exists
    input_path = "data/unified/train_v2.jsonl"
    if not Path(input_path).exists():
        print(f"❌ Unified dataset not found: {input_path}")
        print("\nPlease ensure unified v2 dataset exists:")
        print("  python scripts/integrate_p2_data.py")
        sys.exit(1)
    
    # Parse arguments
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    
    # Run generation
    retrieval_history = run_generation(
        input_path=input_path,
        n_queries=n_queries,
        alpha=alpha
    )
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("1. Run Bayesian experiment with synthetic data:")
    print("   python experiments/bayesian_uncertainty_modeling.py --synthetic")
    print()
    print("2. Adjust simulation parameters:")
    print(f"   python {sys.argv[0]} <n_queries> <alpha>")
    print("   - n_queries: more queries = higher retrieval counts")
    print("   - alpha: higher = more concentrated (e.g., 2.0 for heavy-tail)")
