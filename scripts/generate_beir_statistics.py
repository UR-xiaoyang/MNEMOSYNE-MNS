#!/usr/bin/env python3
"""
Generate realistic retrieval statistics from existing BEIR data
BEIR datasets have query-document relevance, we can simulate retrieval patterns
"""
import json
from pathlib import Path
from collections import defaultdict
import random

print("=" * 60)
print("Generate Realistic Bayesian Statistics from BEIR")
print("=" * 60)

# Use our existing unified data
unified_dir = Path("data/unified")
if not (unified_dir / "train_v2.jsonl").exists():
    print("❌ Unified training data not found")
    print("   Expected: data/unified/train_v2.jsonl")
    exit(1)

print("\n1. Loading existing training data...")
samples = []
with open(unified_dir / "train_v2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

print(f"   ✓ Loaded {len(samples):,} samples")

# Build memory-level statistics
print("\n2. Building memory-level statistics...")
# Track per-query retrieval events
memory_stats = defaultdict(lambda: {
    "retrievals": [],  # List of (query_id, was_cited)
    "source": "unknown"
})

for sample in samples:
    memory_id = sample.get("memory_id")
    if not memory_id:
        continue
    
    query_id = sample.get("query_id", "unknown")
    label = sample.get("label", "unknown")
    
    # Each sample represents one retrieval event
    was_cited = label in ["citation", "correction"]
    memory_stats[memory_id]["retrievals"].append((query_id, was_cited))
    
    # Track source
    if "source" in sample:
        memory_stats[memory_id]["source"] = sample["source"]

print(f"   ✓ Analyzed {len(memory_stats):,} unique memories")

# Convert to retrieval statistics
print("\n3. Converting to retrieval/citation statistics...")
bayesian_stats = []

for memory_id, stats in memory_stats.items():
    n_retrievals = len(stats["retrievals"])
    n_citations = sum(1 for _, was_cited in stats["retrievals"] if was_cited)
    
    # Only include memories with meaningful data
    if n_retrievals >= 1:
        bayesian_stats.append({
            "memory_id": memory_id,
            "n_retrievals": n_retrievals,
            "n_citations": n_citations,
            "source": stats["source"]
        })

# Sort by retrieval count (most retrieved first)
bayesian_stats.sort(key=lambda x: x["n_retrievals"], reverse=True)

print(f"   ✓ Generated {len(bayesian_stats):,} memory statistics")

# Distribution analysis
print("\n4. Distribution analysis...")
retrieval_dist = defaultdict(int)
for stat in bayesian_stats:
    retrieval_dist[stat["n_retrievals"]] += 1

print("   Retrieval frequency:")
for threshold in [1, 2, 5, 10, 20]:
    count = sum(c for r, c in retrieval_dist.items() if r >= threshold)
    print(f"     >= {threshold:2d} retrievals: {count:>6,} memories")

# Citation rate analysis
citation_rates = []
for stat in bayesian_stats:
    if stat["n_retrievals"] > 0:
        rate = stat["n_citations"] / stat["n_retrievals"]
        citation_rates.append(rate)

avg_citation_rate = sum(citation_rates) / len(citation_rates)
print(f"\n   Average citation rate: {avg_citation_rate:.2%}")

# Export
output_path = Path("data/msmarco/bayesian_statistics_from_beir.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for record in bayesian_stats:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n✓ Exported to {output_path}")

# Summary
total_retrievals = sum(s["n_retrievals"] for s in bayesian_stats)
total_citations = sum(s["n_citations"] for s in bayesian_stats)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Unique memories:    {len(bayesian_stats):>10,}")
print(f"Total retrievals:   {total_retrievals:>10,}")
print(f"Total citations:    {total_citations:>10,}")
print(f"Avg citation rate:  {avg_citation_rate:>10.2%}")
print(f"\nOutput: {output_path}")

# Compare with synthetic data
synthetic_path = Path("data/synthetic_retrieval/retrieval_history.jsonl")
if synthetic_path.exists():
    with open(synthetic_path, "r") as f:
        synthetic_count = sum(1 for _ in f)
    print(f"\nComparison:")
    print(f"  Synthetic data:  {synthetic_count:>6,} memories")
    print(f"  Real-based data: {len(bayesian_stats):>6,} memories")
    print(f"  Increase:        {len(bayesian_stats)/synthetic_count:>6.1f}x")

print("\n" + "=" * 60)
print("Next: Retrain V3 with real-based statistics")
print("=" * 60)
