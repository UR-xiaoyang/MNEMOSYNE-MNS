#!/usr/bin/env python3
"""
Use Hugging Face Datasets to load MS MARCO (more reliable than direct download)
"""
import json
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("MS MARCO via Hugging Face Datasets")
print("=" * 60)

# Check if datasets library is available
try:
    from datasets import load_dataset
    print("✓ datasets library available")
except ImportError:
    print("❌ datasets library not found")
    print("\nInstall with: pip install datasets")
    print("\nAlternatively, we can use a smaller pre-processed sample")
    exit(1)

# Output
data_dir = Path("data/msmarco")
data_dir.mkdir(parents=True, exist_ok=True)

print("\n1. Loading MS MARCO passage ranking dataset...")
print("   (First run: downloads ~2GB, cached afterwards)")

# Load only the qrels (query-passage relevance judgments)
# This is much smaller than the full corpus
dataset = load_dataset("ms_marco", "v1.1", split="train[:100000]")  # Limit to 100K for testing

print(f"   ✓ Loaded {len(dataset):,} query-passage pairs")

# Build passage statistics
print("\n2. Building passage retrieval statistics...")
passage_stats = defaultdict(lambda: {"n_retrievals": 0, "n_citations": 0})

for i, example in enumerate(dataset):
    if i % 10000 == 0 and i > 0:
        print(f"   Processed {i:,} / {len(dataset):,}")
    
    query_id = example.get("query_id", str(i))
    passages = example.get("passages", {})
    
    # MS MARCO format: passages = {"is_selected": [0,1,0,...], "passage_text": [...]}
    is_selected = passages.get("is_selected", [])
    
    for j, selected in enumerate(is_selected):
        passage_id = f"p_{query_id}_{j}"
        
        # This passage was retrieved for this query
        passage_stats[passage_id]["n_retrievals"] += 1
        
        # If selected (clicked), count as citation
        if selected == 1:
            passage_stats[passage_id]["n_citations"] += 1

print(f"\n   ✓ Processed {len(passage_stats):,} unique passages")

# Filter to passages with enough data
min_retrievals = 3  # Lower threshold since we're using a sample
filtered = {
    pid: stats for pid, stats in passage_stats.items()
    if stats["n_retrievals"] >= min_retrievals
}

print(f"   ✓ {len(filtered):,} passages with >={min_retrievals} retrievals")

# Export
output_path = data_dir / "bayesian_statistics_hf.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for memory_id, stats in filtered.items():
        record = {
            "memory_id": f"msmarco_{memory_id}",
            "n_retrievals": stats["n_retrievals"],
            "n_citations": stats["n_citations"],
            "source": "msmarco_hf",
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n✓ Exported {len(filtered):,} records to {output_path}")

# Summary
total_retrievals = sum(s["n_retrievals"] for s in filtered.values())
total_citations = sum(s["n_citations"] for s in filtered.values())
citation_rate = total_citations / total_retrievals if total_retrievals > 0 else 0

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Passages:       {len(filtered):>10,}")
print(f"Retrievals:     {total_retrievals:>10,}")
print(f"Citations:      {total_citations:>10,}")
print(f"Citation rate:  {citation_rate:>10.2%}")
print(f"\nOutput: {output_path}")
