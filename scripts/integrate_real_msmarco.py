#!/usr/bin/env python3
"""
Real MS MARCO data integration via HuggingFace Datasets

Strategy: Use ms_marco passage ranking dataset directly
- Avoid network/SSL issues with HF mirror
- Extract passage-level click/relevance statistics
- Merge with existing Bayesian stats
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

print("=" * 60)
print("MS MARCO Real Data Integration")
print("Via HuggingFace Datasets (cached, reliable)")
print("=" * 60)

try:
    from datasets import load_dataset
    print("✓ datasets library available")
except ImportError:
    print("❌ datasets not installed")
    print("   Install: pip install datasets")
    exit(1)

# Load MS MARCO passage ranking (smaller, focused dataset)
print("\n1. Loading MS MARCO passage ranking...")
print("   (First run: downloads ~500MB, then cached)")
print("   Using 'train' split, limited to 50K for speed")

try:
    # Load passage ranking dataset (qrels format)
    dataset = load_dataset(
        "microsoft/ms_marco",
        "v2.1",
        split="train[:50000]",
        trust_remote_code=True
    )
    print(f"   ✓ Loaded {len(dataset):,} query-passage pairs")
except Exception as e:
    print(f"   ⚠️  Load failed: {e}")
    print("\n   Fallback: Use existing intelligent statistics")
    print("   MS MARCO optional for V3 - intelligent synthesis sufficient")
    exit(0)

# Extract passage statistics
print("\n2. Extracting passage retrieval statistics...")
passage_stats = defaultdict(lambda: {"retrievals": 0, "citations": 0, "queries": set()})

for i, example in enumerate(dataset):
    if i % 10000 == 0 and i > 0:
        print(f"   Processed {i:,} / {len(dataset):,}")
    
    # MS MARCO format varies by version
    # v2.1: {query_id, query, passages: {passage_text: [...], is_selected: [...]}}
    query_id = str(example.get("query_id", i))
    passages = example.get("passages", {})
    
    if isinstance(passages, dict):
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        for j, (text, selected) in enumerate(zip(passage_texts, is_selected)):
            # Use first 100 chars as passage ID (for matching)
            passage_id = f"msmarco_{hash(text[:100]) % 1000000}"
            
            passage_stats[passage_id]["retrievals"] += 1
            passage_stats[passage_id]["queries"].add(query_id)
            
            if selected == 1:
                passage_stats[passage_id]["citations"] += 1

print(f"\n   ✓ Processed {len(passage_stats):,} unique passages")

# Filter to high-quality passages
min_retrievals = 3
filtered = {
    pid: {
        "n_retrievals": stats["retrievals"],
        "n_citations": stats["citations"],
        "source": "msmarco_real"
    }
    for pid, stats in passage_stats.items()
    if stats["retrievals"] >= min_retrievals
}

print(f"   ✓ {len(filtered):,} passages with >={min_retrievals} retrievals")

# Analysis
print("\n3. Data quality analysis...")
total_ret = sum(s["n_retrievals"] for s in filtered.values())
total_cit = sum(s["n_citations"] for s in filtered.values())
cite_rate = total_cit / total_ret if total_ret > 0 else 0

print(f"   Total retrievals: {total_ret:,}")
print(f"   Total citations: {total_cit:,}")
print(f"   Citation rate: {cite_rate:.1%}")

# Retrieval distribution
ret_counts = [s["n_retrievals"] for s in filtered.values()]
print(f"\n   Retrieval distribution:")
for thresh in [3, 5, 10, 20]:
    count = sum(1 for r in ret_counts if r >= thresh)
    print(f"     >= {thresh:2d} retrievals: {count:>6,} ({count/len(filtered)*100:.1f}%)")

# Citation rate distribution
cite_rates = [s["n_citations"]/s["n_retrievals"] for s in filtered.values()]
print(f"\n   Citation rate distribution:")
for pct in [0.0, 0.2, 0.5, 0.8, 1.0]:
    count = sum(1 for r in cite_rates if r >= pct)
    print(f"     >= {pct:.0%}: {count:>6,} ({count/len(filtered)*100:.1f}%)")

# Export
output_path = Path("data/msmarco/bayesian_statistics_real_msmarco.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for memory_id, stats in filtered.items():
        record = {
            "memory_id": memory_id,
            "n_retrievals": stats["n_retrievals"],
            "n_citations": stats["n_citations"],
            "source": stats["source"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n✓ Exported to {output_path}")

# Merge with intelligent statistics
print("\n4. Merging with existing intelligent statistics...")
intelligent_path = Path("data/msmarco/bayesian_statistics_intelligent.jsonl")

if intelligent_path.exists():
    intelligent_stats = []
    with open(intelligent_path, "r") as f:
        for line in f:
            intelligent_stats.append(json.loads(line))
    
    # Combine (real MS MARCO takes precedence)
    merged = dict(filtered)  # Start with real data
    
    # Add intelligent data that's not in MS MARCO
    for record in intelligent_stats:
        mem_id = record["memory_id"]
        if mem_id not in merged:
            merged[mem_id] = {
                "n_retrievals": record["n_retrievals"],
                "n_citations": record["n_citations"],
                "source": "intelligent"
            }
    
    # Export merged
    merged_path = Path("data/msmarco/bayesian_statistics_merged.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for memory_id, stats in merged.items():
            record = {
                "memory_id": memory_id,
                "n_retrievals": stats["n_retrievals"],
                "n_citations": stats["n_citations"],
                "source": stats["source"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"   ✓ Merged: {len(merged):,} total memories")
    print(f"     - Real MS MARCO: {len(filtered):,}")
    print(f"     - Intelligent: {len(merged) - len(filtered):,}")
    print(f"   ✓ Exported to {merged_path}")

print("\n" + "=" * 60)
print("Success: Real MS MARCO data integrated")
print("Next: Retrain V3 with merged statistics")
print("=" * 60)
