#!/usr/bin/env python3
"""
Download and process MS MARCO click data for Bayesian statistics

MS MARCO provides query-passage pairs with click/relevance information
We'll extract passage-level retrieval and citation statistics
"""
import json
import gzip
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

print("=" * 60)
print("MS MARCO Click Data Processing")
print("=" * 60)

# MS MARCO URLs
URLS = {
    "qrels_train": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
    "qrels_dev": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv",
    "passages": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tsv",
}

# Output paths
data_dir = Path("data/msmarco")
data_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Download files
print("\n1. Downloading MS MARCO data...")
print("   (This may take 10-20 minutes for the full dataset)")

def download_file(url: str, output_path: Path):
    if output_path.exists():
        print(f"   ✓ {output_path.name} already exists, skipping")
        return
    
    print(f"   Downloading {output_path.name}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"   ✓ Downloaded to {output_path}")

# Download relevance judgments (query-passage pairs)
download_file(URLS["qrels_train"], data_dir / "qrels.train.tsv")
download_file(URLS["qrels_dev"], data_dir / "qrels.dev.tsv")

# Note: collection.tsv is 2.5GB, only download if needed
passages_path = data_dir / "collection.tsv"
if not passages_path.exists():
    print(f"\n   ⚠️  Passage collection (2.5GB) not found.")
    print(f"   To download, run:")
    print(f"   wget {URLS['passages']} -O {passages_path}")
    print(f"\n   For now, we'll work with passage IDs only.")

# Step 2: Build passage-level statistics
print("\n2. Building passage-level retrieval statistics...")

passage_stats = defaultdict(lambda: {"n_retrievals": 0, "n_citations": 0, "queries": []})

def process_qrels(qrels_path: Path):
    """Process qrels file: query_id \t 0 \t passage_id \t relevance"""
    with open(qrels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"      Processed {i:,} judgments...")
            
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            
            query_id, _, passage_id, relevance = parts
            relevance = int(relevance)
            
            # Count retrieval (any relevance judgment means it was retrieved)
            passage_stats[passage_id]["n_retrievals"] += 1
            passage_stats[passage_id]["queries"].append(query_id)
            
            # Count citation (relevance >= 1 means it was "cited"/clicked)
            if relevance >= 1:
                passage_stats[passage_id]["n_citations"] += 1

print("   Processing training qrels...")
process_qrels(data_dir / "qrels.train.tsv")

print("   Processing dev qrels...")
process_qrels(data_dir / "qrels.dev.tsv")

print(f"\n   ✓ Processed {len(passage_stats):,} unique passages")

# Step 3: Filter to high-quality passages
print("\n3. Filtering high-quality passages...")

# Keep passages with at least 5 retrievals (enough for Bayesian inference)
min_retrievals = 5
filtered_passages = {
    pid: stats for pid, stats in passage_stats.items()
    if stats["n_retrievals"] >= min_retrievals
}

print(f"   ✓ {len(filtered_passages):,} passages with >={min_retrievals} retrievals")

# Distribution analysis
retrieval_counts = Counter(stats["n_retrievals"] for stats in filtered_passages.values())
citation_counts = Counter(stats["n_citations"] for stats in filtered_passages.values())

print(f"\n   Retrieval distribution:")
for count in [5, 10, 20, 50, 100]:
    n = sum(c for r, c in retrieval_counts.items() if r >= count)
    print(f"     >= {count:3d} retrievals: {n:>8,} passages")

print(f"\n   Citation rate distribution:")
for pid, stats in list(filtered_passages.items())[:10]:
    rate = stats["n_citations"] / stats["n_retrievals"]
    print(f"     Passage {pid}: {stats['n_retrievals']} retrievals, {stats['n_citations']} citations ({rate:.1%})")

# Step 4: Export to MNEMOSYNE format
print("\n4. Exporting to MNEMOSYNE Bayesian format...")

output_path = data_dir / "bayesian_statistics.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for memory_id, stats in filtered_passages.items():
        record = {
            "memory_id": f"msmarco_{memory_id}",
            "n_retrievals": stats["n_retrievals"],
            "n_citations": stats["n_citations"],
            "source": "msmarco",
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"   ✓ Exported {len(filtered_passages):,} records to {output_path}")

# Step 5: Summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

total_retrievals = sum(s["n_retrievals"] for s in filtered_passages.values())
total_citations = sum(s["n_citations"] for s in filtered_passages.values())
overall_citation_rate = total_citations / total_retrievals

print(f"Total passages:        {len(filtered_passages):>10,}")
print(f"Total retrievals:      {total_retrievals:>10,}")
print(f"Total citations:       {total_citations:>10,}")
print(f"Overall citation rate: {overall_citation_rate:>10.2%}")
print(f"\nOutput: {output_path}")

print("\n" + "=" * 60)
print("Next Steps")
print("=" * 60)
print("1. Use this data to retrain RANK V3 with real statistics")
print("2. Compare feature importance with synthetic data baseline")
print("3. Expected: bayes_std importance should increase significantly")
print("=" * 60)
