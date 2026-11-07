#!/usr/bin/env python3
"""
Integrate shadow traces with unified v2 training data for P2 iteration.

Strategy: Use shadow traces as high-confidence validation set, keep unified v2 as training base.
"""
import json
from pathlib import Path
from collections import defaultdict

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def integrate_shadow_with_unified():
    """Integrate shadow traces into unified v2 dataset."""
    
    print("Loading shadow traces...")
    shadow_samples = load_jsonl("data/shadow_export/shadow_training_samples.jsonl")
    print(f"Shadow samples: {len(shadow_samples)}")
    
    # Filter shadow samples with positive labels only (citations)
    shadow_positive = [s for s in shadow_samples if s.get("label") == "positive"]
    print(f"Shadow positive (citations): {len(shadow_positive)}")
    
    print("\nLoading unified v2 data...")
    unified_train = load_jsonl("data/unified/train_v2.jsonl")
    unified_val = load_jsonl("data/unified/val_v2.jsonl")
    unified_holdout = load_jsonl("data/unified/holdout_v2.jsonl")
    
    print(f"Unified train: {len(unified_train)}")
    print(f"Unified val: {len(unified_val)}")
    print(f"Unified holdout: {len(unified_holdout)}")
    
    # Convert shadow samples to unified schema
    shadow_converted = []
    for s in shadow_positive:
        # Map shadow schema to unified schema
        unified_sample = {
            "query_id": s.get("retrieval_event_id", "shadow_" + str(s.get("trace_id", ""))),
            "query_text": s.get("query_text", ""),
            "memory_id": s.get("memory_id"),
            "memory_text": s.get("memory_text", ""),
            "label": "citation",  # Shadow positive = citation
            "source": "shadow_trace",
            "query_timestamp": s.get("retrieval_timestamp"),
            "memory_timestamp": s.get("memory_timestamp"),
        }
        shadow_converted.append(unified_sample)
    
    print(f"\nConverted shadow samples: {len(shadow_converted)}")
    
    # Strategy: Add shadow traces to validation set (high-confidence real data)
    # Keep training set from unified v2 (larger, more diverse)
    
    new_val = unified_val + shadow_converted
    print(f"\nNew validation set: {len(new_val)} (unified {len(unified_val)} + shadow {len(shadow_converted)})")
    
    # Save P2 iteration dataset
    output_dir = Path("data/p2_iteration")
    output_dir.mkdir(exist_ok=True)
    
    save_jsonl(unified_train, output_dir / "train.jsonl")
    save_jsonl(new_val, output_dir / "val.jsonl")
    save_jsonl(unified_holdout, output_dir / "holdout.jsonl")
    
    # Statistics
    train_pos = sum(1 for s in unified_train if s.get("memory_id"))
    val_pos = sum(1 for s in new_val if s.get("memory_id"))
    holdout_pos = sum(1 for s in unified_holdout if s.get("memory_id"))
    
    print("\n" + "="*60)
    print("P2 Iteration Dataset Statistics")
    print("="*60)
    print(f"Train:   {len(unified_train):6d} samples ({train_pos:5d} pos, {len(unified_train)-train_pos:5d} neg)")
    print(f"Val:     {len(new_val):6d} samples ({val_pos:5d} pos, {len(new_val)-val_pos:5d} neg)")
    print(f"         - Unified v2: {len(unified_val)} samples")
    print(f"         - Shadow traces: {len(shadow_converted)} samples (real citations)")
    print(f"Holdout: {len(unified_holdout):6d} samples ({holdout_pos:5d} pos, {len(unified_holdout)-holdout_pos:5d} neg)")
    print("="*60)
    
    # Manifest
    manifest = {
        "iteration": "P2_v1",
        "base_model": "v0.3.0-rank-enhanced-v2",
        "train_samples": len(unified_train),
        "val_samples": len(new_val),
        "holdout_samples": len(unified_holdout),
        "shadow_citations_added": len(shadow_converted),
        "strategy": "Shadow traces added to validation set for real-world evaluation",
    }
    
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ P2 iteration dataset saved to {output_dir}/")
    return True

if __name__ == "__main__":
    success = integrate_shadow_with_unified()
    exit(0 if success else 1)
