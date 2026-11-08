#!/usr/bin/env python3
"""
Fix data split - Stratified train/val/test split

Problem: Current val/holdout sets are 100% citation (positive only)
Solution: Re-split with stratification to maintain label distribution
"""
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

print("=" * 60)
print("Data Split Fix - Stratified Split")
print("=" * 60)

# Load all data
print("\n1. Loading all data...")
all_samples = []

train_path = Path("data/unified/train_v2.jsonl")
val_path = Path("data/unified/val_v2.jsonl")
holdout_path = Path("data/unified/holdout_v2.jsonl")

for path in [train_path, val_path, holdout_path]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                all_samples.append(json.loads(line))

print(f"   Total samples: {len(all_samples):,}")

# Check label distribution
labels = [s.get("label") for s in all_samples]
label_dist = Counter(labels)
print(f"\n   Original label distribution:")
for label, count in label_dist.most_common():
    print(f"     {label}: {count:,} ({count/len(labels)*100:.1f}%)")

# Convert labels for stratification
# citation/correction = positive (1), unknown = negative (0)
binary_labels = [
    1 if s.get("label") in ["citation", "correction"] else 0
    for s in all_samples
]

pos_rate = sum(binary_labels) / len(binary_labels)
print(f"\n   Positive rate: {pos_rate:.1%}")
print(f"   Negative rate: {1-pos_rate:.1%}")

# Stratified split: 70% train, 15% val, 15% test
print("\n2. Performing stratified split...")
print("   Target: 70% train, 15% val, 15% holdout")

# First split: 70% train, 30% temp
train_samples, temp_samples, train_labels, temp_labels = train_test_split(
    all_samples,
    binary_labels,
    test_size=0.3,
    stratify=binary_labels,
    random_state=42
)

# Second split: split temp into 50% val, 50% holdout (15% each of total)
val_samples, holdout_samples, val_labels, holdout_labels = train_test_split(
    temp_samples,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)

print(f"\n   Results:")
print(f"     Train:   {len(train_samples):,} samples")
print(f"     Val:     {len(val_samples):,} samples")
print(f"     Holdout: {len(holdout_samples):,} samples")

# Verify stratification
def check_distribution(samples, name):
    labels = [s.get("label") for s in samples]
    dist = Counter(labels)
    pos_count = sum(1 for l in labels if l in ["citation", "correction"])
    neg_count = len(labels) - pos_count
    
    print(f"\n   {name}:")
    for label, count in sorted(dist.items()):
        print(f"     {label:12s}: {count:>6,} ({count/len(labels)*100:>5.1f}%)")
    print(f"     {'Positive':12s}: {pos_count:>6,} ({pos_count/len(labels)*100:>5.1f}%)")
    print(f"     {'Negative':12s}: {neg_count:>6,} ({neg_count/len(labels)*100:>5.1f}%)")
    
    return pos_count, neg_count

print("\n3. Verifying stratification...")
train_pos, train_neg = check_distribution(train_samples, "Train")
val_pos, val_neg = check_distribution(val_samples, "Val")
holdout_pos, holdout_neg = check_distribution(holdout_samples, "Holdout")

# Check if stratification worked
train_pos_rate = train_pos / len(train_samples)
val_pos_rate = val_pos / len(val_samples)
holdout_pos_rate = holdout_pos / len(holdout_samples)

print(f"\n   Positive rate consistency:")
print(f"     Train:   {train_pos_rate:.2%}")
print(f"     Val:     {val_pos_rate:.2%}")
print(f"     Holdout: {holdout_pos_rate:.2%}")
print(f"     Max diff: {max(abs(train_pos_rate-val_pos_rate), abs(train_pos_rate-holdout_pos_rate))*100:.1f}pp")

if val_neg == 0 or holdout_neg == 0:
    print("\n❌ ERROR: Validation or holdout still has no negative samples!")
    print("   This should not happen with stratification.")
    exit(1)
else:
    print("\n✅ SUCCESS: All splits have both positive and negative samples!")

# Save fixed splits
print("\n4. Saving fixed splits...")
output_dir = Path("data/unified_fixed")
output_dir.mkdir(parents=True, exist_ok=True)

def save_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

save_jsonl(train_samples, output_dir / "train_v2_fixed.jsonl")
save_jsonl(val_samples, output_dir / "val_v2_fixed.jsonl")
save_jsonl(holdout_samples, output_dir / "holdout_v2_fixed.jsonl")

print(f"   ✓ Saved to {output_dir}/")

# Summary
print("\n" + "=" * 60)
print("Data Split Fix Complete")
print("=" * 60)
print("\nBefore (broken):")
print(f"  Train:   24,829 (80.3% pos, 19.7% neg)")
print(f"  Val:      7,094 (100% pos, 0% neg) ❌")
print(f"  Holdout:  3,548 (100% pos, 0% neg) ❌")

print("\nAfter (fixed):")
print(f"  Train:   {len(train_samples):>6,} ({train_pos_rate:.1%} pos, {1-train_pos_rate:.1%} neg) ✅")
print(f"  Val:     {len(val_samples):>6,} ({val_pos_rate:.1%} pos, {1-val_pos_rate:.1%} neg) ✅")
print(f"  Holdout: {len(holdout_samples):>6,} ({holdout_pos_rate:.1%} pos, {1-holdout_pos_rate:.1%} neg) ✅")

print("\n✓ All splits now have mixed labels")
print("✓ AUC computation will work")
print("✓ Model validation is now possible")
print("\nNext: Retrain V3 with fixed validation set")
print("=" * 60)
