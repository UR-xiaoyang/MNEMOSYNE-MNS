#!/usr/bin/env python3
"""
RANK V3 Final - Train on fixed data with proper validation

Uses stratified train/val split with correct label distribution
"""
import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import beta as beta_dist

print("=" * 60)
print("RANK V3 FINAL - Fixed Validation")
print("Stratified data split + Complete validation")
print("=" * 60)
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    print("✓ All dependencies available")
except ImportError as e:
    print(f"❌ Missing: {e}")
    exit(1)

# Load FIXED data
print("\n1. Loading FIXED data...")
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_samples = load_jsonl("data/unified_fixed/train_v2_fixed.jsonl")
val_samples = load_jsonl("data/unified_fixed/val_v2_fixed.jsonl")
print(f"   Train: {len(train_samples):,}")
print(f"   Val: {len(val_samples):,}")

# Check label distribution
from collections import Counter
train_labels = [s.get("label") for s in train_samples]
val_labels = [s.get("label") for s in val_samples]
print(f"\n   Train labels: {Counter(train_labels)}")
print(f"   Val labels: {Counter(val_labels)}")

# Load Bayesian statistics
print("\n2. Loading Bayesian statistics...")
bayesian_stats = {}
stats_path = Path("data/msmarco/bayesian_statistics_intelligent.jsonl")

if stats_path.exists():
    with open(stats_path, "r") as f:
        for line in f:
            record = json.loads(line)
            bayesian_stats[record["memory_id"]] = record
    print(f"   ✓ {len(bayesian_stats):,} memories")
else:
    print(f"   ⚠️  No Bayesian statistics found")

# Load E5
print("\n3. Loading E5 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("deploy/e5_model" if Path("deploy/e5_model").exists() else "intfloat/e5-base-v2")
model = AutoModel.from_pretrained("deploy/e5_model" if Path("deploy/e5_model").exists() else "intfloat/e5-base-v2").to(device)
model.eval()
print(f"   ✓ E5 on {device}")

# Encode
def encode_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

print("\n4. Encoding...")
train_q = [str(s.get("query_text", "")) for s in train_samples]
train_m = [str(s.get("memory_text", "")) for s in train_samples]
val_q = [str(s.get("query_text", "")) for s in val_samples]
val_m = [str(s.get("memory_text", "")) for s in val_samples]

t0 = time.time()
train_q_emb = encode_batch(train_q)
train_m_emb = encode_batch(train_m)
val_q_emb = encode_batch(val_q)
val_m_emb = encode_batch(val_m)
print(f"   ✓ Done ({time.time()-t0:.1f}s)")

# Extract 19-dim features
print("\n5. Extracting features...")

def extract_features(sample, q_emb, m_emb, bayesian_stats, n_train):
    query = str(sample.get("query_text", ""))
    memory = str(sample.get("memory_text", ""))
    mem_id = sample.get("memory_id")
    
    q_words = query.lower().split()
    m_words = memory.lower().split()
    q_set, m_set = set(q_words), set(m_words)
    overlap = len(q_set & m_set)
    union = len(q_set | m_set)
    
    features = [
        np.log1p(len(q_words)),
        np.log1p(len(m_words)),
        len(q_words) / (len(m_words) + 1e-8),
        overlap,
        overlap / (union + 1e-8),
        overlap / (len(q_set) + 1e-8),
        1.0 if any(w in query for w in ["不", "没", "dont", "not", "no"]) else 0.0,
        1.0 if any(w in memory for w in ["不", "没", "dont", "not", "no"]) else 0.0,
    ]
    
    features.extend([
        float(np.dot(q_emb, m_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(m_emb) + 1e-8)),
        float(np.linalg.norm(q_emb - m_emb)),
        float(np.sum(np.abs(q_emb - m_emb))),
    ])
    
    features.extend([
        len(q_set) / (len(q_words) + 1e-8),
        len(m_set) / (len(m_words) + 1e-8),
    ])
    
    if mem_id and mem_id in bayesian_stats:
        stat = bayesian_stats[mem_id]
        n_ret, n_cit = stat["n_retrievals"], stat["n_citations"]
        alpha, beta_p = 1.0 + n_cit, 1.0 + (n_ret - n_cit)
        post = beta_dist(alpha, beta_p)
        mean, std = post.mean(), post.std()
        ci_l, ci_h = post.ppf(0.025), post.ppf(0.975)
        ci_w = ci_h - ci_l
        coverage = n_ret / n_train
        consistency = 1.0 / (1.0 + np.log1p(n_ret))
        confidence = 1.0 - ci_w
    else:
        mean, std, ci_w = 0.5, 0.29, 0.95
        coverage, consistency, confidence = 0.0, 0.0, 0.05
    
    features.extend([mean, std, ci_w, coverage, consistency, confidence])
    
    return features

X_train, y_train = [], []
for i, s in enumerate(train_samples):
    if i % 5000 == 0:
        print(f"   Train {i:,}/{len(train_samples):,}")
    X_train.append(extract_features(s, train_q_emb[i], train_m_emb[i], bayesian_stats, len(train_samples)))
    y_train.append(1 if s.get("label") in ["citation", "correction"] else 0)

X_val, y_val = [], []
for i, s in enumerate(val_samples):
    X_val.append(extract_features(s, val_q_emb[i], val_m_emb[i], bayesian_stats, len(train_samples)))
    y_val.append(1 if s.get("label") in ["citation", "correction"] else 0)

X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train)
X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val)

print(f"   ✓ {X_train.shape}")
print(f"   Train pos/neg: {sum(y_train)}/{len(y_train)-sum(y_train)}")
print(f"   Val pos/neg: {sum(y_val)}/{len(y_val)-sum(y_val)}")

# Train
print("\n6. Training LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 1,
}

model = lgb.train(params, train_data, num_boost_round=200,
                  valid_sets=[train_data, val_data],
                  valid_names=["train", "valid"])

# Evaluate
print("\n7. Evaluating...")
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_auc = roc_auc_score(y_train, train_pred)
val_auc = roc_auc_score(y_val, val_pred)
val_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
val_f1 = f1_score(y_val, (val_pred >= 0.5).astype(int))

print(f"\n   Performance:")
print(f"     Train AUC: {train_auc:.4f}")
print(f"     Val AUC:   {val_auc:.4f} ✅")
print(f"     Val Acc:   {val_acc:.4f}")
print(f"     Val F1:    {val_f1:.4f}")

# Save
print("\n8. Saving...")
out_dir = Path("runs/rank_v3_final")
out_dir.mkdir(parents=True, exist_ok=True)
model.save_model(str(out_dir / "model.txt"))

names = [
    "query_len", "memory_len", "len_ratio",
    "lexical_overlap", "lexical_jaccard", "lexical_containment",
    "query_negation", "memory_negation",
    "e5_cosine", "e5_euclidean", "e5_manhattan",
    "query_specificity", "memory_informativeness",
    "bayes_mean", "bayes_std", "bayes_ci_width",
    "bayes_coverage", "bayes_consistency", "bayes_confidence",
]

importance = dict(zip(names, model.feature_importance().tolist()))
with open(out_dir / "importance.json", "w") as f:
    json.dump(importance, f, indent=2)

metadata = {
    "train_samples": len(train_samples),
    "val_samples": len(val_samples),
    "train_auc": float(train_auc),
    "val_auc": float(val_auc),
    "val_accuracy": float(val_acc),
    "val_f1": float(val_f1),
    "bayesian_stats_count": len(bayesian_stats),
}

with open(out_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTop 10 features:")
for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:<25}: {imp:>8.0f}")

print(f"\n✓ Saved to {out_dir}")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print("VALIDATION NOW WORKS! Val AUC successfully computed!")
print("=" * 60)
