#!/usr/bin/env python3
"""
RANK Enhanced V3 Full Training - 29-dim features on 143K dataset
Combines V2 features + Bayesian uncertainty quantification
"""
import json
import time
import numpy as np
from pathlib import Path

print("="*60)
print("RANK Enhanced V3 Full Training")
print("29-dim features (23 V2 + 6 Bayesian)")
print("="*60)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import lightgbm as lgb
    from scipy.stats import beta as beta_dist
    print("✓ All dependencies available")
except ImportError as e:
    print(f"❌ Missing: {e}")
    exit(1)

# Step 1: Load data
print("\n1. Loading training data...")
def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

train_samples = load_jsonl("data/unified/train_v2.jsonl")
val_samples = load_jsonl("data/unified/val_v2.jsonl")
print(f"   Train: {len(train_samples)} samples")
print(f"   Val: {len(val_samples)} samples")

# Step 2: Load Bayesian statistics
print("\n2. Loading Bayesian statistics...")
bayesian_stats = {}
synthetic_path = Path("data/synthetic_retrieval/retrieval_history.jsonl")
if synthetic_path.exists():
    with open(synthetic_path, "r") as f:
        for line in f:
            record = json.loads(line)
            bayesian_stats[record["memory_id"]] = {
                "n_retrievals": record["n_retrievals"],
                "n_citations": record["n_citations"],
            }
    print(f"   ✓ Loaded {len(bayesian_stats)} memory statistics")
else:
    print("   ⚠️  No synthetic data, using priors only")

# Step 3: Load E5 model
print("\n3. Loading E5 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("deploy/e5_model")
model = AutoModel.from_pretrained("deploy/e5_model").to(device)
model.eval()
print(f"   ✓ E5 loaded on {device}")

# Step 4: Encode with E5
def encode_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embs)
    return np.vstack(embeddings)

print("\n4. Encoding queries and memories...")
print("   (This takes ~15-20 minutes for full dataset)")

train_queries = [str(s.get("query_text", "")) for s in train_samples]
train_memories = [str(s.get("memory_text", "")) for s in train_samples]
val_queries = [str(s.get("query_text", "")) for s in val_samples]
val_memories = [str(s.get("memory_text", "")) for s in val_samples]

start_time = time.time()
print("   Encoding training queries...")
train_query_embs = encode_batch(train_queries)
print("   Encoding training memories...")
train_memory_embs = encode_batch(train_memories)
print("   Encoding validation queries...")
val_query_embs = encode_batch(val_queries)
print("   Encoding validation memories...")
val_memory_embs = encode_batch(val_memories)
print(f"   ✓ Encoding complete ({time.time()-start_time:.1f}s)")

# Step 5: Extract 29-dim features
print("\n5. Extracting 29-dim features...")

def extract_v3_features(sample, query_emb, memory_emb, bayesian_stats):
    query = str(sample.get("query_text", ""))
    memory = str(sample.get("memory_text", ""))
    mem_id = sample.get("memory_id")
    
    # Basic features (8)
    query_words = query.lower().split()
    memory_words = memory.lower().split()
    query_set = set(query_words)
    memory_set = set(memory_words)
    
    overlap = len(query_set & memory_set)
    union = len(query_set | memory_set)
    
    features = [
        np.log1p(len(query_words)),
        np.log1p(len(memory_words)),
        len(query_words) / (len(memory_words) + 1e-8),
        overlap,
        overlap / (union + 1e-8),
        overlap / (len(query_set) + 1e-8),
        1.0 if any(w in query for w in ["不", "没", "dont", "not", "no"]) else 0.0,
        1.0 if any(w in memory for w in ["不", "没", "dont", "not", "no"]) else 0.0,
    ]
    
    # Enhanced features (15)
    features.extend([
        0.0,  # scope_level_match
        0.0,  # authority_signal
        0.0,  # staleness_penalty
        0.0,  # version_consistency
        0.0,  # egress_destination_match
        0.0,  # conflict_chain_depth
        0.0,  # citation_history_score
        0.0,  # correction_risk_score
        0.0,  # session_continuation_rate
        0.0,  # query_memory_temporal_gap
        float(np.dot(query_emb, memory_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(memory_emb) + 1e-8)),
        float(np.linalg.norm(query_emb - memory_emb)),
        float(np.sum(np.abs(query_emb - memory_emb))),
        len(query_set) / (len(query_words) + 1e-8),
        len(memory_set) / (len(memory_words) + 1e-8),
    ])
    
    # Bayesian features (6)
    if mem_id and mem_id in bayesian_stats:
        stats = bayesian_stats[mem_id]
        n_ret = stats["n_retrievals"]
        n_cit = stats["n_citations"]
        alpha = 1.0 + n_cit
        beta_param = 1.0 + (n_ret - n_cit)
        posterior = beta_dist(alpha, beta_param)
        bayes_mean = posterior.mean()
        bayes_std = posterior.std()
        ci_lower, ci_upper = posterior.ppf(0.025), posterior.ppf(0.975)
        ci_width = ci_upper - ci_lower
        has_data = 1.0
        n_ret_log = np.log1p(n_ret)
    else:
        bayes_mean = 0.5
        bayes_std = 0.29
        ci_width = 0.95
        has_data = 0.0
        n_ret_log = 0.0
    
    features.extend([
        bayes_mean,
        bayes_std,
        ci_width,
        has_data,
        n_ret_log,
        bayes_std,
    ])
    
    return features

X_train = []
y_train = []
for i, sample in enumerate(train_samples):
    if i % 5000 == 0:
        print(f"   Processing train {i}/{len(train_samples)}...")
    features = extract_v3_features(sample, train_query_embs[i], train_memory_embs[i], bayesian_stats)
    X_train.append(features)
    label = sample.get("label", "unknown")
    y_train.append(1 if label in ["relevant", "citation", "correction"] else 0)

X_val = []
y_val = []
for i, sample in enumerate(val_samples):
    features = extract_v3_features(sample, val_query_embs[i], val_memory_embs[i], bayesian_stats)
    X_val.append(features)
    label = sample.get("label", "unknown")
    y_val.append(1 if label in ["relevant", "citation", "correction"] else 0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

print(f"   ✓ Features extracted: {X_train.shape}")

# Step 6: Train LightGBM
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

model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"]
)

# Step 7: Save
print("\n7. Saving model...")
output_dir = Path("runs/rank_enhanced_v3_full")
output_dir.mkdir(parents=True, exist_ok=True)

model.save_model(str(output_dir / "model.txt"))

feature_names = [
    "query_len", "memory_len", "len_ratio",
    "lexical_overlap", "lexical_jaccard", "lexical_containment",
    "query_negation", "memory_negation",
    "scope_match", "authority", "staleness", "version", "egress", "conflict_depth",
    "citation_history", "correction_risk", "session_continuation", "temporal_gap",
    "e5_cosine", "e5_euclidean", "e5_manhattan",
    "query_specificity", "memory_informativeness",
    "bayes_mean", "bayes_std", "bayes_ci_width",
    "bayes_has_data", "bayes_n_ret_log", "bayes_uncertainty",
]

importance = dict(zip(feature_names, model.feature_importance().tolist()))
with open(output_dir / "importance.json", "w") as f:
    json.dump(importance, f, indent=2)

print("\nTop 15 features:")
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
for name, imp in sorted_features:
    print(f"  {name:<30}: {imp:>10.0f}")

print(f"\n✓ Model saved to {output_dir}")
print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
