#!/usr/bin/env python3
"""
RANK V3 MoE - FIXED VERSION with robust data loading

Fixes:
1. Multiple data path fallbacks
2. Mandatory validation (abort if no data)
3. Loud failures (raise, not print)
4. Pre-flight checks
"""
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("RANK V3 MoE - FIXED VERSION")
print("Robust data loading + validation")
print("=" * 60)
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import lightgbm as lgb
    from scipy.stats import beta as beta_dist
    from sklearn.linear_model import LogisticRegression
    print("✓ All dependencies available")
except ImportError as e:
    print(f"❌ Missing: {e}")
    exit(1)

# Load data
print("\n1. Loading training data...")
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_samples = load_jsonl("data/unified/train_v2.jsonl")
val_samples = load_jsonl("data/unified/val_v2.jsonl")
print(f"   Train: {len(train_samples):,}")
print(f"   Val: {len(val_samples):,}")

# Load Bayesian statistics - FIXED with multiple fallbacks
print("\n2. Loading Bayesian statistics (ROBUST)...")
bayesian_stats = {}

# Try multiple paths (absolute and relative)
stats_paths = [
    Path("data/msmarco/bayesian_statistics_merged.jsonl"),
    Path("data/msmarco/bayesian_statistics_intelligent.jsonl"),
    Path.home() / "MNEMOSYNE-MNS/data/msmarco/bayesian_statistics_intelligent.jsonl",
    Path.home() / "data/msmarco/bayesian_statistics_intelligent.jsonl",
    Path("/home/jx/MNEMOSYNE-MNS/data/msmarco/bayesian_statistics_intelligent.jsonl"),
]

print("   Attempting to load from:")
loaded = False
for path in stats_paths:
    print(f"     - {path}")
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                record = json.loads(line)
                bayesian_stats[record["memory_id"]] = record
        print(f"   ✅ SUCCESS: Loaded from {path}")
        loaded = True
        break

# MANDATORY VALIDATION - abort if no data
if not loaded or len(bayesian_stats) == 0:
    print("\n" + "=" * 60)
    print("❌ CRITICAL ERROR: No Bayesian statistics loaded!")
    print("=" * 60)
    print("\nMoE training REQUIRES Bayesian data for Expert 3.")
    print("Without it, Bayesian Expert is useless and MoE collapses.")
    print("\nExpected files:")
    for p in stats_paths:
        print(f"  - {p}")
    print("\nABORTING TRAINING.")
    print("=" * 60)
    exit(1)

# Additional validation
print(f"   ✓ Loaded {len(bayesian_stats):,} memories with statistics")

if len(bayesian_stats) < 1000:
    print(f"\n⚠️  WARNING: Only {len(bayesian_stats):,} statistics loaded!")
    print(f"   Recommended: 4,000+ for effective MoE training")
    print(f"   Proceeding anyway, but expect degraded Bayesian Expert")

# Verify data quality
sample_stat = next(iter(bayesian_stats.values()))
required_keys = ["n_retrievals", "n_citations"]
for key in required_keys:
    if key not in sample_stat:
        print(f"\n❌ ERROR: Missing key '{key}' in statistics")
        print(f"   Sample record: {sample_stat}")
        exit(1)

print(f"   ✓ Data quality check passed")

# Load E5 model
print("\n3. Loading E5 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
e5_path = "/home/jx/MNEMOSYNE-MNS/deploy/e5_model"
tokenizer = AutoTokenizer.from_pretrained(e5_path)
e5_model = AutoModel.from_pretrained(e5_path).to(device)
e5_model.eval()
print(f"   ✓ E5 on {device}")

# Encode
print("\n4. Encoding with E5...")
def encode_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = e5_model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

train_q = [str(s.get("query_text", "")) for s in train_samples]
train_m = [str(s.get("memory_text", "")) for s in train_samples]
val_q = [str(s.get("query_text", "")) for s in val_samples]
val_m = [str(s.get("memory_text", "")) for s in val_samples]

t0 = time.time()
print("   Encoding train queries...")
train_q_emb = encode_batch(train_q)
print("   Encoding train memories...")
train_m_emb = encode_batch(train_m)
print("   Encoding val queries...")
val_q_emb = encode_batch(val_q)
print("   Encoding val memories...")
val_m_emb = encode_batch(val_m)
print(f"   ✓ Encoding complete ({time.time()-t0:.1f}s)")

# Define feature extraction for each expert
print("\n5. Extracting expert-specific features...")

def extract_lexical_features(sample, q_emb, m_emb):
    """Expert 1: Lexical-only (8-dim)"""
    query = str(sample.get("query_text", ""))
    memory = str(sample.get("memory_text", ""))
    
    q_words = query.lower().split()
    m_words = memory.lower().split()
    q_set, m_set = set(q_words), set(m_words)
    overlap = len(q_set & m_set)
    union = len(q_set | m_set)
    
    return [
        np.log1p(len(q_words)),
        np.log1p(len(m_words)),
        len(q_words) / (len(m_words) + 1e-8),
        overlap,
        overlap / (union + 1e-8),
        overlap / (len(q_set) + 1e-8),
        1.0 if any(w in query for w in ["不", "没", "dont", "not"]) else 0.0,
        1.0 if any(w in memory for w in ["不", "没", "dont", "not"]) else 0.0,
    ]

def extract_semantic_features(sample, q_emb, m_emb):
    """Expert 2: Semantic-only (5-dim)"""
    return [
        float(np.dot(q_emb, m_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(m_emb) + 1e-8)),
        float(np.linalg.norm(q_emb - m_emb)),
        float(np.sum(np.abs(q_emb - m_emb))),
        float(np.max(np.abs(q_emb - m_emb))),
        float(np.dot(q_emb, m_emb)),
    ]

def extract_bayesian_features(sample, q_emb, m_emb, bayesian_stats, n_train):
    """Expert 3: Bayesian-focused (8-dim) - NOW WITH REAL DATA"""
    mem_id = sample.get("memory_id")
    
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
        has_data = 1.0
        log_ret = np.log1p(n_ret)
    else:
        mean, std, ci_w = 0.5, 0.29, 0.95
        coverage, consistency, confidence = 0.0, 0.0, 0.05
        has_data = 0.0
        log_ret = 0.0
    
    return [mean, std, ci_w, coverage, consistency, confidence, has_data, log_ret]

def extract_hybrid_features(sample, q_emb, m_emb, bayesian_stats, n_train):
    """Expert 4: Balanced hybrid (19-dim)"""
    lexical = extract_lexical_features(sample, q_emb, m_emb)
    semantic = extract_semantic_features(sample, q_emb, m_emb)[:3]
    
    query = str(sample.get("query_text", ""))
    memory = str(sample.get("memory_text", ""))
    q_words = query.lower().split()
    m_words = memory.lower().split()
    q_set, m_set = set(q_words), set(m_words)
    
    enhanced = [
        len(q_set) / (len(q_words) + 1e-8),
        len(m_set) / (len(m_words) + 1e-8),
    ]
    
    bayesian = extract_bayesian_features(sample, q_emb, m_emb, bayesian_stats, n_train)[:6]
    
    return lexical + semantic + enhanced + bayesian

# Extract features
print("   Extracting features for 4 experts...")
X_train_experts = [[], [], [], []]
X_val_experts = [[], [], [], []]
y_train = []
y_val = []

for i, s in enumerate(train_samples):
    if i % 5000 == 0:
        print(f"   Train {i:,}/{len(train_samples):,}")
    
    X_train_experts[0].append(extract_lexical_features(s, train_q_emb[i], train_m_emb[i]))
    X_train_experts[1].append(extract_semantic_features(s, train_q_emb[i], train_m_emb[i]))
    X_train_experts[2].append(extract_bayesian_features(s, train_q_emb[i], train_m_emb[i], bayesian_stats, len(train_samples)))
    X_train_experts[3].append(extract_hybrid_features(s, train_q_emb[i], train_m_emb[i], bayesian_stats, len(train_samples)))
    
    y_train.append(1 if s.get("label") in ["citation", "correction"] else 0)

for i, s in enumerate(val_samples):
    X_val_experts[0].append(extract_lexical_features(s, val_q_emb[i], val_m_emb[i]))
    X_val_experts[1].append(extract_semantic_features(s, val_q_emb[i], val_m_emb[i]))
    X_val_experts[2].append(extract_bayesian_features(s, val_q_emb[i], val_m_emb[i], bayesian_stats, len(train_samples)))
    X_val_experts[3].append(extract_hybrid_features(s, val_q_emb[i], val_m_emb[i], bayesian_stats, len(train_samples)))
    
    y_val.append(1 if s.get("label") in ["citation", "correction"] else 0)

for i in range(4):
    X_train_experts[i] = np.array(X_train_experts[i], dtype=np.float32)
    X_val_experts[i] = np.array(X_val_experts[i], dtype=np.float32)

y_train = np.array(y_train)
y_val = np.array(y_val)

print(f"   ✓ Expert shapes: {[x.shape for x in X_train_experts]}")

# Verify Bayesian features have variance
bayes_features = X_train_experts[2]
bayes_variance = np.var(bayes_features, axis=0)
print(f"\n   Bayesian feature variance check:")
print(f"     Mean variance: {np.mean(bayes_variance):.6f}")
print(f"     Min variance:  {np.min(bayes_variance):.6f}")
print(f"     Max variance:  {np.max(bayes_variance):.6f}")

if np.mean(bayes_variance) < 0.001:
    print(f"\n⚠️  WARNING: Bayesian features have very low variance!")
    print(f"   This suggests most samples are using priors.")
    print(f"   Expected if <50% of memories have statistics.")

# Train experts
print("\n6. Training 4 expert LightGBM models...")
experts = []
expert_names = ["Lexical", "Semantic", "Bayesian", "Hybrid"]

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

for i, name in enumerate(expert_names):
    print(f"\n   Training Expert {i+1}: {name}...")
    train_data = lgb.Dataset(X_train_experts[i], label=y_train)
    val_data = lgb.Dataset(X_val_experts[i], label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=150,
        valid_sets=[val_data],
        valid_names=["valid"],
    )
    experts.append(model)
    
    # Report AUC
    from sklearn.metrics import roc_auc_score
    train_pred = model.predict(X_train_experts[i])
    val_pred = model.predict(X_val_experts[i])
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"   ✓ Expert {i+1} trained: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

# Get predictions
print("\n7. Getting expert predictions...")
train_expert_preds = np.column_stack([exp.predict(X_train_experts[i]) for i, exp in enumerate(experts)])
val_expert_preds = np.column_stack([exp.predict(X_val_experts[i]) for i, exp in enumerate(experts)])

print(f"   Train predictions: {train_expert_preds.shape}")
print(f"   Val predictions: {val_expert_preds.shape}")

# Train router
print("\n8. Training router (Logistic Regression)...")
router = LogisticRegression(max_iter=1000, C=1.0)
router.fit(train_expert_preds, y_train)

router_weights = router.coef_[0]
print(f"\n   Router learned weights:")
for i, name in enumerate(expert_names):
    print(f"     {name:12s}: {router_weights[i]:>7.3f}")

# Analyze weight distribution
total_pos_weight = sum(w for w in router_weights if w > 0)
print(f"\n   Effective weight distribution:")
for i, name in enumerate(expert_names):
    if router_weights[i] > 0:
        pct = router_weights[i] / total_pos_weight * 100
        print(f"     {name:12s}: {pct:>5.1f}%")
    else:
        print(f"     {name:12s}: suppressed (negative weight)")

# Final predictions
print("\n9. Computing final MoE predictions...")
train_moe_preds = router.predict_proba(train_expert_preds)[:, 1]
val_moe_preds = router.predict_proba(val_expert_preds)[:, 1]

# Evaluate
from sklearn.metrics import roc_auc_score
train_auc_moe = roc_auc_score(y_train, train_moe_preds)
val_auc_moe = roc_auc_score(y_val, val_moe_preds)

print(f"\n   MoE Ensemble Performance:")
print(f"     Train AUC: {train_auc_moe:.4f}")
print(f"     Val AUC:   {val_auc_moe:.4f}")

# Save
print("\n10. Saving MoE ensemble...")
out_dir = Path("runs/rank_v3_moe_fixed")
out_dir.mkdir(parents=True, exist_ok=True)

for i, (exp, name) in enumerate(zip(experts, expert_names)):
    exp.save_model(str(out_dir / f"expert_{i}_{name.lower()}.txt"))

import pickle
with open(out_dir / "router.pkl", "wb") as f:
    pickle.dump(router, f)

metadata = {
    "expert_names": expert_names,
    "router_weights": router_weights.tolist(),
    "feature_dims": [x.shape[1] for x in X_train_experts],
    "train_auc": float(train_auc_moe),
    "val_auc": float(val_auc_moe),
    "bayesian_stats_count": len(bayesian_stats),
}

with open(out_dir / "moe_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ MoE ensemble saved to {out_dir}/")

print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print("MoE Training Complete - FIXED VERSION")
print("Next: Compare with V3 Optimized")
print("=" * 60)
