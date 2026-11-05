#!/usr/bin/env python3
"""
重建验证集和holdout集，确保包含负样本。

问题：当前验证集100%正样本，无法有效验证模型。
解决：使用分层采样(stratified split)，按query分组，确保正负样本比例一致。
"""
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

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

def rebuild_splits():
    """重新切分数据，确保验证集和holdout集包含负样本"""
    
    # 加载当前所有数据
    train_path = Path("data/unified/train_v2.jsonl")
    val_path = Path("data/unified/val_v2.jsonl")
    holdout_path = Path("data/unified/holdout_v2.jsonl")
    
    all_data = []
    all_data.extend(load_jsonl(train_path))
    all_data.extend(load_jsonl(val_path))
    all_data.extend(load_jsonl(holdout_path))
    
    print(f"Total samples loaded: {len(all_data)}")
    
    # 按query_id分组
    query_groups = defaultdict(list)
    for sample in all_data:
        query_id = sample.get("query_id")
        if query_id:
            query_groups[query_id].append(sample)
    
    print(f"Total unique queries: {len(query_groups)}")
    
    # 统计每个query的正负样本比例
    query_stats = {}
    for query_id, samples in query_groups.items():
        pos = sum(1 for s in samples if s.get("memory_id"))
        neg = len(samples) - pos
        query_stats[query_id] = {"pos": pos, "neg": neg, "total": len(samples)}
    
    # 分离有负样本和无负样本的query
    queries_with_neg = [qid for qid, stats in query_stats.items() if stats["neg"] > 0]
    queries_no_neg = [qid for qid, stats in query_stats.items() if stats["neg"] == 0]
    
    print(f"Queries with negative samples: {len(queries_with_neg)}")
    print(f"Queries without negative samples: {len(queries_no_neg)}")
    
    # 随机打乱
    random.shuffle(queries_with_neg)
    random.shuffle(queries_no_neg)
    
    # 按70/20/10切分query（确保验证集和holdout集都有负样本）
    n_total_with_neg = len(queries_with_neg)
    n_train_with_neg = int(n_total_with_neg * 0.70)
    n_val_with_neg = int(n_total_with_neg * 0.20)
    
    train_queries_with_neg = queries_with_neg[:n_train_with_neg]
    val_queries_with_neg = queries_with_neg[n_train_with_neg:n_train_with_neg+n_val_with_neg]
    holdout_queries_with_neg = queries_with_neg[n_train_with_neg+n_val_with_neg:]
    
    # 将无负样本的query也按70/20/10分配
    n_total_no_neg = len(queries_no_neg)
    n_train_no_neg = int(n_total_no_neg * 0.70)
    n_val_no_neg = int(n_total_no_neg * 0.20)
    
    train_queries_no_neg = queries_no_neg[:n_train_no_neg]
    val_queries_no_neg = queries_no_neg[n_train_no_neg:n_train_no_neg+n_val_no_neg]
    holdout_queries_no_neg = queries_no_neg[n_train_no_neg+n_val_no_neg:]
    
    # 合并
    train_queries = set(train_queries_with_neg + train_queries_no_neg)
    val_queries = set(val_queries_with_neg + val_queries_no_neg)
    holdout_queries = set(holdout_queries_with_neg + holdout_queries_no_neg)
    
    # 构建新数据集
    new_train = []
    new_val = []
    new_holdout = []
    
    for query_id, samples in query_groups.items():
        if query_id in train_queries:
            new_train.extend(samples)
        elif query_id in val_queries:
            new_val.extend(samples)
        elif query_id in holdout_queries:
            new_holdout.extend(samples)
    
    # 统计
    def count_samples(data):
        pos = sum(1 for s in data if s.get("memory_id"))
        neg = len(data) - pos
        return pos, neg
    
    train_pos, train_neg = count_samples(new_train)
    val_pos, val_neg = count_samples(new_val)
    holdout_pos, holdout_neg = count_samples(new_holdout)
    
    print("\n" + "="*60)
    print("New Split Statistics:")
    print("="*60)
    print(f"Train:   {len(new_train):5d} samples ({train_pos:5d} pos, {train_neg:5d} neg, {train_neg/len(new_train)*100:.1f}% neg)")
    print(f"Val:     {len(new_val):5d} samples ({val_pos:5d} pos, {val_neg:5d} neg, {val_neg/len(new_val)*100:.1f}% neg)")
    print(f"Holdout: {len(new_holdout):5d} samples ({holdout_pos:5d} pos, {holdout_neg:5d} neg, {holdout_neg/len(new_holdout)*100:.1f}% neg)")
    print(f"Total:   {len(new_train)+len(new_val)+len(new_holdout)} samples")
    print("="*60)
    
    # 保存
    save_jsonl(new_train, "data/unified/train_v2.jsonl")
    save_jsonl(new_val, "data/unified/val_v2.jsonl")
    save_jsonl(new_holdout, "data/unified/holdout_v2.jsonl")
    
    print("\n✓ New datasets saved successfully!")
    
    # 验证
    if val_neg == 0 or holdout_neg == 0:
        print("\n⚠️  WARNING: Validation or holdout set still has no negative samples!")
        return False
    
    return True

if __name__ == "__main__":
    success = rebuild_splits()
    exit(0 if success else 1)
