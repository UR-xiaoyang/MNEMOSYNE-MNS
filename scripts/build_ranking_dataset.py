#!/usr/bin/env python3
"""
为ranking任务正确构建数据集。

Ranking需求：每个query有多个候选memory（正例+负例混合）
当前问题：数据是(query, memory)对，每个query只对应一个memory
解决方案：为每个query采样多个候选memory（正例+随机负例）
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

def build_ranking_dataset():
    """为每个query构建候选列表（正例+负采样）"""
    
    # 1. 加载原始数据并按source分组
    print("Loading original data...")
    all_files = [
        "data/unified/train_v2.jsonl",
        "data/unified/val_v2.jsonl", 
        "data/unified/holdout_v2.jsonl"
    ]
    
    all_data = []
    for f in all_files:
        all_data.extend(load_jsonl(f))
    
    print(f"Total samples: {len(all_data)}")
    
    # 2. 分离query-memory对和memory库
    positive_pairs = []  # (query_id, query_text, memory_id, memory_text, label)
    memory_pool = {}     # memory_id -> memory_text
    
    for row in all_data:
        query_id = row.get("query_id")
        query_text = row.get("query_text", "")
        memory_id = row.get("memory_id")
        memory_text = row.get("memory_text", "")
        label = row.get("label", "unknown")
        
        if memory_id and memory_text:
            # 这是一个正例
            positive_pairs.append({
                "query_id": query_id,
                "query_text": query_text,
                "memory_id": memory_id,
                "memory_text": memory_text,
                "label": label,
                "source": row.get("source"),
                "query_timestamp": row.get("query_timestamp"),
                "memory_timestamp": row.get("memory_timestamp"),
            })
            memory_pool[memory_id] = memory_text
    
    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Memory pool size: {len(memory_pool)}")
    
    # 3. 为每个positive pair添加负采样
    print("\\nBuilding ranking samples with negative sampling...")
    ranking_data = []
    all_memory_ids = list(memory_pool.keys())
    
    NEG_SAMPLES_PER_QUERY = 4  # 每个query采样4个负例
    
    for pos_pair in positive_pairs:
        query_id = pos_pair["query_id"]
        pos_memory_id = pos_pair["memory_id"]
        
        # 添加正例
        pos_sample = pos_pair.copy()
        ranking_data.append(pos_sample)
        
        # 随机采样负例
        neg_candidates = [mid for mid in all_memory_ids if mid != pos_memory_id]
        sampled_neg = random.sample(neg_candidates, min(NEG_SAMPLES_PER_QUERY, len(neg_candidates)))
        
        for neg_memory_id in sampled_neg:
            neg_sample = {
                "query_id": query_id,
                "query_text": pos_pair["query_text"],
                "memory_id": None,  # 负例没有memory_id
                "memory_text": memory_pool[neg_memory_id],
                "label": "unknown",  # 负例标签
                "source": pos_pair["source"],
                "query_timestamp": pos_pair["query_timestamp"],
                "memory_timestamp": None,
            }
            ranking_data.append(neg_sample)
    
    print(f"Total ranking samples (with negatives): {len(ranking_data)}")
    
    # 4. 按时序切分（70/20/10）
    ranking_data.sort(key=lambda x: x.get("query_timestamp") or "")
    
    n_total = len(ranking_data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.20)
    
    train_data = ranking_data[:n_train]
    val_data = ranking_data[n_train:n_train+n_val]
    holdout_data = ranking_data[n_train+n_val:]
    
    # 统计
    def count_stats(data):
        pos = sum(1 for x in data if x.get("memory_id"))
        neg = len(data) - pos
        queries = len(set(x["query_id"] for x in data))
        return pos, neg, queries
    
    train_pos, train_neg, train_q = count_stats(train_data)
    val_pos, val_neg, val_q = count_stats(val_data)
    holdout_pos, holdout_neg, holdout_q = count_stats(holdout_data)
    
    print("\\n" + "="*60)
    print("Ranking Dataset Statistics:")
    print("="*60)
    print(f"Train:   {len(train_data):6d} samples, {train_q:5d} queries ({train_pos:5d} pos, {train_neg:5d} neg, {train_neg/len(train_data)*100:.1f}% neg)")
    print(f"Val:     {len(val_data):6d} samples, {val_q:5d} queries ({val_pos:5d} pos, {val_neg:5d} neg, {val_neg/len(val_data)*100:.1f}% neg)")
    print(f"Holdout: {len(holdout_data):6d} samples, {holdout_q:5d} queries ({holdout_pos:5d} pos, {holdout_neg:5d} neg, {holdout_neg/len(holdout_data)*100:.1f}% neg)")
    print("="*60)
    
    # 5. 保存
    save_jsonl(train_data, "data/unified/train_v2.jsonl")
    save_jsonl(val_data, "data/unified/val_v2.jsonl")
    save_jsonl(holdout_data, "data/unified/holdout_v2.jsonl")
    
    print("\\n✓ Ranking datasets saved successfully!")
    return True

if __name__ == "__main__":
    success = build_ranking_dataset()
    exit(0 if success else 1)
