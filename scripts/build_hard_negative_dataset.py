#!/usr/bin/env python3
"""
Build ranking dataset with Hard Negative Mining.

Strategy: For each positive (query, memory) pair, sample negatives that are
semantically similar but incorrect (hard negatives), instead of random sampling.
"""
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import math

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

class BM25:
    """Simple BM25 for hard negative mining."""
    def __init__(self, corpus: List[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        
        # Build inverted index
        self.doc_lens = []
        self.term_doc_freq = defaultdict(int)
        self.inverted_index = defaultdict(list)
        
        for doc_id, doc in enumerate(corpus):
            tokens = doc.lower().split()
            self.doc_lens.append(len(tokens))
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                self.term_doc_freq[token] += 1
                self.inverted_index[token].append(doc_id)
        
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0
    
    def score(self, query: str, doc_id: int) -> float:
        """BM25 score for a query-document pair."""
        tokens = query.lower().split()
        score = 0.0
        
        for token in tokens:
            if token not in self.inverted_index:
                continue
            
            df = self.term_doc_freq[token]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            
            # Term frequency in document
            doc_tokens = self.corpus[doc_id].lower().split()
            tf = doc_tokens.count(token)
            
            # BM25 formula
            dl = self.doc_lens[doc_id]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve_top_k(self, query: str, k: int, exclude_ids: set = None) -> List[int]:
        """Retrieve top-k documents by BM25 score."""
        if exclude_ids is None:
            exclude_ids = set()
        
        # Candidate documents (union of all query terms)
        candidates = set()
        for token in query.lower().split():
            if token in self.inverted_index:
                candidates.update(self.inverted_index[token])
        
        # Score and rank
        scored = []
        for doc_id in candidates:
            if doc_id not in exclude_ids:
                score = self.score(query, doc_id)
                scored.append((doc_id, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:k]]

def build_hard_negative_dataset():
    """Build ranking dataset with hard negative mining."""
    
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
    
    # Extract positive pairs and memory pool
    positive_pairs = []
    memory_pool = {}  # memory_id -> (memory_text, memory_id)
    
    for row in all_data:
        query_id = row.get("query_id")
        query_text = row.get("query_text", "")
        memory_id = row.get("memory_id")
        memory_text = row.get("memory_text", "")
        
        if memory_id and memory_text:
            positive_pairs.append({
                "query_id": query_id,
                "query_text": query_text,
                "memory_id": memory_id,
                "memory_text": memory_text,
                "label": row.get("label"),
                "source": row.get("source"),
                "query_timestamp": row.get("query_timestamp"),
                "memory_timestamp": row.get("memory_timestamp"),
            })
            memory_pool[memory_id] = memory_text
    
    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Memory pool size: {len(memory_pool)}")
    
    # Build BM25 index for hard negative mining
    print("\nBuilding BM25 index for hard negative mining...")
    memory_ids = list(memory_pool.keys())
    memory_texts = [memory_pool[mid] for mid in memory_ids]
    bm25 = BM25(memory_texts)
    
    # Build ranking data with hard negatives
    print("Mining hard negatives...")
    ranking_data = []
    NEG_SAMPLES_PER_QUERY = 4
    
    for i, pos_pair in enumerate(positive_pairs):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(positive_pairs)} queries...")
        
        query_text = pos_pair["query_text"]
        pos_memory_id = pos_pair["memory_id"]
        pos_memory_idx = memory_ids.index(pos_memory_id)
        
        # Add positive sample
        ranking_data.append(pos_pair.copy())
        
        # Retrieve hard negatives (BM25 top-k, excluding positive)
        hard_neg_indices = bm25.retrieve_top_k(
            query_text,
            k=NEG_SAMPLES_PER_QUERY * 2,  # Retrieve more to have buffer
            exclude_ids={pos_memory_idx}
        )
        
        # Sample from top hard negatives
        sampled_neg_indices = hard_neg_indices[:NEG_SAMPLES_PER_QUERY]
        
        # If not enough hard negatives, fill with random
        if len(sampled_neg_indices) < NEG_SAMPLES_PER_QUERY:
            remaining = NEG_SAMPLES_PER_QUERY - len(sampled_neg_indices)
            exclude_set = {pos_memory_idx} | set(sampled_neg_indices)
            available = [idx for idx in range(len(memory_ids)) if idx not in exclude_set]
            sampled_neg_indices.extend(random.sample(available, min(remaining, len(available))))
        
        # Add negative samples
        for neg_idx in sampled_neg_indices:
            neg_memory_id = memory_ids[neg_idx]
            neg_sample = {
                "query_id": pos_pair["query_id"],
                "query_text": pos_pair["query_text"],
                "memory_id": None,  # Negative has no memory_id
                "memory_text": memory_pool[neg_memory_id],
                "label": "unknown",
                "source": pos_pair["source"],
                "query_timestamp": pos_pair["query_timestamp"],
                "memory_timestamp": None,
            }
            ranking_data.append(neg_sample)
    
    print(f"\nTotal ranking samples (with hard negatives): {len(ranking_data)}")
    
    # Time-based split (70/20/10)
    ranking_data.sort(key=lambda x: x.get("query_timestamp") or "")
    
    n_total = len(ranking_data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.20)
    
    train_data = ranking_data[:n_train]
    val_data = ranking_data[n_train:n_train+n_val]
    holdout_data = ranking_data[n_train+n_val:]
    
    # Statistics
    def count_stats(data):
        pos = sum(1 for x in data if x.get("memory_id"))
        neg = len(data) - pos
        queries = len(set(x["query_id"] for x in data))
        return pos, neg, queries
    
    train_pos, train_neg, train_q = count_stats(train_data)
    val_pos, val_neg, val_q = count_stats(val_data)
    holdout_pos, holdout_neg, holdout_q = count_stats(holdout_data)
    
    print("\n" + "="*60)
    print("Hard Negative Mining Dataset Statistics:")
    print("="*60)
    print(f"Train:   {len(train_data):6d} samples, {train_q:5d} queries ({train_pos:5d} pos, {train_neg:5d} neg, {train_neg/len(train_data)*100:.1f}% neg)")
    print(f"Val:     {len(val_data):6d} samples, {val_q:5d} queries ({val_pos:5d} pos, {val_neg:5d} neg, {val_neg/len(val_data)*100:.1f}% neg)")
    print(f"Holdout: {len(holdout_data):6d} samples, {holdout_q:5d} queries ({holdout_pos:5d} pos, {holdout_neg:5d} neg, {holdout_neg/len(holdout_data)*100:.1f}% neg)")
    print("="*60)
    
    # Save
    save_jsonl(train_data, "data/unified/train_v2.jsonl")
    save_jsonl(val_data, "data/unified/val_v2.jsonl")
    save_jsonl(holdout_data, "data/unified/holdout_v2.jsonl")
    
    print("\n✓ Hard negative mining dataset saved successfully!")
    print("Note: Negatives are now BM25-similar but incorrect (hard negatives)")
    return True

if __name__ == "__main__":
    success = build_hard_negative_dataset()
    exit(0 if success else 1)
