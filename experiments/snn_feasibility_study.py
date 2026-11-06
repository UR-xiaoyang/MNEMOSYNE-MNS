#!/usr/bin/env python3
"""
SNN (Spiking Neural Network) Feasibility Study for MNEMOSYNE L1 Layer.

Experiment Goal: Validate if SNN can replace TF-IDF/BM25 for L1 keyword matching
with better latency and recall.

Phase 1: CPU-based prototype using snnTorch
- Compare SNN vs TF-IDF on latency and recall
- Dataset: 1000 queries from unified v2
- Success criteria: Recall >= 95%, Latency < 10ms
"""
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

# Try to import SNN libraries
try:
    import torch
    import snntorch as snn
    from snntorch import utils
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False
    print("⚠️  snnTorch not installed. Run: pip install snntorch torch")

def load_test_queries(dataset_path: str, n_samples: int = 1000) -> List[dict]:
    """Load test queries from unified dataset."""
    queries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            queries.append(json.loads(line))
    return queries

class TFIDFBaseline:
    """Baseline: Traditional TF-IDF keyword matching."""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.idf = {}
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        from collections import Counter
        
        # Extract all words
        all_words = []
        for text in texts:
            words = str(text or "").lower().split()
            all_words.extend(words)
        
        # Get top vocab_size words
        word_counts = Counter(all_words)
        top_words = [word for word, _ in word_counts.most_common(self.vocab_size)]
        
        self.vocabulary = {word: idx for idx, word in enumerate(top_words)}
        
        # Compute IDF
        doc_freq = defaultdict(int)
        for text in texts:
            words = set(str(text or "").lower().split())
            for word in words:
                if word in self.vocabulary:
                    doc_freq[word] += 1
        
        n_docs = len(texts)
        for word in self.vocabulary:
            self.idf[word] = np.log(n_docs / (doc_freq[word] + 1))
    
    def vectorize(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
        vec = np.zeros(self.vocab_size)
        words = str(text or "").lower().split()
        
        # Compute TF
        word_counts = defaultdict(int)
        for word in words:
            if word in self.vocabulary:
                word_counts[word] += 1
        
        # Compute TF-IDF
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / len(words) if words else 0
                vec[idx] = tf * self.idf.get(word, 0)
        
        return vec
    
    def match(self, query_vec: np.ndarray, memory_vecs: List[np.ndarray], top_k: int = 10) -> List[int]:
        """Match query to memories using cosine similarity."""
        scores = []
        for mem_vec in memory_vecs:
            # Cosine similarity
            dot = np.dot(query_vec, mem_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_m = np.linalg.norm(mem_vec)
            score = dot / (norm_q * norm_m + 1e-8)
            scores.append(score)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()

class SNNMatcher:
    """SNN-based keyword matcher using Leaky Integrate-and-Fire neurons."""
    
    def __init__(self, vocab_size: int = 5000, time_steps: int = 50):
        if not SNN_AVAILABLE:
            raise RuntimeError("snnTorch not available")
        
        self.vocab_size = vocab_size
        self.time_steps = time_steps
        self.device = torch.device("cpu")  # CPU for now
        
        # Simple 1-layer LIF network
        self.lif = snn.Leaky(beta=0.9)
        self.vocabulary = {}
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary (same as TF-IDF for fair comparison)."""
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = str(text or "").lower().split()
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_words = [word for word, _ in word_counts.most_common(self.vocab_size)]
        self.vocabulary = {word: idx for idx, word in enumerate(top_words)}
    
    def text_to_spikes(self, text: str) -> torch.Tensor:
        """Convert text to spike train using rate encoding."""
        words = str(text or "").lower().split()
        
        # Word frequency as spike rate
        word_freq = np.zeros(self.vocab_size)
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                word_freq[idx] += 1
        
        # Normalize to [0, 1]
        if word_freq.max() > 0:
            word_freq = word_freq / word_freq.max()
        
        # Rate encoding: generate Poisson spike train
        spike_train = torch.zeros((self.time_steps, self.vocab_size), device=self.device)
        for t in range(self.time_steps):
            # Poisson process: P(spike) = rate
            spikes = torch.rand(self.vocab_size) < torch.tensor(word_freq)
            spike_train[t] = spikes.float()
        
        return spike_train
    
    def match(self, query_text: str, memory_texts: List[str], top_k: int = 10) -> List[int]:
        """Match query to memories using SNN."""
        query_spikes = self.text_to_spikes(query_text)
        
        scores = []
        for memory_text in memory_texts:
            memory_spikes = self.text_to_spikes(memory_text)
            
            # Compute spike correlation over time
            mem = self.lif.init_leaky()
            total_output_spikes = 0
            
            for t in range(self.time_steps):
                # Input: element-wise product of query and memory spikes
                input_current = query_spikes[t] * memory_spikes[t]
                
                # LIF dynamics
                spk, mem = self.lif(input_current, mem)
                total_output_spikes += spk.sum().item()
            
            scores.append(total_output_spikes)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()

def run_experiment(dataset_path: str, n_samples: int = 1000):
    """Run SNN vs TF-IDF comparison experiment."""
    
    print("="*60)
    print("SNN Feasibility Experiment for MNEMOSYNE L1 Layer")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading {n_samples} test queries...")
    queries = load_test_queries(dataset_path, n_samples)
    print(f"   ✓ Loaded {len(queries)} queries")
    
    # Extract texts
    query_texts = [q.get('query_text', '') for q in queries]
    memory_texts = [q.get('memory_text', '') for q in queries]
    
    # Build vocabulary
    print("\n2. Building vocabulary...")
    all_texts = query_texts + memory_texts
    
    # Initialize models
    print("\n3. Initializing models...")
    tfidf_model = TFIDFBaseline(vocab_size=5000)
    tfidf_model.build_vocab(all_texts)
    print("   ✓ TF-IDF baseline ready")
    
    if SNN_AVAILABLE:
        snn_model = SNNMatcher(vocab_size=5000, time_steps=50)
        snn_model.build_vocab(all_texts)
        print("   ✓ SNN model ready")
    else:
        snn_model = None
        print("   ⚠️  SNN model skipped (library not available)")
    
    # Benchmark TF-IDF
    print("\n4. Benchmarking TF-IDF...")
    tfidf_times = []
    for i, query_text in enumerate(query_texts[:100]):  # Test on 100 queries
        query_vec = tfidf_model.vectorize(query_text)
        memory_vecs = [tfidf_model.vectorize(mem) for mem in memory_texts[:100]]
        
        start = time.time()
        top_k = tfidf_model.match(query_vec, memory_vecs, top_k=10)
        elapsed = (time.time() - start) * 1000  # ms
        tfidf_times.append(elapsed)
    
    tfidf_mean = np.mean(tfidf_times)
    tfidf_std = np.std(tfidf_times)
    print(f"   TF-IDF Latency: {tfidf_mean:.2f} ± {tfidf_std:.2f} ms")
    
    # Benchmark SNN
    if SNN_AVAILABLE and snn_model:
        print("\n5. Benchmarking SNN...")
        snn_times = []
        for i, query_text in enumerate(query_texts[:100]):
            start = time.time()
            top_k = snn_model.match(query_text, memory_texts[:100], top_k=10)
            elapsed = (time.time() - start) * 1000  # ms
            snn_times.append(elapsed)
        
        snn_mean = np.mean(snn_times)
        snn_std = np.std(snn_times)
        print(f"   SNN Latency: {snn_mean:.2f} ± {snn_std:.2f} ms")
        
        # Compare
        speedup = tfidf_mean / snn_mean
        print(f"\n   Speedup: {speedup:.2f}x {'(SNN faster)' if speedup > 1 else '(TF-IDF faster)'}")
    else:
        print("\n5. SNN benchmark skipped")
        snn_mean = None
    
    # Results summary
    print("\n" + "="*60)
    print("Experiment Results")
    print("="*60)
    print(f"Dataset: {n_samples} queries")
    print(f"Vocabulary size: 5000 words")
    print(f"Test set: 100 queries x 100 memories")
    print()
    print(f"TF-IDF Baseline:  {tfidf_mean:.2f} ± {tfidf_std:.2f} ms")
    if snn_mean:
        print(f"SNN (CPU):        {snn_mean:.2f} ± {snn_std:.2f} ms")
        print(f"Speedup:          {speedup:.2f}x")
    print("="*60)
    
    # Save results
    output_dir = Path("experiments/snn_feasibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "SNN_feasibility_L1_layer",
        "dataset_size": n_samples,
        "vocab_size": 5000,
        "test_queries": 100,
        "test_memories": 100,
        "tfidf_latency_ms": {
            "mean": float(tfidf_mean),
            "std": float(tfidf_std)
        },
        "snn_available": SNN_AVAILABLE,
    }
    
    if snn_mean:
        results["snn_latency_ms"] = {
            "mean": float(snn_mean),
            "std": float(snn_std)
        }
        results["speedup"] = float(speedup)
        results["conclusion"] = "SNN faster" if speedup > 1 else "TF-IDF faster"
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/results.json")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Check if dataset exists
    dataset_path = "data/unified/train_v2.jsonl"
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\nPlease run data integration first:")
        print("  python scripts/integrate_p2_data.py")
        sys.exit(1)
    
    # Run experiment
    results = run_experiment(dataset_path, n_samples=1000)
    
    # Next steps
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    if not SNN_AVAILABLE:
        print("1. Install snnTorch:")
        print("   pip install snntorch torch")
        print("2. Re-run this script")
    elif results.get("speedup", 0) > 1:
        print("✓ SNN shows promise! Consider:")
        print("  1. Test on neuromorphic hardware (Intel Loihi 2)")
        print("  2. Implement in production L1 layer")
        print("  3. Measure recall vs TF-IDF")
    else:
        print("⚠️  SNN slower on CPU. Options:")
        print("  1. Optimize spike encoding")
        print("  2. Test on neuromorphic hardware")
        print("  3. Stick with TF-IDF for L1")
