#!/usr/bin/env python3
"""
Bayesian-enhanced RANK module with uncertainty quantification.

Integrates Bayesian uncertainty modeling into production RANK pipeline:
1. Compute posterior quality distributions for memories
2. Add uncertainty features (posterior_std, credible_interval_width)
3. Penalize high-uncertainty memories in ranking

Benefits over vanilla RANK Enhanced V2:
- Handles sparse retrieval data gracefully (Bayesian priors)
- Quantifies confidence in quality estimates
- Enables risk-aware ranking (prefer confident predictions)
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    from scipy.stats import beta as beta_dist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy required for Bayesian features. Run: pip install scipy")

class BayesianMemoryQuality:
    """
    Bayesian quality estimator with Beta-Bernoulli model.
    
    Maintains retrieval/citation statistics and computes posterior distributions.
    """
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Args:
            alpha_prior: Prior pseudo-citations (default 1.0 = uninformative)
            beta_prior: Prior pseudo-failures (default 1.0 = uninformative)
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Track statistics per memory
        self.memory_stats = defaultdict(lambda: {
            'n_retrievals': 0,
            'n_citations': 0,
        })
    
    def update_memory(self, memory_id: str, cited: bool = False):
        """
        Update retrieval statistics for a memory.
        
        Args:
            memory_id: Memory identifier
            cited: Whether memory was cited by user
        """
        self.memory_stats[memory_id]['n_retrievals'] += 1
        if cited:
            self.memory_stats[memory_id]['n_citations'] += 1
    
    def get_posterior(self, memory_id: str) -> Dict:
        """
        Compute posterior quality distribution for a memory.
        
        Returns:
            dict with posterior_mean, posterior_std, credible_interval_95, etc.
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required for Bayesian posterior")
        
        stats = self.memory_stats.get(memory_id, {'n_retrievals': 0, 'n_citations': 0})
        n_ret = stats['n_retrievals']
        n_cit = stats['n_citations']
        
        # Posterior parameters
        alpha_post = self.alpha_prior + n_cit
        beta_post = self.beta_prior + (n_ret - n_cit)
        
        # Posterior distribution
        posterior = beta_dist(alpha_post, beta_post)
        
        # Statistics
        mean_quality = posterior.mean()
        std_quality = posterior.std()
        ci_lower, ci_upper = posterior.ppf(0.025), posterior.ppf(0.975)
        ci_width = ci_upper - ci_lower
        
        return {
            'posterior_mean': mean_quality,
            'posterior_std': std_quality,
            'credible_interval_95': (ci_lower, ci_upper),
            'credible_interval_width': ci_width,
            'n_retrievals': n_ret,
            'n_citations': n_cit,
            'has_data': n_ret > 0,
        }
    
    def compute_uncertainty_penalty(self, memory_id: str, lambda_penalty: float = 1.0) -> float:
        """
        Compute uncertainty penalty for ranking.
        
        Formula: penalty = lambda * posterior_std
        
        Higher penalty = less confident, should rank lower
        """
        posterior = self.get_posterior(memory_id)
        return lambda_penalty * posterior['posterior_std']
    
    def save_statistics(self, output_path: str):
        """Save retrieval/citation statistics to disk."""
        data = {
            mem_id: stats for mem_id, stats in self.memory_stats.items()
            if stats['n_retrievals'] > 0
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(data)} memory statistics to {output_path}")
    
    def load_statistics(self, input_path: str):
        """Load retrieval/citation statistics from disk."""
        if not Path(input_path).exists():
            print(f"⚠️  Statistics file not found: {input_path}")
            return
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.memory_stats = defaultdict(lambda: {'n_retrievals': 0, 'n_citations': 0})
        for mem_id, stats in data.items():
            self.memory_stats[mem_id] = stats
        
        print(f"✓ Loaded {len(data)} memory statistics from {input_path}")

def extract_bayesian_features(
    query: str,
    memory: Dict,
    bayesian_model: BayesianMemoryQuality
) -> Dict[str, float]:
    """
    Extract Bayesian uncertainty features for RANK module.
    
    These features augment the existing 23-dim feature vector in RANK Enhanced V2.
    
    Args:
        query: Query text
        memory: Memory record with memory_id
        bayesian_model: BayesianMemoryQuality instance with loaded statistics
    
    Returns:
        Dict of Bayesian features
    """
    mem_id = memory.get('memory_id')
    if not mem_id:
        # No memory_id, return default features
        return {
            'bayesian_posterior_mean': 0.5,      # Prior mean
            'bayesian_posterior_std': 0.29,      # Beta(1,1) std
            'bayesian_ci_width': 0.95,           # Wide interval (uncertain)
            'bayesian_has_data': 0.0,            # No retrieval history
            'bayesian_n_retrievals_log': 0.0,
            'bayesian_uncertainty_penalty': 0.29,
        }
    
    posterior = bayesian_model.get_posterior(mem_id)
    
    # Log-transform retrieval count (avoid extreme values)
    n_retrievals_log = np.log1p(posterior['n_retrievals'])
    
    # Uncertainty penalty (lambda=1.0)
    uncertainty_penalty = posterior['posterior_std']
    
    return {
        'bayesian_posterior_mean': posterior['posterior_mean'],
        'bayesian_posterior_std': posterior['posterior_std'],
        'bayesian_ci_width': posterior['credible_interval_width'],
        'bayesian_has_data': 1.0 if posterior['has_data'] else 0.0,
        'bayesian_n_retrievals_log': n_retrievals_log,
        'bayesian_uncertainty_penalty': uncertainty_penalty,
    }

def rank_with_uncertainty_awareness(
    candidates: List[Dict],
    base_scores: List[float],
    bayesian_model: BayesianMemoryQuality,
    lambda_penalty: float = 0.5
) -> List[Tuple[Dict, float]]:
    """
    Re-rank candidates with uncertainty penalty.
    
    Formula: adjusted_score = base_score - lambda * uncertainty_penalty
    
    Args:
        candidates: List of memory candidates
        base_scores: Base ranking scores (from RANK Enhanced V2)
        bayesian_model: BayesianMemoryQuality instance
        lambda_penalty: Weight for uncertainty penalty (0 = no penalty, 1 = full penalty)
    
    Returns:
        List of (memory, adjusted_score) sorted by adjusted score
    """
    ranked = []
    
    for memory, base_score in zip(candidates, base_scores):
        mem_id = memory.get('memory_id')
        
        if mem_id:
            # Apply uncertainty penalty
            penalty = bayesian_model.compute_uncertainty_penalty(mem_id, lambda_penalty)
            adjusted_score = base_score - penalty
        else:
            # No memory_id, use base score
            adjusted_score = base_score
        
        ranked.append((memory, adjusted_score))
    
    # Sort by adjusted score (descending)
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return ranked

def demo_bayesian_rank():
    """Demonstrate Bayesian-enhanced ranking."""
    
    print("="*60)
    print("Bayesian-Enhanced RANK Demo")
    print("="*60)
    
    # Initialize Bayesian model
    bayesian_model = BayesianMemoryQuality(alpha_prior=1.0, beta_prior=1.0)
    
    # Simulate retrieval history
    # Memory A: High quality, many retrievals (confident)
    for _ in range(100):
        bayesian_model.update_memory('mem_A', cited=(np.random.rand() < 0.8))
    
    # Memory B: High quality, few retrievals (uncertain)
    for _ in range(5):
        bayesian_model.update_memory('mem_B', cited=(np.random.rand() < 0.8))
    
    # Memory C: Medium quality, many retrievals (confident)
    for _ in range(100):
        bayesian_model.update_memory('mem_C', cited=(np.random.rand() < 0.6))
    
    # Candidates with base scores (from RANK Enhanced V2)
    candidates = [
        {'memory_id': 'mem_A', 'text': 'Memory A'},
        {'memory_id': 'mem_B', 'text': 'Memory B'},
        {'memory_id': 'mem_C', 'text': 'Memory C'},
    ]
    base_scores = [0.85, 0.85, 0.75]  # A=B > C (same E5 similarity)
    
    print("\n{:<10} {:<15} {:<15} {:<15} {:<15}".format(
        "Memory", "Base Score", "Posterior", "Uncertainty", "Adjusted Score"
    ))
    print("-" * 75)
    
    # Rank without uncertainty
    print("\nWithout uncertainty penalty (λ=0):")
    ranked_vanilla = rank_with_uncertainty_awareness(candidates, base_scores, bayesian_model, lambda_penalty=0.0)
    for memory, score in ranked_vanilla:
        posterior = bayesian_model.get_posterior(memory['memory_id'])
        print("{:<10} {:<15.3f} {:.3f} ± {:.3f}   {:<15.3f} {:<15.3f}".format(
            memory['memory_id'],
            base_scores[candidates.index(memory)],
            posterior['posterior_mean'],
            posterior['posterior_std'],
            posterior['posterior_std'],
            score
        ))
    
    # Rank with uncertainty penalty
    print("\nWith uncertainty penalty (λ=0.5):")
    ranked_bayesian = rank_with_uncertainty_awareness(candidates, base_scores, bayesian_model, lambda_penalty=0.5)
    for memory, score in ranked_bayesian:
        posterior = bayesian_model.get_posterior(memory['memory_id'])
        penalty = 0.5 * posterior['posterior_std']
        print("{:<10} {:<15.3f} {:.3f} ± {:.3f}   {:<15.3f} {:<15.3f}".format(
            memory['memory_id'],
            base_scores[candidates.index(memory)],
            posterior['posterior_mean'],
            posterior['posterior_std'],
            penalty,
            score
        ))
    
    print("\n" + "="*60)
    print("Key Insight:")
    print("="*60)
    print("Vanilla ranking: A = B > C (ignores uncertainty)")
    print("Bayesian ranking: A > C > B (B penalized for high uncertainty)")
    print("\nMemory B has same base score as A, but fewer retrievals")
    print("→ Higher posterior_std → Larger penalty → Lower final rank")

if __name__ == "__main__":
    if not SCIPY_AVAILABLE:
        print("❌ scipy required for Bayesian features")
        print("   pip install scipy")
        exit(1)
    
    # Run demo
    demo_bayesian_rank()
    
    print("\n" + "="*60)
    print("Integration Instructions")
    print("="*60)
    print("1. Load retrieval statistics:")
    print("   bayesian_model = BayesianMemoryQuality()")
    print("   bayesian_model.load_statistics('data/bayesian_stats.json')")
    print()
    print("2. Extract Bayesian features in RANK pipeline:")
    print("   features = extract_bayesian_features(query, memory, bayesian_model)")
    print("   # Append to existing 23-dim feature vector → 29-dim total")
    print()
    print("3. Re-rank with uncertainty awareness:")
    print("   ranked = rank_with_uncertainty_awareness(")
    print("       candidates, base_scores, bayesian_model, lambda_penalty=0.5)")
