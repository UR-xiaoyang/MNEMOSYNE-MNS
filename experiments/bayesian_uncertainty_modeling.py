#!/usr/bin/env python3
"""
Bayesian Uncertainty Modeling for MNEMOSYNE Memory Quality.

Goal: Model memory quality as probability distributions rather than point estimates.

Benefits:
1. Quantify uncertainty ("70% confident this memory is correct")
2. Small data friendly (Bayesian priors handle sparse data)
3. Natural handling of corrections (Bayesian updates)
4. Interpretable (posterior distributions are intuitive)

Use Cases:
- Memory confidence scoring (for ASSEMBLE context selection)
- Correction risk prediction (for AUDIT alerts)
- Citation success prediction (for RANK features)
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Try to import Bayesian libraries
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("⚠️  PyMC not installed. Run: pip install pymc arviz")

# Fallback: scipy.stats for simple Beta distributions
try:
    from scipy.stats import beta as beta_dist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not installed. Run: pip install scipy")

def load_shadow_traces(data_dir: str) -> List[dict]:
    """Load shadow trace data."""
    data_path = Path(data_dir) / "shadow_training_samples.jsonl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Shadow data not found: {data_path}")
    
    traces = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            traces.append(json.loads(line))
    
    return traces

class BetaBernoulliMemoryQuality:
    """
    Simple Bayesian model for memory quality using Beta-Bernoulli conjugacy.
    
    Model:
        quality ~ Beta(α, β)                      # Prior belief
        citations ~ Binomial(n_retrievals, quality)  # Likelihood
        
    Posterior (after observing data):
        quality | data ~ Beta(α + citations, β + (n_retrievals - citations))
    
    Intuition:
        α = "pseudo-citations" (prior successes)
        β = "pseudo-failures" (prior failures)
        α + β = strength of prior belief
    """
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize with prior hyperparameters.
        
        Args:
            alpha_prior: Prior pseudo-citations (default 1.0 = uninformative)
            beta_prior: Prior pseudo-failures (default 1.0 = uninformative)
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
    
    def fit_memory(self, n_retrievals: int, n_citations: int) -> Dict:
        """
        Compute posterior distribution for a single memory.
        
        Args:
            n_retrievals: Total times memory was retrieved
            n_citations: Times memory was actually cited by user
        
        Returns:
            dict with posterior statistics
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required for Beta distribution")
        
        # Posterior parameters (Beta-Bernoulli conjugacy)
        alpha_post = self.alpha_prior + n_citations
        beta_post = self.beta_prior + (n_retrievals - n_citations)
        
        # Posterior distribution
        posterior = beta_dist(alpha_post, beta_post)
        
        # Statistics
        mean_quality = posterior.mean()  # E[quality]
        variance = posterior.var()       # Var[quality]
        std_dev = np.sqrt(variance)
        
        # Credible interval (Bayesian confidence interval)
        ci_95_lower, ci_95_upper = posterior.ppf(0.025), posterior.ppf(0.975)
        
        # Probability quality > threshold
        prob_good = 1 - posterior.cdf(0.5)  # P(quality > 0.5)
        prob_excellent = 1 - posterior.cdf(0.8)  # P(quality > 0.8)
        
        return {
            "n_retrievals": n_retrievals,
            "n_citations": n_citations,
            "empirical_rate": n_citations / n_retrievals if n_retrievals > 0 else 0,
            "posterior_mean": mean_quality,
            "posterior_std": std_dev,
            "credible_interval_95": (ci_95_lower, ci_95_upper),
            "prob_quality_good": prob_good,
            "prob_quality_excellent": prob_excellent,
            "alpha_posterior": alpha_post,
            "beta_posterior": beta_post,
        }
    
    def predict_next_citation_prob(self, n_retrievals: int, n_citations: int) -> float:
        """
        Predict probability next retrieval results in citation.
        
        This is the posterior predictive distribution:
            P(next citation | data) = E[quality | data] = α/(α+β)
        """
        alpha_post = self.alpha_prior + n_citations
        beta_post = self.beta_prior + (n_retrievals - n_citations)
        
        return alpha_post / (alpha_post + beta_post)

class HierarchicalMemoryQuality:
    """
    Hierarchical Bayesian model for memory quality across different memory types.
    
    Model:
        μ, σ ~ Hyperpriors (global quality distribution)
        quality_i ~ Beta(μ, σ) for each memory i
        citations_i ~ Binomial(n_i, quality_i)
    
    Benefits:
        - Borrows strength across memories (partial pooling)
        - Handles sparse data better than independent models
        - Discovers memory clusters automatically
    """
    
    def __init__(self):
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required for hierarchical models")
        
        self.model = None
        self.trace = None
    
    def fit(self, memory_data: List[Dict], n_samples: int = 2000):
        """
        Fit hierarchical model to multiple memories.
        
        Args:
            memory_data: List of {memory_id, n_retrievals, n_citations, scope_level}
            n_samples: MCMC samples per chain
        """
        n_memories = len(memory_data)
        n_retrievals = np.array([m['n_retrievals'] for m in memory_data])
        n_citations = np.array([m['n_citations'] for m in memory_data])
        
        with pm.Model() as self.model:
            # Hyperpriors (global quality distribution)
            μ = pm.Beta("μ", alpha=2, beta=2)  # Global mean quality
            κ = pm.Gamma("κ", alpha=2, beta=0.1)  # Concentration (inverse variance)
            
            # Memory-specific quality (hierarchical)
            # Reparameterize Beta: α = μκ, β = (1-μ)κ
            quality = pm.Beta("quality", alpha=μ * κ, beta=(1 - μ) * κ, shape=n_memories)
            
            # Likelihood
            citations = pm.Binomial("citations", n=n_retrievals, p=quality, observed=n_citations)
            
            # Sample posterior
            print(f"Sampling {n_samples} iterations with NUTS sampler...")
            self.trace = pm.sample(n_samples, tune=1000, return_inferencedata=True, progressbar=True)
        
        return self.trace
    
    def get_memory_quality_posterior(self, memory_idx: int) -> Dict:
        """Get posterior statistics for specific memory."""
        if self.trace is None:
            raise RuntimeError("Model not fitted yet")
        
        quality_samples = self.trace.posterior["quality"].sel(quality_dim_0=memory_idx).values.flatten()
        
        return {
            "posterior_mean": np.mean(quality_samples),
            "posterior_std": np.std(quality_samples),
            "credible_interval_95": (np.percentile(quality_samples, 2.5), np.percentile(quality_samples, 97.5)),
            "median": np.median(quality_samples),
        }

def compare_frequentist_vs_bayesian(traces: List[dict]):
    """Compare frequentist point estimates vs Bayesian distributions."""
    
    print("\n" + "="*60)
    print("Frequentist vs Bayesian Comparison")
    print("="*60)
    
    # Aggregate data by memory_id
    memory_stats = defaultdict(lambda: {"retrievals": 0, "citations": 0})
    
    for trace in traces:
        mem_id = trace.get("memory_id")
        if mem_id and trace.get("event_type") == "retrieve":
            memory_stats[mem_id]["retrievals"] += 1
            if trace.get("label") == "citation":
                memory_stats[mem_id]["citations"] += 1
    
    # Filter memories with at least 5 retrievals
    qualified_memories = {
        mem_id: stats for mem_id, stats in memory_stats.items()
        if stats["retrievals"] >= 5
    }
    
    if len(qualified_memories) < 10:
        print(f"⚠️  Only {len(qualified_memories)} memories with ≥5 retrievals")
        print("Using all available memories...")
        qualified_memories = memory_stats
    
    print(f"\nAnalyzing {len(qualified_memories)} memories...")
    
    # Initialize Bayesian model
    bayesian_model = BetaBernoulliMemoryQuality(alpha_prior=1.0, beta_prior=1.0)
    
    # Compare for sample memories
    sample_memories = list(qualified_memories.items())[:10]
    
    print("\n{:<15} {:<10} {:<10} {:<15} {:<25}".format(
        "Memory ID", "Retrievals", "Citations", "Frequentist", "Bayesian (Mean ± Std)"
    ))
    print("-" * 85)
    
    for mem_id, stats in sample_memories:
        n_ret = stats["retrievals"]
        n_cit = stats["citations"]
        
        # Frequentist: MLE point estimate
        freq_estimate = n_cit / n_ret if n_ret > 0 else 0
        
        # Bayesian: Posterior distribution
        posterior = bayesian_model.fit_memory(n_ret, n_cit)
        
        print("{:<15} {:<10} {:<10} {:<15.3f} {:.3f} ± {:.3f}".format(
            mem_id[:12], n_ret, n_cit, freq_estimate,
            posterior["posterior_mean"], posterior["posterior_std"]
        ))
    
    # Key insight: Bayesian shrinks extreme estimates toward prior
    print("\n" + "="*60)
    print("Key Insights:")
    print("="*60)
    print("1. Frequentist: 1/1 citations → 100% quality (overconfident)")
    print("2. Bayesian: 1/1 citations → ~67% quality (shrunk toward prior)")
    print("3. As data grows, Bayesian → Frequentist (prior becomes negligible)")
    print("4. Bayesian provides uncertainty: low std = high confidence")

def experiment_uncertainty_guided_ranking():
    """
    Demonstrate how uncertainty can improve RANK module.
    
    Idea: Penalize memories with high uncertainty in ranking.
    """
    print("\n" + "="*60)
    print("Uncertainty-Guided Ranking Experiment")
    print("="*60)
    
    # Synthetic example
    memories = [
        {"id": "A", "n_retrievals": 100, "n_citations": 70, "e5_score": 0.85},
        {"id": "B", "n_retrievals": 5, "n_citations": 4, "e5_score": 0.85},  # High uncertainty
        {"id": "C", "n_retrievals": 50, "n_citations": 35, "e5_score": 0.82},
    ]
    
    bayesian_model = BetaBernoulliMemoryQuality(alpha_prior=1.0, beta_prior=1.0)
    
    print("\n{:<5} {:<12} {:<10} {:<15} {:<20} {:<15}".format(
        "ID", "Retrievals", "Citations", "E5 Score", "Bayesian Quality", "Uncertainty"
    ))
    print("-" * 85)
    
    for mem in memories:
        posterior = bayesian_model.fit_memory(mem["n_retrievals"], mem["n_citations"])
        
        print("{:<5} {:<12} {:<10} {:<15.3f} {:.3f} (CI: {:.2f}-{:.2f})  {:.3f}".format(
            mem["id"], mem["n_retrievals"], mem["n_citations"], mem["e5_score"],
            posterior["posterior_mean"],
            posterior["credible_interval_95"][0],
            posterior["credible_interval_95"][1],
            posterior["posterior_std"]
        ))
    
    print("\n" + "="*60)
    print("Ranking Strategy:")
    print("="*60)
    print("Traditional: Rank by E5 score only")
    print("  → A=B > C (memory B ranked high despite uncertainty)")
    print()
    print("Bayesian: Rank by E5 * (posterior_mean - λ * posterior_std)")
    print("  λ = uncertainty penalty (e.g., λ=1.0)")
    print("  → A > C > B (memory B penalized for uncertainty)")

def run_experiment(data_dir: str = "data/shadow_export"):
    """Run full Bayesian uncertainty experiment."""
    
    print("="*60)
    print("Bayesian Uncertainty Modeling Experiment")
    print("="*60)
    
    # Load shadow traces
    print(f"\n1. Loading shadow trace data from {data_dir}...")
    try:
        traces = load_shadow_traces(data_dir)
        print(f"   ✓ Loaded {len(traces)} shadow trace events")
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        print("\nPlease ensure shadow data exists:")
        print("  python scripts/export_shadow_data.py")
        return
    
    # Experiment 1: Frequentist vs Bayesian
    if SCIPY_AVAILABLE:
        compare_frequentist_vs_bayesian(traces)
    else:
        print("\n⚠️  Skipping Frequentist vs Bayesian (scipy not available)")
    
    # Experiment 2: Uncertainty-guided ranking
    if SCIPY_AVAILABLE:
        experiment_uncertainty_guided_ranking()
    else:
        print("\n⚠️  Skipping uncertainty-guided ranking (scipy not available)")
    
    # Experiment 3: Hierarchical model (optional, slow)
    if PYMC_AVAILABLE:
        print("\n" + "="*60)
        print("Optional: Hierarchical Bayesian Model")
        print("="*60)
        print("⚠️  This requires PyMC and takes ~5 minutes to run.")
        print("Run with: python experiments/bayesian_uncertainty_modeling.py --hierarchical")
    
    # Save results
    output_dir = Path("experiments/bayesian_uncertainty")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "bayesian_uncertainty_modeling",
        "dataset": data_dir,
        "n_traces": len(traces),
        "scipy_available": SCIPY_AVAILABLE,
        "pymc_available": PYMC_AVAILABLE,
        "conclusion": "Bayesian methods provide uncertainty quantification for memory quality",
        "benefits": [
            "Small data friendly (priors handle sparse data)",
            "Uncertainty quantification (credible intervals)",
            "Natural correction handling (Bayesian updates)",
            "Interpretable (probability of quality > threshold)",
        ],
        "next_steps": [
            "Integrate uncertainty into RANK features",
            "Use credible intervals for AUDIT alerts",
            "Explore hierarchical models for memory clustering",
        ],
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/results.json")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("✓ Bayesian uncertainty modeling provides:")
    print("  1. Probability distributions instead of point estimates")
    print("  2. Confidence intervals for memory quality")
    print("  3. Natural handling of sparse data (priors)")
    print("  4. Uncertainty-aware ranking (penalize uncertain memories)")
    print()
    print("Next steps:")
    print("  1. Add uncertainty features to RANK Enhanced V3")
    print("  2. Use credible intervals for AUDIT risk scoring")
    print("  3. Deploy uncertainty-guided memory selection in ASSEMBLE")

if __name__ == "__main__":
    import sys
    
    # Check if shadow data exists
    data_dir = "data/shadow_export"
    if not Path(data_dir).exists():
        print(f"❌ Shadow data not found: {data_dir}")
        print("\nPlease export shadow data first:")
        print("  # Option 1: Use existing export script")
        print("  python scripts/export_shadow_data.py")
        print()
        print("  # Option 2: Create mock data for testing")
        print("  mkdir -p data/shadow_export")
        print("  echo '{\"memory_id\":\"test\",\"event_type\":\"retrieve\",\"label\":\"citation\"}' > data/shadow_export/shadow_training_samples.jsonl")
        sys.exit(1)
    
    # Run experiment
    run_experiment(data_dir)
    
    # Hierarchical model (optional)
    if "--hierarchical" in sys.argv and PYMC_AVAILABLE:
        print("\n" + "="*60)
        print("Running Hierarchical Bayesian Model...")
        print("="*60)
        print("(This may take 5-10 minutes)")
        # TODO: Implement hierarchical model experiment
