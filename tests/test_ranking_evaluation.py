"""
Test suite for ranking evaluation module.

Tests query-grouped ranking metrics and paired bootstrap comparison.
"""

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class TestRankingMetrics(unittest.TestCase):
    """Test query-grouped ranking metrics computation."""

    def test_single_query_single_positive(self):
        """Test NDCG/Recall for one query with one positive at different ranks."""
        from scripts.evaluate_ranking import compute_ranking_metrics

        # Positive at rank 1
        predictions = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
            {"pair_id": "q1::m3", "query_id": "q1", "label": 0, "score": 0.3},
        ]
        metrics = compute_ranking_metrics(predictions)
        
        self.assertEqual(metrics["queries"], 1)
        self.assertEqual(metrics["Recall@1"], 1.0)
        self.assertEqual(metrics["Recall@5"], 1.0)
        self.assertEqual(metrics["MRR"], 1.0)
        # NDCG@10 with 1 positive at rank 1: 1.0 / log2(2) / (1.0 / log2(2)) = 1.0
        self.assertAlmostEqual(metrics["NDCG@10"], 1.0, places=5)

    def test_single_query_positive_at_rank_3(self):
        """Test metrics when positive is at rank 3."""
        from scripts.evaluate_ranking import compute_ranking_metrics

        predictions = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 0, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.7},
            {"pair_id": "q1::m3", "query_id": "q1", "label": 1, "score": 0.5},
            {"pair_id": "q1::m4", "query_id": "q1", "label": 0, "score": 0.3},
        ]
        metrics = compute_ranking_metrics(predictions)
        
        self.assertEqual(metrics["Recall@1"], 0.0)
        self.assertEqual(metrics["Recall@5"], 1.0)
        self.assertAlmostEqual(metrics["MRR"], 1.0 / 3, places=5)
        # NDCG@10: 1.0 / log2(4) / (1.0 / log2(2))
        expected_ndcg = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(metrics["NDCG@10"], expected_ndcg, places=5)

    def test_multiple_positives(self):
        """Test NDCG with multiple positives per query."""
        from scripts.evaluate_ranking import compute_ranking_metrics

        predictions = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.7},
            {"pair_id": "q1::m3", "query_id": "q1", "label": 1, "score": 0.5},
            {"pair_id": "q1::m4", "query_id": "q1", "label": 0, "score": 0.3},
        ]
        metrics = compute_ranking_metrics(predictions)
        
        self.assertEqual(metrics["Recall@1"], 1.0)
        # DCG@10: 1/log2(2) + 1/log2(4)
        # IDCG@10: 1/log2(2) + 1/log2(3)
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected_ndcg = dcg / idcg
        self.assertAlmostEqual(metrics["NDCG@10"], expected_ndcg, places=5)

    def test_no_positives_skipped(self):
        """Test that queries with no positives are skipped."""
        from scripts.evaluate_ranking import compute_ranking_metrics

        predictions = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 0, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
            {"pair_id": "q2::m1", "query_id": "q2", "label": 1, "score": 0.8},
        ]
        metrics = compute_ranking_metrics(predictions)
        
        # Only q2 should count
        self.assertEqual(metrics["queries"], 1)
        self.assertEqual(metrics["Recall@1"], 1.0)

    def test_tie_breaking_stable(self):
        """Test that tie scores have deterministic ordering."""
        from scripts.evaluate_ranking import compute_ranking_metrics

        predictions = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.5},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
            {"pair_id": "q1::m3", "query_id": "q1", "label": 0, "score": 0.5},
        ]
        
        # Run multiple times to ensure stability
        results = [compute_ranking_metrics(predictions) for _ in range(5)]
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i])

    def test_multiple_queries(self):
        """Test metrics averaged across multiple queries."""
        from scripts.evaluate_ranking import compute_ranking_metrics

        predictions = [
            # Query 1: perfect
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
            # Query 2: miss at rank 1
            {"pair_id": "q2::m1", "query_id": "q2", "label": 0, "score": 0.9},
            {"pair_id": "q2::m2", "query_id": "q2", "label": 1, "score": 0.7},
        ]
        metrics = compute_ranking_metrics(predictions)
        
        self.assertEqual(metrics["queries"], 2)
        # Average of [1.0, 0.0]
        self.assertEqual(metrics["Recall@1"], 0.5)
        # Both have recall@5=1.0
        self.assertEqual(metrics["Recall@5"], 1.0)


class TestBootstrapComparison(unittest.TestCase):
    """Test paired bootstrap comparison."""

    def test_bootstrap_rejects_mismatched_pair_ids(self):
        """Test that bootstrap raises error when pair_id sets don't match for a query."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        # Model A has q1::m1 and q1::m2
        predictions_a = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score_a": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score_a": 0.5},
        ]
        
        # Model B has q1::m1 and q1::m3 (different candidate)
        predictions_b = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score_b": 0.95},
            {"pair_id": "q1::m3", "query_id": "q1", "label": 0, "score_b": 0.55},
        ]

        with self.assertRaises(ValueError) as context:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score_a", "score_b", n_resamples=100, seed=42
            )
        
        # Error message should be informative
        self.assertIn("pair_id", str(context.exception).lower())

    def test_bootstrap_rejects_mismatched_labels(self):
        """Test that bootstrap raises error when labels differ for same pair_id."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        predictions_a = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score_a": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score_a": 0.5},
        ]
        
        # Same pair_ids but m1 has different label
        predictions_b = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 0, "score_b": 0.95},  # label changed!
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score_b": 0.55},
        ]

        with self.assertRaises(ValueError) as context:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score_a", "score_b", n_resamples=100, seed=42
            )
        
        self.assertIn("label", str(context.exception).lower())

    def test_bootstrap_rejects_missing_score_key(self):
        """Test that bootstrap raises error when score key is missing."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        predictions_a = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score_a": 0.9},
        ]
        
        predictions_b = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1},  # Missing score_b
        ]

        with self.assertRaises((KeyError, ValueError)) as context:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score_a", "score_b", n_resamples=100, seed=42
            )

    def test_bootstrap_rejects_no_common_queries(self):
        """Test that bootstrap raises error when there are no common queries."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        predictions_a = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score_a": 0.9},
        ]
        
        predictions_b = [
            {"pair_id": "q2::m1", "query_id": "q2", "label": 1, "score_b": 0.95},
        ]

        with self.assertRaises(ValueError) as context:
            paired_bootstrap_comparison(
                predictions_a, predictions_b, "score_a", "score_b", n_resamples=100, seed=42
            )
        
        # Updated to match stricter validation message
        error_msg = str(context.exception).lower()
        self.assertTrue(
            "no common" in error_msg or "must match" in error_msg,
            msg=f"Should indicate query sets don't match. Got: {error_msg}"
        )

    def test_bootstrap_uses_local_rng(self):
        """Test that bootstrap uses local RNG and doesn't pollute global state."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        # Set global seed
        np.random.seed(12345)
        initial_state = np.random.get_state()
        
        # Generate a reference random number
        reference_before = np.random.rand()
        
        # Reset to same state
        np.random.seed(12345)
        
        # Run bootstrap (should use its own local RNG)
        predictions = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
            {"pair_id": "q2::m1", "query_id": "q2", "label": 1, "score": 0.8},
            {"pair_id": "q2::m2", "query_id": "q2", "label": 0, "score": 0.6},
        ]
        
        paired_bootstrap_comparison(
            predictions, predictions, "score", "score", n_resamples=100, seed=99
        )
        
        # After bootstrap, global RNG should produce same number as before
        reference_after = np.random.rand()
        
        # If bootstrap used global RNG, these would differ
        # If it used local RNG, they should match
        self.assertAlmostEqual(reference_before, reference_after, places=10,
                              msg="Bootstrap should use local RNG and not pollute global state")

    def test_bootstrap_seed_reproducible(self):
        """Test that bootstrap with same seed produces identical results."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        predictions_a = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score_a": 0.9},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score_a": 0.5},
            {"pair_id": "q2::m1", "query_id": "q2", "label": 1, "score_a": 0.8},
            {"pair_id": "q2::m2", "query_id": "q2", "label": 0, "score_a": 0.6},
        ]
        predictions_b = [
            {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score_b": 0.95},
            {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score_b": 0.55},
            {"pair_id": "q2::m1", "query_id": "q2", "label": 1, "score_b": 0.85},
            {"pair_id": "q2::m2", "query_id": "q2", "label": 0, "score_b": 0.65},
        ]

        result1 = paired_bootstrap_comparison(
            predictions_a, predictions_b, "score_a", "score_b", n_resamples=1000, seed=42
        )
        result2 = paired_bootstrap_comparison(
            predictions_a, predictions_b, "score_a", "score_b", n_resamples=1000, seed=42
        )

        # All bootstrap statistics should be identical
        for metric in result1:
            self.assertEqual(result1[metric]["delta"], result2[metric]["delta"])
            self.assertEqual(result1[metric]["ci95_lower"], result2[metric]["ci95_lower"])
            self.assertEqual(result1[metric]["ci95_upper"], result2[metric]["ci95_upper"])
            self.assertEqual(result1[metric]["p_b_gt_a"], result2[metric]["p_b_gt_a"])

    def test_bootstrap_model_b_better(self):
        """Test bootstrap when model B is clearly better."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        # Create predictions where B always ranks positives higher than A
        # Model A: positive has score 0.6, negative has 0.5 (positive wins but barely)
        # Model B: positive has score 0.9, negative has 0.4 (positive wins by more)
        predictions_a = []
        predictions_b = []
        for q in range(50):
            qid = f"q{q}"
            # Model A predictions
            predictions_a.extend([
                {"pair_id": f"{qid}::m1", "query_id": qid, "label": 1, "score_a": 0.6},
                {"pair_id": f"{qid}::m2", "query_id": qid, "label": 0, "score_a": 0.5},
            ])
            # Model B predictions (same pairs, different scores)
            predictions_b.extend([
                {"pair_id": f"{qid}::m1", "query_id": qid, "label": 1, "score_b": 0.9},
                {"pair_id": f"{qid}::m2", "query_id": qid, "label": 0, "score_b": 0.4},
            ])

        result = paired_bootstrap_comparison(
            predictions_a, predictions_b, "score_a", "score_b", n_resamples=3000, seed=42
        )

        # Both models achieve Recall@1=1.0, so delta should be 0
        # This test needs to be redesigned to show a real difference
        # Let's make model A fail on some queries
        predictions_a_mixed = []
        predictions_b_perfect = []
        for q in range(50):
            qid = f"q{q}"
            if q < 25:
                # First half: A ranks negative first (fails)
                predictions_a_mixed.extend([
                    {"pair_id": f"{qid}::m1", "query_id": qid, "label": 1, "score_a": 0.4},
                    {"pair_id": f"{qid}::m2", "query_id": qid, "label": 0, "score_a": 0.6},
                ])
            else:
                # Second half: A ranks positive first (succeeds)
                predictions_a_mixed.extend([
                    {"pair_id": f"{qid}::m1", "query_id": qid, "label": 1, "score_a": 0.6},
                    {"pair_id": f"{qid}::m2", "query_id": qid, "label": 0, "score_a": 0.4},
                ])
            
            # B always ranks positive first
            predictions_b_perfect.extend([
                {"pair_id": f"{qid}::m1", "query_id": qid, "label": 1, "score_b": 0.9},
                {"pair_id": f"{qid}::m2", "query_id": qid, "label": 0, "score_b": 0.3},
            ])

        result = paired_bootstrap_comparison(
            predictions_a_mixed, predictions_b_perfect, "score_a", "score_b", n_resamples=3000, seed=42
        )

        # B should be clearly better: A has Recall@1=0.5, B has Recall@1=1.0
        self.assertGreater(result["Recall@1"]["delta"], 0.4)  # delta should be ~0.5
        self.assertGreater(result["Recall@1"]["ci95_lower"], 0.3)
        self.assertGreater(result["Recall@1"]["p_b_gt_a"], 0.99)

    def test_bootstrap_identical_models(self):
        """Test bootstrap when models produce identical rankings."""
        from scripts.evaluate_ranking import paired_bootstrap_comparison

        predictions = []
        for q in range(30):
            qid = f"q{q}"
            predictions.extend([
                {"pair_id": f"{qid}::m1", "query_id": qid, "label": 1, "score": 0.8},
                {"pair_id": f"{qid}::m2", "query_id": qid, "label": 0, "score": 0.5},
            ])

        result = paired_bootstrap_comparison(
            predictions, predictions, "score", "score", n_resamples=1000, seed=42
        )

        # Delta should be zero (identical models)
        self.assertAlmostEqual(result["Recall@1"]["delta"], 0.0, places=10)
        self.assertAlmostEqual(result["NDCG@10"]["delta"], 0.0, places=10)
        # When deltas are all exactly 0, p_b_gt_a should be 0 (no sample where B > A)
        self.assertEqual(result["Recall@1"]["p_b_gt_a"], 0.0)
        self.assertEqual(result["NDCG@10"]["p_b_gt_a"], 0.0)


class TestEvaluationCLI(unittest.TestCase):
    """Test the evaluate_ranking.py CLI."""

    def test_cli_with_predictions_file(self):
        """Test CLI reads predictions and outputs metrics."""
        from scripts.evaluate_ranking import main

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create test predictions
            predictions_path = tmpdir_path / "predictions.jsonl"
            with predictions_path.open("w", encoding="utf-8") as f:
                test_data = [
                    {"pair_id": "q1::m1", "query_id": "q1", "label": 1, "score": 0.9},
                    {"pair_id": "q1::m2", "query_id": "q1", "label": 0, "score": 0.5},
                    {"pair_id": "q2::m1", "query_id": "q2", "label": 1, "score": 0.8},
                    {"pair_id": "q2::m2", "query_id": "q2", "label": 0, "score": 0.7},
                ]
                for pred in test_data:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")

            output_path = tmpdir_path / "metrics.json"

            # Run CLI
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "evaluate_ranking.py",
                    "--predictions", str(predictions_path),
                    "--score-key", "score",
                    "--output", str(output_path),
                    "--seed", "42",
                ]
                main()
            finally:
                sys.argv = original_argv

            # Check output exists and is valid
            self.assertTrue(output_path.exists())
            with output_path.open(encoding="utf-8") as f:
                metrics = json.load(f)
            
            self.assertIn("metrics", metrics)
            self.assertEqual(metrics["metrics"]["queries"], 2)
            self.assertIn("Recall@1", metrics["metrics"])
            self.assertIn("NDCG@10", metrics["metrics"])


if __name__ == "__main__":
    unittest.main()
