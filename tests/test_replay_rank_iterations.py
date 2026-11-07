#!/usr/bin/env python3
"""
Tests for replay_rank_iterations.py - P3 offline baseline vs candidate comparison.

TDD RED phase: These tests should FAIL until the script is implemented.
"""

import json
import tempfile
import unittest
from pathlib import Path


class TestReplayRankIterations(unittest.TestCase):
    """Test offline replay of baseline vs candidate models."""

    def test_replay_requires_identical_query_pools(self):
        """RED: Must validate baseline and candidate have same query sets."""
        from scripts.replay_rank_iterations import replay_and_compare
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.jsonl"
            candidate_file = Path(tmpdir) / "candidate.jsonl"
            
            # Different query sets
            with open(baseline_file, "w") as f:
                f.write(json.dumps({
                    "query_id": "q1",
                    "pair_id": "q1::m1",
                    "label": 1,
                    "score": 0.8,
                }) + "\n")
            
            with open(candidate_file, "w") as f:
                f.write(json.dumps({
                    "query_id": "q2",  # Different query!
                    "pair_id": "q2::m1",
                    "label": 1,
                    "score": 0.9,
                }) + "\n")
            
            with self.assertRaises(ValueError) as ctx:
                replay_and_compare(
                    baseline_predictions=str(baseline_file),
                    candidate_predictions=str(candidate_file),
                )
            
            self.assertIn("query", str(ctx.exception).lower())

    def test_replay_computes_query_level_metrics(self):
        """RED: Must compute Recall@k, MRR, NDCG per query."""
        from scripts.replay_rank_iterations import replay_and_compare
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.jsonl"
            candidate_file = Path(tmpdir) / "candidate.jsonl"
            
            # Same query pool, different rankings
            baseline_preds = [
                {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.8},
                {"query_id": "q1", "pair_id": "q1::m2", "label": 0, "score": 0.7},
            ]
            candidate_preds = [
                {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.9},
                {"query_id": "q1", "pair_id": "q1::m2", "label": 0, "score": 0.6},
            ]
            
            with open(baseline_file, "w") as f:
                for pred in baseline_preds:
                    f.write(json.dumps(pred) + "\n")
            
            with open(candidate_file, "w") as f:
                for pred in candidate_preds:
                    f.write(json.dumps(pred) + "\n")
            
            result = replay_and_compare(
                baseline_predictions=str(baseline_file),
                candidate_predictions=str(candidate_file),
            )
            
            # Check metrics exist
            self.assertIn("baseline_metrics", result)
            self.assertIn("candidate_metrics", result)
            self.assertIn("Recall@1", result["baseline_metrics"])
            self.assertIn("NDCG@10", result["baseline_metrics"])

    def test_replay_uses_paired_bootstrap(self):
        """RED: Must use existing evaluate_ranking.py paired bootstrap."""
        from scripts.replay_rank_iterations import replay_and_compare
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.jsonl"
            candidate_file = Path(tmpdir) / "candidate.jsonl"
            
            # Create toy predictions
            predictions = [
                {"query_id": f"q{i}", "pair_id": f"q{i}::m1", "label": 1, "score": 0.8}
                for i in range(10)
            ]
            
            with open(baseline_file, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
            
            # Candidate slightly better
            with open(candidate_file, "w") as f:
                for pred in predictions:
                    pred_copy = pred.copy()
                    pred_copy["score"] = 0.85
                    f.write(json.dumps(pred_copy) + "\n")
            
            result = replay_and_compare(
                baseline_predictions=str(baseline_file),
                candidate_predictions=str(candidate_file),
                n_bootstrap=100,  # Small for test speed
            )
            
            # Check confidence intervals exist
            self.assertIn("comparison", result)
            self.assertIn("NDCG@10", result["comparison"])
            self.assertIn("ci95_lower", result["comparison"]["NDCG@10"])
            self.assertIn("ci95_upper", result["comparison"]["NDCG@10"])


class TestReplayValidation(unittest.TestCase):
    """Test validation of candidate pools."""

    def test_validates_pair_id_uniqueness_per_query(self):
        """RED: Each (query_id, pair_id) must be unique."""
        from scripts.replay_rank_iterations import validate_predictions
        
        predictions = [
            {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.8},
            {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.9},  # Duplicate!
        ]
        
        with self.assertRaises(ValueError) as ctx:
            validate_predictions(predictions)
        
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_validates_pair_id_prefix_matches_query(self):
        """RED: pair_id prefix must match query_id."""
        from scripts.replay_rank_iterations import validate_predictions
        
        predictions = [
            {"query_id": "q1", "pair_id": "q2::m1", "label": 1, "score": 0.8},  # Mismatch!
        ]
        
        with self.assertRaises(ValueError) as ctx:
            validate_predictions(predictions)
        
        self.assertIn("prefix", str(ctx.exception).lower())


class TestReplayPrivacy(unittest.TestCase):
    """Test privacy boundaries in replay output."""

    def test_output_contains_no_paths(self):
        """RED: Comparison result must not contain file paths."""
        from scripts.replay_rank_iterations import replay_and_compare
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.jsonl"
            candidate_file = Path(tmpdir) / "candidate.jsonl"
            
            predictions = [
                {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.8}
            ]
            
            with open(baseline_file, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
            
            with open(candidate_file, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
            
            result = replay_and_compare(
                baseline_predictions=str(baseline_file),
                candidate_predictions=str(candidate_file),
            )
            
            result_str = json.dumps(result)
            
            # No absolute paths
            self.assertNotIn("C:\\", result_str)
            self.assertNotIn("/home/", result_str)
            self.assertNotIn("/Users/", result_str)


if __name__ == "__main__":
    unittest.main()
