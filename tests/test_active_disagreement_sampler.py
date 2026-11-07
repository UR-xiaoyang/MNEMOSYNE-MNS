#!/usr/bin/env python3
"""
Tests for active_disagreement_sampler.py - P3 high-value sample selection.

TDD RED phase: These tests should FAIL until the script is implemented.
"""

import json
import tempfile
import unittest
from pathlib import Path


class TestDisagreementSampling(unittest.TestCase):
    """Test disagreement-based sample selection."""

    def test_samples_by_baseline_candidate_disagreement(self):
        """RED: Must identify samples where baseline and candidate rankings differ."""
        from scripts.active_disagreement_sampler import sample_disagreements
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.jsonl"
            candidate_file = Path(tmpdir) / "candidate.jsonl"
            
            # Same candidates, different rankings
            baseline = [
                {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.9, "rank": 1},
                {"query_id": "q1", "pair_id": "q1::m2", "label": 0, "score": 0.8, "rank": 2},
            ]
            candidate = [
                {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.7, "rank": 2},  # Disagreement!
                {"query_id": "q1", "pair_id": "q1::m2", "label": 0, "score": 0.9, "rank": 1},
            ]
            
            with open(baseline_file, "w") as f:
                for pred in baseline:
                    f.write(json.dumps(pred) + "\n")
            
            with open(candidate_file, "w") as f:
                for pred in candidate:
                    f.write(json.dumps(pred) + "\n")
            
            samples = sample_disagreements(
                baseline_predictions=str(baseline_file),
                candidate_predictions=str(candidate_file),
                n_samples=10,
            )
            
            # Should find the disagreement
            self.assertGreater(len(samples), 0)
            self.assertIn("reason", samples[0])
            self.assertIn("query_id", samples[0])

    def test_output_contains_ids_not_text(self):
        """RED: Output must contain sample IDs and reason, not query/memory text."""
        from scripts.active_disagreement_sampler import sample_disagreements
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.jsonl"
            candidate_file = Path(tmpdir) / "candidate.jsonl"
            
            predictions = [
                {
                    "query_id": "q1",
                    "pair_id": "q1::m1",
                    "label": 1,
                    "score": 0.8,
                    "rank": 1,
                    "query_text": "sensitive query content",
                    "memory_text": "sensitive memory content",
                }
            ]
            
            with open(baseline_file, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
            
            with open(candidate_file, "w") as f:
                for pred in predictions:
                    pred_copy = pred.copy()
                    pred_copy["score"] = 0.95  # Larger difference to avoid floating point issues
                    f.write(json.dumps(pred_copy) + "\n")
            
            samples = sample_disagreements(
                baseline_predictions=str(baseline_file),
                candidate_predictions=str(candidate_file),
                n_samples=10,
            )
            
            # Serialize to check content
            samples_str = json.dumps(samples)
            
            # Should have IDs
            self.assertIn("q1", samples_str)
            
            # Should NOT have text content
            self.assertNotIn("sensitive query content", samples_str)
            self.assertNotIn("sensitive memory content", samples_str)

    def test_output_contains_no_paths(self):
        """RED: Output must not contain file paths."""
        from scripts.active_disagreement_sampler import sample_disagreements
        
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / "baseline.jsonl"
            candidate_file = Path(tmpdir) / "candidate.jsonl"
            
            predictions = [
                {"query_id": "q1", "pair_id": "q1::m1", "label": 1, "score": 0.8}
            ]
            
            for path in [baseline_file, candidate_file]:
                with open(path, "w") as f:
                    for pred in predictions:
                        f.write(json.dumps(pred) + "\n")
            
            samples = sample_disagreements(
                baseline_predictions=str(baseline_file),
                candidate_predictions=str(candidate_file),
                n_samples=10,
            )
            
            samples_str = json.dumps(samples)
            
            # No paths
            self.assertNotIn("C:\\", samples_str)
            self.assertNotIn("/home/", samples_str)
            self.assertNotIn("/Users/", samples_str)


class TestDisagreementReasons(unittest.TestCase):
    """Test disagreement reason categorization."""

    def test_categorizes_low_confidence(self):
        """RED: Must identify low-confidence predictions."""
        from scripts.active_disagreement_sampler import categorize_disagreement
        
        sample = {
            "query_id": "q1",
            "pair_id": "q1::m1",
            "baseline_score": 0.51,  # Low confidence
            "candidate_score": 0.49,
            "baseline_rank": 5,
            "candidate_rank": 6,
        }
        
        reason = categorize_disagreement(sample)
        
        self.assertIn("low_confidence", reason.lower())

    def test_categorizes_rank_flip(self):
        """RED: Must identify large rank changes."""
        from scripts.active_disagreement_sampler import categorize_disagreement
        
        sample = {
            "query_id": "q1",
            "pair_id": "q1::m1",
            "baseline_score": 0.9,
            "candidate_score": 0.2,
            "baseline_rank": 1,
            "candidate_rank": 10,  # Large flip
        }
        
        reason = categorize_disagreement(sample)
        
        self.assertIn("rank", reason.lower())


if __name__ == "__main__":
    unittest.main()
