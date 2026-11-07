"""
Test suite for dataset validation contract.

Tests that reject:
- Missing trace/session IDs
- Candidates temporally after query
- Session split leakage
- Unredacted sensitive fields
- Evaluation-only data in training
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestDatasetSchemaValidation(unittest.TestCase):
    """Test basic schema validation."""

    def test_reject_missing_trace_id(self):
        """Reject samples without trace ID."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {"session_id": "s1", "query_id": "q1", "timestamp": 100, "label": 1},
            # Missing trace_id
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertFalse(result["valid"])
        self.assertIn("missing_trace_id", result["errors"])

    def test_reject_missing_session_id(self):
        """Reject samples without session ID."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {"trace_id": "t1", "query_id": "q1", "timestamp": 100, "label": 1},
            # Missing session_id
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertFalse(result["valid"])
        self.assertIn("missing_session_id", result["errors"])

    def test_accept_valid_schema(self):
        """Accept samples with all required fields."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "memory_id": "m1",
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)


class TestTemporalOrdering(unittest.TestCase):
    """Test temporal ordering constraints."""

    def test_reject_future_candidate(self):
        """Reject candidates with timestamp after query."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "memory_id": "m1",
                "query_timestamp": 100,
                "candidate_timestamp": 200,  # Future candidate
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertFalse(result["valid"])
        self.assertIn("temporal_violation", result["errors"])

    def test_accept_past_candidate(self):
        """Accept candidates with timestamp before query."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "memory_id": "m1",
                "query_timestamp": 100,
                "candidate_timestamp": 50,  # Past candidate
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertTrue(result["valid"])


class TestSplitIsolation(unittest.TestCase):
    """Test session split isolation."""

    def test_reject_session_in_multiple_splits(self):
        """Reject when same session appears in train and test."""
        from scripts.validate_dataset import validate_split_isolation

        train_samples = [
            {"session_id": "s1", "query_id": "q1"},
            {"session_id": "s2", "query_id": "q2"},
        ]
        
        test_samples = [
            {"session_id": "s1", "query_id": "q3"},  # s1 also in train!
            {"session_id": "s3", "query_id": "q4"},
        ]
        
        result = validate_split_isolation(
            {"train": train_samples, "test": test_samples}
        )
        self.assertFalse(result["valid"])
        self.assertIn("split_leakage", result["errors"])
        self.assertIn("s1", result["leaked_sessions"])

    def test_accept_isolated_splits(self):
        """Accept when sessions are properly isolated."""
        from scripts.validate_dataset import validate_split_isolation

        train_samples = [
            {"session_id": "s1", "query_id": "q1"},
            {"session_id": "s2", "query_id": "q2"},
        ]
        
        test_samples = [
            {"session_id": "s3", "query_id": "q3"},
            {"session_id": "s4", "query_id": "q4"},
        ]
        
        result = validate_split_isolation(
            {"train": train_samples, "test": test_samples}
        )
        self.assertTrue(result["valid"])


class TestSensitiveDataRedaction(unittest.TestCase):
    """Test sensitive field detection."""

    def test_reject_unredacted_api_key(self):
        """Reject samples containing API keys."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "content": "My API key is sk-1234567890abcdef",
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertFalse(result["valid"])
        self.assertIn("sensitive_data", result["errors"])

    def test_reject_unredacted_bearer_token(self):
        """Reject samples containing bearer tokens."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "content": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertFalse(result["valid"])
        self.assertIn("sensitive_data", result["errors"])

    def test_reject_unredacted_email(self):
        """Reject samples containing email addresses."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "content": "Contact me at user@example.com",
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertFalse(result["valid"])
        self.assertIn("sensitive_data", result["errors"])

    def test_accept_redacted_content(self):
        """Accept samples with properly redacted content."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "content": "My API key is [REDACTED_API_KEY]",
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking")
        self.assertTrue(result["valid"])


class TestEvaluationBoundary(unittest.TestCase):
    """Test evaluation-only data exclusion from training."""

    def test_reject_evaluation_data_in_training(self):
        """Reject when evaluation-only data appears in training."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "dataset": "evaluation_only",  # Should not be in training
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking", split="train")
        self.assertFalse(result["valid"])
        self.assertIn("evaluation_data_in_training", result["errors"])

    def test_accept_public_data_in_training(self):
        """Accept public data in training split."""
        from scripts.validate_dataset import validate_dataset_contract

        samples = [
            {
                "trace_id": "t1",
                "session_id": "s1",
                "query_id": "q1",
                "dataset": "public",
                "timestamp": 100,
                "label": 1,
            }
        ]
        
        result = validate_dataset_contract(samples, dataset_type="ranking", split="train")
        self.assertTrue(result["valid"])


class TestValidationCLI(unittest.TestCase):
    """Test the validation CLI."""

    def test_cli_validates_and_reports(self):
        """Test CLI reads data and produces validation report."""
        from scripts.validate_dataset import main

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create invalid test data
            data_path = tmpdir_path / "test_data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                # Missing session_id
                f.write(json.dumps({"trace_id": "t1", "query_id": "q1", "timestamp": 100}) + "\n")
            
            report_path = tmpdir_path / "report.json"
            
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "validate_dataset.py",
                    "--input", str(data_path),
                    "--dataset-type", "ranking",
                    "--output", str(report_path),
                ]
                # Expect SystemExit(1) due to validation failure
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
            finally:
                sys.argv = original_argv
            
            # Check report exists and shows failure
            self.assertTrue(report_path.exists())
            with report_path.open(encoding="utf-8") as f:
                report = json.load(f)
            
            self.assertFalse(report["valid"])
            self.assertGreater(len(report["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
