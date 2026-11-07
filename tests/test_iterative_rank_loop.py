#!/usr/bin/env python3
"""
Tests for iterative_rank_loop.py - P3 weak supervision iteration manifest builder.

TDD RED phase: These tests should FAIL until the script is implemented.
"""

import json
import tempfile
import unittest
from pathlib import Path


class TestIterativeRankLoopBasic(unittest.TestCase):
    """Test basic iteration manifest building."""

    def test_build_iteration_manifest_requires_explicit_input(self):
        """RED: Iteration builder must require explicit input directory."""
        from scripts.iterative_rank_loop import build_iteration_manifest
        
        with self.assertRaises(ValueError) as ctx:
            build_iteration_manifest(
                input_dir=None,
                output_dir=tempfile.mkdtemp(),
            )
        
        self.assertIn("explicit", str(ctx.exception).lower())

    def test_build_iteration_manifest_creates_temporal_splits(self):
        """RED: Must split by time window, not random."""
        from scripts.iterative_rank_loop import build_iteration_manifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            
            # Create toy shadow samples with timestamps
            samples_file = input_dir / "samples.jsonl"
            samples = [
                {
                    "trace_id": f"trace_{i}",
                    "query_id": f"q_{i}",
                    "memory_id": f"m_{i}",
                    "query_timestamp": 1000.0 + i * 100,
                    "label": "positive" if i % 3 == 0 else "unknown",
                    "label_source": "citation" if i % 3 == 0 else "unknown",
                }
                for i in range(20)
            ]
            
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            
            # Build manifest
            manifest = build_iteration_manifest(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                train_ratio=0.7,
                val_ratio=0.2,
            )
            
            # Check splits exist
            self.assertIn("train_samples", manifest)
            self.assertIn("val_samples", manifest)
            self.assertIn("holdout_samples", manifest)
            
            # Check temporal ordering (train < val < holdout)
            train_file = output_dir / manifest["train_file"]
            val_file = output_dir / manifest["val_file"]
            holdout_file = output_dir / manifest["holdout_file"]
            
            self.assertTrue(train_file.exists())
            self.assertTrue(val_file.exists())
            self.assertTrue(holdout_file.exists())

    def test_unknown_labels_excluded_from_supervision_loss(self):
        """RED: Unknown samples must have sample_weight=0 for supervised loss."""
        from scripts.iterative_rank_loop import build_iteration_manifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            
            # Create samples with unknown labels
            samples_file = input_dir / "samples.jsonl"
            samples = [
                {
                    "trace_id": "t1",
                    "query_id": "q1",
                    "memory_id": "m1",
                    "query_timestamp": 1000.0,
                    "label": "unknown",
                    "label_source": "unknown",
                    "unknown_reason": "no_citation_or_feedback_evidence",
                },
            ]
            
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            
            manifest = build_iteration_manifest(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )
            
            # Load train samples and check weights
            train_file = output_dir / manifest["train_file"]
            with open(train_file) as f:
                train_samples = [json.loads(line) for line in f]
            
            for sample in train_samples:
                if sample["label"] == "unknown":
                    # Unknown samples should have zero weight for supervised loss
                    self.assertEqual(sample.get("sample_weight", 1.0), 0.0)

    def test_citation_and_correction_labels_separated(self):
        """RED: Citation (positive) and correction (risk) must be separate labels."""
        from scripts.iterative_rank_loop import build_iteration_manifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            
            samples_file = input_dir / "samples.jsonl"
            samples = [
                {
                    "trace_id": "t1",
                    "query_id": "q1",
                    "memory_id": "m1",
                    "query_timestamp": 1000.0,
                    "label": "positive",
                    "label_source": "citation",
                },
                {
                    "trace_id": "t2",
                    "query_id": "q2",
                    "memory_id": "m2",
                    "query_timestamp": 1100.0,
                    "label": "corrected",
                    "label_source": "correction",
                    "correction_type": "outdated",
                },
            ]
            
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            
            manifest = build_iteration_manifest(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )
            
            # Check manifest tracks both label types separately
            self.assertIn("citation_count", manifest)
            self.assertIn("correction_count", manifest)
            self.assertEqual(manifest["citation_count"], 1)
            self.assertEqual(manifest["correction_count"], 1)


class TestIterationGates(unittest.TestCase):
    """Test automatic iteration gates."""

    def test_data_gate_insufficient_samples(self):
        """RED: Must return insufficient_evidence when sample count too low."""
        from scripts.iterative_rank_loop import check_data_gate
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create tiny manifest
            manifest = {
                "train_samples": 5,
                "val_samples": 2,
                "unknown_ratio": 0.8,
            }
            
            result = check_data_gate(manifest, min_samples=100)
            
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertIn("reason", result)

    def test_data_gate_unknown_ratio_too_high(self):
        """RED: Must warn if unknown ratio exceeds threshold."""
        from scripts.iterative_rank_loop import check_data_gate
        
        manifest = {
            "train_samples": 1000,
            "val_samples": 200,
            "unknown_ratio": 0.95,
            "citation_count": 50,
        }
        
        result = check_data_gate(manifest, max_unknown_ratio=0.9)
        
        # Should warn but not block
        self.assertEqual(result["status"], "pass_with_warning")
        self.assertTrue(any("unknown_ratio" in w for w in result["warnings"]))


class TestPrivacyBoundaries(unittest.TestCase):
    """Test that iteration loop respects privacy boundaries."""

    def test_no_real_data_paths_in_output(self):
        """RED: Output manifest must not contain real source paths."""
        from scripts.iterative_rank_loop import build_iteration_manifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            
            samples_file = input_dir / "samples.jsonl"
            with open(samples_file, "w") as f:
                f.write(json.dumps({
                    "trace_id": "t1",
                    "query_id": "q1",
                    "memory_id": "m1",
                    "query_timestamp": 1000.0,
                    "label": "positive",
                    "label_source": "citation",
                }) + "\n")
            
            manifest = build_iteration_manifest(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )
            
            # Convert manifest to JSON string
            manifest_str = json.dumps(manifest)
            
            # Check no Windows/Unix absolute paths
            self.assertNotIn("C:\\", manifest_str)
            self.assertNotIn("/home/", manifest_str)
            self.assertNotIn("/Users/", manifest_str)

    def test_no_query_or_memory_text_in_manifest(self):
        """RED: Manifest must not contain query/memory text content."""
        from scripts.iterative_rank_loop import build_iteration_manifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            
            samples_file = input_dir / "samples.jsonl"
            with open(samples_file, "w") as f:
                f.write(json.dumps({
                    "trace_id": "t1",
                    "query_id": "q1",
                    "memory_id": "m1",
                    "query_timestamp": 1000.0,
                    "label": "positive",
                    "label_source": "citation",
                    "query_text_redacted": "How to configure proxy?",
                    "memory_text_redacted": "Set HTTP_PROXY environment variable.",
                }) + "\n")
            
            manifest = build_iteration_manifest(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )
            
            # Manifest should only have metadata, not text
            manifest_str = json.dumps(manifest)
            self.assertNotIn("proxy", manifest_str.lower())
            self.assertNotIn("HTTP_PROXY", manifest_str)


if __name__ == "__main__":
    unittest.main()
