"""Tests for shadow trace evaluation script."""

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class ShadowEvaluationTest(unittest.TestCase):
    """Test shadow trace evaluation with toy data only."""

    def test_evaluation_reports_insufficient_evidence_when_empty(self):
        """Evaluation must return insufficient_evidence for empty dataset."""
        from evaluate_shadow_traces import evaluate_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            
            # Create empty samples file
            samples_path = data_dir / "shadow_training_samples.jsonl"
            samples_path.write_text("")
            
            result = evaluate_shadow_traces(data_dir=data_dir)
            
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertIn("reason", result)

    def test_evaluation_computes_label_distribution(self):
        """Evaluation must compute label distribution from samples."""
        from evaluate_shadow_traces import evaluate_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            
            # Create toy samples
            samples = [
                {"label": "positive", "label_source": "citation", "memory_id": "m1"},
                {"label": "positive", "label_source": "citation", "memory_id": "m2"},
                {"label": "corrected", "label_source": "correction", "memory_id": "m3"},
            ]
            
            samples_path = data_dir / "shadow_training_samples.jsonl"
            with open(samples_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            
            result = evaluate_shadow_traces(data_dir=data_dir, min_samples=1)
            
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["total_samples"], 3)
            self.assertEqual(result["label_counts"]["positive"], 2)
            self.assertEqual(result["label_counts"]["corrected"], 1)

    def test_evaluation_output_contains_no_source_paths(self):
        """Evaluation report must not leak source paths."""
        from evaluate_shadow_traces import evaluate_shadow_traces
        
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            
            samples = [
                {"label": "positive", "label_source": "citation", "memory_id": "m1"},
            ]
            
            samples_path = data_dir / "shadow_training_samples.jsonl"
            with open(samples_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            
            result = evaluate_shadow_traces(data_dir=data_dir)
            
            result_str = json.dumps(result)
            # Should not contain common path patterns
            self.assertNotIn("C:\\", result_str)
            self.assertNotIn("/home/", result_str)
            self.assertNotIn("/Users/", result_str)

    def test_cli_stdout_does_not_print_output_path(self):
        """CLI with --output must not print the output path to stdout."""
        import subprocess
        
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            output_file = Path(tmp) / "report.json"
            
            # Create toy samples
            samples = [
                {"label": "positive", "label_source": "citation", "memory_id": "m1"},
            ]
            
            samples_path = data_dir / "shadow_training_samples.jsonl"
            with open(samples_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            
            # Run CLI with --output
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_shadow_traces.py"
            result = subprocess.run(
                [sys.executable, str(script_path), 
                 "--data-dir", str(data_dir), 
                 "--output", str(output_file)],
                capture_output=True,
                text=True,
            )
            
            # Stdout must not contain the output path
            self.assertNotIn(str(output_file), result.stdout)
            self.assertNotIn("report.json", result.stdout)
            # Should contain a fixed message without path
            self.assertIn("Report written", result.stdout)


if __name__ == "__main__":
    unittest.main()
