"""Test run output directory management and manifest generation."""

import json
import tempfile
import unittest
from pathlib import Path


class TestOutputDirectoryPolicy(unittest.TestCase):
    """Test atomic output directory management."""

    def test_prepare_rejects_existing_without_overwrite(self):
        """Existing directory without --overwrite must be rejected with zero writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run_output"
            output_dir.mkdir()
            marker = output_dir / "existing_marker.txt"
            marker.write_text("original content", encoding="utf-8")

            from scripts.run_output_manager import prepare_output_directory

            with self.assertRaises(FileExistsError) as ctx:
                prepare_output_directory(output_dir, overwrite=False)

            self.assertIn("already exists", str(ctx.exception).lower())
            # Marker must be untouched
            self.assertTrue(marker.exists())
            self.assertEqual(marker.read_text(encoding="utf-8"), "original content")

    def test_prepare_creates_new_directory(self):
        """New directory path should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_run"
            self.assertFalse(output_dir.exists())

            from scripts.run_output_manager import prepare_output_directory

            audit = prepare_output_directory(output_dir, overwrite=False)

            self.assertTrue(output_dir.exists())
            self.assertTrue(output_dir.is_dir())
            self.assertIsNone(audit)  # No overwrite, so no audit

    def test_prepare_with_overwrite_clears_stale_files(self):
        """Existing directory with --overwrite must clear stale files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run_output"
            output_dir.mkdir()
            stale1 = output_dir / "old_model.joblib"
            stale2 = output_dir / "old_manifest.json"
            stale1.write_text("stale model", encoding="utf-8")
            stale2.write_text("{}", encoding="utf-8")

            from scripts.run_output_manager import prepare_output_directory

            audit = prepare_output_directory(output_dir, overwrite=True)

            # Directory exists but stale files are gone
            self.assertTrue(output_dir.exists())
            self.assertFalse(stale1.exists())
            self.assertFalse(stale2.exists())

            # Audit must be returned
            self.assertIsNotNone(audit)
            self.assertIn("overwritten_at", audit)
            self.assertIn("removed_files", audit)
            self.assertGreaterEqual(len(audit["removed_files"]), 2)


class TestManifestGeneration(unittest.TestCase):
    """Test manifest construction with real data."""

    def test_build_manifest_includes_required_fields(self):
        """Manifest must include schema, run_id, model_id, timestamps."""
        from scripts.run_output_manager import build_training_manifest

        manifest = build_training_manifest(
            run_id="test-run-123",
            seed=42,
            input_hashes={"train": "abc123", "val": "def456"},
            counts={
                "train_rows": 100,
                "val_rows": 20,
                "train_queries": 10,
                "val_queries": 5,
                "train_pairs": 300,
            },
            elapsed_seconds=123.45,
            overwrite_audit=None,
        )

        self.assertEqual(manifest["schema_version"], "1.0.0")
        self.assertEqual(manifest["run_id"], "test-run-123")
        self.assertEqual(manifest["model_id"], "MNEMOSYNE-MNS-RANK-HYBRID-01")
        self.assertIn("created_at", manifest)
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["train_queries"], 10)
        self.assertEqual(manifest["validation_queries"], 5)
        self.assertEqual(manifest["train_pair_count"], 300)
        self.assertIsNone(manifest.get("overwrite_audit"))

    def test_build_manifest_with_overwrite_audit(self):
        """Manifest must include overwrite audit when provided."""
        from scripts.run_output_manager import build_training_manifest

        audit = {
            "overwritten_at": "2026-11-03T12:00:00",
            "removed_files": ["old.joblib", "old.json"],
            "policy_version": "clear_on_overwrite_v1",
        }

        manifest = build_training_manifest(
            run_id="test-run-456",
            seed=99,
            input_hashes={"train": "aaa", "val": "bbb"},
            counts={
                "train_rows": 50,
                "val_rows": 10,
                "train_queries": 5,
                "val_queries": 2,
                "train_pairs": 150,
            },
            elapsed_seconds=60.0,
            overwrite_audit=audit,
        )

        self.assertIsNotNone(manifest["overwrite_audit"])
        self.assertEqual(manifest["overwrite_audit"]["policy_version"], "clear_on_overwrite_v1")
        self.assertEqual(len(manifest["overwrite_audit"]["removed_files"]), 2)

    def test_build_manifest_computes_real_hashes(self):
        """Manifest should use real SHA-256 hashes, not placeholders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"hello world")

            from scripts.run_output_manager import compute_file_hash

            sha = compute_file_hash(test_file)

            # Real SHA-256 of "hello world"
            expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            self.assertEqual(sha, expected)


if __name__ == "__main__":
    unittest.main()
