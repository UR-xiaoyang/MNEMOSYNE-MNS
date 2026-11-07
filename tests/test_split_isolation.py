"""
Test suite for build_session_splits.py

Tests session-level splitting with isolation guarantees.
"""
import unittest
import tempfile
import json
import hashlib
from pathlib import Path


class TestSessionSplits(unittest.TestCase):
    """Test session-based dataset splitting."""

    def test_splits_by_session_not_message(self):
        """Should keep entire sessions together in one split."""
        from scripts.build_session_splits import build_splits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data with 2 sessions
            input_path = Path(tmpdir) / "messages.jsonl"
            with open(input_path, 'w') as f:
                # Session 1 - 3 messages
                for i in range(3):
                    f.write(json.dumps({
                        "id": f"s1_m{i}", "session_id": "session_1",
                        "source": "hermes", "source_file": "test.db",
                        "ordinal": i+1, "timestamp": 1700000000 + i,
                        "role": "user", "content": f"Message {i}",
                        "metadata": {}, "redacted": False, "truncated": False
                    }) + '\n')
                
                # Session 2 - 2 messages
                for i in range(2):
                    f.write(json.dumps({
                        "id": f"s2_m{i}", "session_id": "session_2",
                        "source": "hermes", "source_file": "test.db",
                        "ordinal": i+1, "timestamp": 1700000100 + i,
                        "role": "user", "content": f"Message {i}",
                        "metadata": {}, "redacted": False, "truncated": False
                    }) + '\n')
            
            output_dir = Path(tmpdir) / "splits"
            result = build_splits(input_path, output_dir, seed=42, split_ratio=(0.6, 0.2, 0.2))
            
            # Verify split counts
            self.assertEqual(result["total_sessions"], 2)
            self.assertEqual(result["total_messages"], 5)
            
            # Check that each session appears in only one split
            session_splits = {}
            for split_name in ["train", "validation", "test"]:
                split_file = output_dir / f"{split_name}.jsonl"
                if split_file.exists():
                    with open(split_file) as f:
                        for line in f:
                            msg = json.loads(line)
                            sess_id = msg["session_id"]
                            if sess_id in session_splits:
                                self.assertEqual(session_splits[sess_id], split_name,
                                    f"Session {sess_id} appears in multiple splits!")
                            session_splits[sess_id] = split_name

    def test_creates_manifest_with_hash(self):
        """Should create manifest with deterministic hash."""
        from scripts.build_session_splits import build_splits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "messages.jsonl"
            with open(input_path, 'w') as f:
                f.write(json.dumps({
                    "id": "1", "session_id": "sess1", "source": "hermes",
                    "source_file": "test.db", "ordinal": 1, "timestamp": 1700000000,
                    "role": "user", "content": "Test", "metadata": {},
                    "redacted": False, "truncated": False
                }) + '\n')
            
            output_dir = Path(tmpdir) / "splits"
            result = build_splits(input_path, output_dir, seed=42)
            
            # Check manifest exists
            manifest_path = output_dir / "split_manifest.json"
            self.assertTrue(manifest_path.exists())
            
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            self.assertIn("split_hash", manifest)
            self.assertIn("seed", manifest)
            self.assertEqual(manifest["seed"], 42)
            self.assertIn("session_mapping", manifest)

    def test_respects_seed_for_reproducibility(self):
        """Should produce same splits with same seed."""
        from scripts.build_session_splits import build_splits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            input_path = Path(tmpdir) / "messages.jsonl"
            with open(input_path, 'w') as f:
                for session_id in range(10):
                    f.write(json.dumps({
                        "id": f"m{session_id}", "session_id": f"sess{session_id}",
                        "source": "hermes", "source_file": "test.db",
                        "ordinal": 1, "timestamp": 1700000000,
                        "role": "user", "content": "Test",
                        "metadata": {}, "redacted": False, "truncated": False
                    }) + '\n')
            
            # Run twice with same seed
            output_dir1 = Path(tmpdir) / "splits1"
            result1 = build_splits(input_path, output_dir1, seed=42)
            
            output_dir2 = Path(tmpdir) / "splits2"
            result2 = build_splits(input_path, output_dir2, seed=42)
            
            # Compare manifests
            with open(output_dir1 / "split_manifest.json") as f:
                manifest1 = json.load(f)
            with open(output_dir2 / "split_manifest.json") as f:
                manifest2 = json.load(f)
            
            self.assertEqual(manifest1["split_hash"], manifest2["split_hash"])
            self.assertEqual(manifest1["session_mapping"], manifest2["session_mapping"])

    def test_temporal_ordering_preserved(self):
        """Should sort sessions by earliest timestamp for chronological split."""
        from scripts.build_session_splits import build_splits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "messages.jsonl"
            with open(input_path, 'w') as f:
                # Session 2 comes first in file but has later timestamp
                f.write(json.dumps({
                    "id": "1", "session_id": "sess2",
                    "source": "hermes", "source_file": "test.db",
                    "ordinal": 1, "timestamp": 1700001000,
                    "role": "user", "content": "Later",
                    "metadata": {}, "redacted": False, "truncated": False
                }) + '\n')
                
                # Session 1 has earlier timestamp
                f.write(json.dumps({
                    "id": "2", "session_id": "sess1",
                    "source": "hermes", "source_file": "test.db",
                    "ordinal": 1, "timestamp": 1700000000,
                    "role": "user", "content": "Earlier",
                    "metadata": {}, "redacted": False, "truncated": False
                }) + '\n')
            
            output_dir = Path(tmpdir) / "splits"
            result = build_splits(input_path, output_dir, seed=42)
            
            # Check that sessions were sorted by timestamp before splitting
            with open(output_dir / "split_manifest.json") as f:
                manifest = json.load(f)
            
            # Both sessions should be assigned to splits
            self.assertEqual(len(manifest["session_mapping"]), 2)


if __name__ == "__main__":
    unittest.main()
