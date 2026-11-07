"""
P1 Round 3: Test source_token enforcement in normalization API.

Verifies that normalize_hermes_db() and normalize_jsonl_file()
require explicit source_token and never output absolute paths.
"""
import unittest
import tempfile
import json
import sqlite3
from pathlib import Path


class TestSourceTokenRequired(unittest.TestCase):
    """Test that source_token is required and paths are never exposed."""

    def test_normalize_hermes_db_requires_source_token(self):
        """Should reject normalization without explicit source_token."""
        from scripts.normalize_conversations import normalize_hermes_db
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy database
            db_path = Path(tmpdir) / "hermes.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at INTEGER)")
            cursor.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, created_at INTEGER)")
            cursor.execute("INSERT INTO sessions VALUES ('s1', 1700000000)")
            cursor.execute("INSERT INTO messages VALUES (1, 's1', 'user', 'test', 1700000001)")
            conn.commit()
            conn.close()
            
            output_path = Path(tmpdir) / "out.jsonl"
            
            # Should raise ValueError when source_token is None
            with self.assertRaises(ValueError) as ctx:
                normalize_hermes_db(str(db_path), output_path, source_token=None)
            
            self.assertIn("source_token", str(ctx.exception).lower())

    def test_normalize_hermes_db_without_token_keyword_raises(self):
        """Should reject normalization when source_token keyword is omitted."""
        from scripts.normalize_conversations import normalize_hermes_db
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy database
            db_path = Path(tmpdir) / "hermes.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at INTEGER)")
            cursor.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, created_at INTEGER)")
            cursor.execute("INSERT INTO sessions VALUES ('s1', 1700000000)")
            cursor.execute("INSERT INTO messages VALUES (1, 's1', 'user', 'test', 1700000001)")
            conn.commit()
            conn.close()
            
            output_path = Path(tmpdir) / "out.jsonl"
            
            # Should raise ValueError when called without source_token keyword
            # (testing positional call which should fail)
            with self.assertRaises((ValueError, TypeError)):
                # Try calling without source_token parameter at all
                normalize_hermes_db(str(db_path), output_path)

    def test_normalize_jsonl_requires_source_token(self):
        """Should reject JSONL normalization without explicit source_token."""
        from scripts.normalize_conversations import normalize_jsonl_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy JSONL
            source_file = Path(tmpdir) / "source.jsonl"
            with open(source_file, "w") as f:
                f.write(json.dumps({"role": "user", "content": "test"}) + "\n")
            
            output_path = Path(tmpdir) / "out.jsonl"
            
            # Should raise ValueError when source_token is None
            with self.assertRaises(ValueError) as ctx:
                normalize_jsonl_file("test", source_file, output_path, source_token=None)
            
            self.assertIn("source_token", str(ctx.exception).lower())

    def test_normalize_jsonl_without_token_keyword_raises(self):
        """Should reject JSONL normalization when source_token keyword is omitted."""
        from scripts.normalize_conversations import normalize_jsonl_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy JSONL
            source_file = Path(tmpdir) / "source.jsonl"
            with open(source_file, "w") as f:
                f.write(json.dumps({"role": "user", "content": "test"}) + "\n")
            
            output_path = Path(tmpdir) / "out.jsonl"
            
            # Should raise when called without source_token parameter
            with self.assertRaises((ValueError, TypeError)):
                normalize_jsonl_file("test", source_file, output_path)

    def test_normalized_output_contains_no_absolute_paths(self):
        """Should use token in output, never absolute paths."""
        from scripts.normalize_conversations import normalize_hermes_db
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy database
            db_path = Path(tmpdir) / "hermes.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at INTEGER)")
            cursor.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, created_at INTEGER)")
            cursor.execute("INSERT INTO sessions VALUES ('s1', 1700000000)")
            cursor.execute("INSERT INTO messages VALUES (1, 's1', 'user', 'test message', 1700000001)")
            conn.commit()
            conn.close()
            
            output_path = Path(tmpdir) / "out.jsonl"
            
            # Normalize with opaque token
            normalize_hermes_db(str(db_path), output_path, source_token="opaque_token_123")
            
            # Read output and verify
            with open(output_path) as f:
                messages = [json.loads(line) for line in f]
            
            self.assertEqual(len(messages), 1)
            msg = messages[0]
            
            # Should use token, not path
            self.assertEqual(msg["source_file"], "opaque_token_123")
            
            # Should not contain tmpdir path anywhere
            msg_str = json.dumps(msg)
            self.assertNotIn(str(tmpdir), msg_str)
            self.assertNotIn("hermes.db", msg_str)

    def test_jsonl_normalized_output_contains_no_absolute_paths(self):
        """Should use token in JSONL output, never absolute paths."""
        from scripts.normalize_conversations import normalize_jsonl_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy JSONL
            source_file = Path(tmpdir) / "source.jsonl"
            with open(source_file, "w") as f:
                f.write(json.dumps({"role": "user", "content": "test query", "session_id": "s1"}) + "\n")
            
            output_path = Path(tmpdir) / "out.jsonl"
            
            # Normalize with opaque token
            normalize_jsonl_file("test", source_file, output_path, source_token="token_xyz")
            
            # Read output and verify
            with open(output_path) as f:
                messages = [json.loads(line) for line in f]
            
            self.assertEqual(len(messages), 1)
            msg = messages[0]
            
            # Should use token, not path
            self.assertEqual(msg["source_file"], "token_xyz")
            
            # Should not contain source file path anywhere
            msg_str = json.dumps(msg)
            self.assertNotIn(str(tmpdir), msg_str)
            self.assertNotIn("source.jsonl", msg_str)


if __name__ == "__main__":
    unittest.main()
