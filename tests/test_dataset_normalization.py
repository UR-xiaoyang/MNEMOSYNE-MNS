"""
Test suite for normalize_conversations.py

Tests message normalization from various sources to standardized JSONL.
"""
import unittest
import tempfile
import json
import sqlite3
from pathlib import Path


class TestDatasetNormalization(unittest.TestCase):
    """Test conversation normalization to standard schema."""

    def test_normalizes_hermes_messages_to_standard_schema(self):
        """Should extract visible messages with stable IDs and metadata."""
        from scripts.normalize_conversations import normalize_hermes_db
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create toy Hermes database
            db_path = Path(tmpdir) / "hermes.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    created_at INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at INTEGER,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            """)
            
            cursor.execute("INSERT INTO sessions VALUES ('sess1', 1700000000)")
            cursor.execute("INSERT INTO messages VALUES (1, 'sess1', 'user', 'Hello', 1700000001)")
            cursor.execute("INSERT INTO messages VALUES (2, 'sess1', 'assistant', 'Hi there', 1700000002)")
            cursor.execute("INSERT INTO messages VALUES (3, 'sess1', 'system', 'Hidden prompt', 1700000000)")
            conn.commit()
            conn.close()
            
            output_path = Path(tmpdir) / "normalized.jsonl"
            normalize_hermes_db(str(db_path), output_path, source_token="test_token")
            
            # Read normalized output
            messages = []
            with open(output_path) as f:
                for line in f:
                    messages.append(json.loads(line))
            
            # Should exclude system message
            self.assertEqual(len(messages), 2)
            
            # Verify schema
            msg = messages[0]
            self.assertIn("id", msg)
            self.assertEqual(msg["source"], "hermes")
            self.assertEqual(msg["session_id"], "sess1")
            self.assertEqual(msg["role"], "user")
            self.assertEqual(msg["content"], "Hello")
            self.assertIn("timestamp", msg)
            self.assertIn("ordinal", msg)
            self.assertFalse(msg["redacted"])
            self.assertFalse(msg["truncated"])

    def test_excludes_system_and_developer_messages(self):
        """Should filter out system/developer/hidden messages."""
        from scripts.normalize_conversations import should_include_message
        
        # Should include
        self.assertTrue(should_include_message("user", "Hello"))
        self.assertTrue(should_include_message("assistant", "Response"))
        self.assertTrue(should_include_message("tool", "Tool output"))
        
        # Should exclude
        self.assertFalse(should_include_message("system", "System prompt"))
        self.assertFalse(should_include_message("developer", "Dev instruction"))

    def test_generates_stable_message_ids(self):
        """Should create deterministic IDs for deduplication."""
        from scripts.normalize_conversations import generate_message_id
        
        id1 = generate_message_id("hermes", "file.db", "sess1", 1, "user", "Hello", 1700000001)
        id2 = generate_message_id("hermes", "file.db", "sess1", 1, "user", "Hello", 1700000001)
        
        # Same inputs should produce same ID
        self.assertEqual(id1, id2)
        
        # Different content should produce different ID
        id3 = generate_message_id("hermes", "file.db", "sess1", 1, "user", "Different", 1700000001)
        self.assertNotEqual(id1, id3)

    def test_truncates_oversized_tool_output_with_marker(self):
        """Should truncate long tool output and mark it."""
        from scripts.normalize_conversations import truncate_if_needed
        
        short_content = "Short message"
        result = truncate_if_needed(short_content, max_chars=10000)
        self.assertEqual(result["content"], short_content)
        self.assertFalse(result["truncated"])
        
        # Create content over limit
        long_content = "x" * 15000
        result = truncate_if_needed(long_content, max_chars=10000)
        self.assertTrue(result["truncated"])
        self.assertLess(len(result["content"]), len(long_content))
        self.assertIn("[TRUNCATED:", result["content"])
        self.assertIn("original length 15000", result["content"])
        self.assertEqual(result["original_length"], 15000)

    def test_rejects_framework_injected_content(self):
        """Should detect and mark framework/skill/agent injected text."""
        from scripts.normalize_conversations import is_framework_injection
        
        # Should detect injections
        self.assertTrue(is_framework_injection("[System note: Task assignment]"))
        self.assertTrue(is_framework_injection("# Skill: test-skill"))
        self.assertTrue(is_framework_injection("<!DOCTYPE html>"))  # IDE event
        
        # Should not flag normal content
        self.assertFalse(is_framework_injection("User asked a question"))
        self.assertFalse(is_framework_injection("Assistant response"))


if __name__ == "__main__":
    unittest.main()
