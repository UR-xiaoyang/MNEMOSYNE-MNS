"""
P1 Round 3: Test framework injection filtering in normalization.

Verifies that should_include_message() calls is_framework_injection()
and filters out injected content from normalized output.
"""
import unittest
import tempfile
import json
import sqlite3
from pathlib import Path


class TestInjectionFiltering(unittest.TestCase):
    """Test that framework/skill/agent injections are filtered during normalization."""

    def test_should_include_filters_system_note_injection(self):
        """Should reject [System note:] injections."""
        from scripts.normalize_conversations import should_include_message
        
        # System note injection should be excluded
        self.assertFalse(should_include_message("user", "[System note: Task assignment]"))
        
        # Normal user content should be included
        self.assertTrue(should_include_message("user", "What is the weather?"))

    def test_should_include_filters_skill_injection(self):
        """Should reject # Skill: injections."""
        from scripts.normalize_conversations import should_include_message
        
        # Skill injection should be excluded
        self.assertFalse(should_include_message("user", "# Skill: test-driven-development"))
        
        # Normal content with # should be included if not injection pattern
        self.assertTrue(should_include_message("user", "# This is a markdown header"))

    def test_should_include_filters_html_agent_events(self):
        """Should reject HTML/IDE agent event injections."""
        from scripts.normalize_conversations import should_include_message
        
        # HTML injection should be excluded
        self.assertFalse(should_include_message("user", "<!DOCTYPE html><html>..."))
        
        # IMPORTANT marker injection should be excluded
        self.assertFalse(should_include_message("user", "[IMPORTANT: The user launched this CLI...]"))

    def test_should_include_allows_ordinary_angle_brackets(self):
        """Should not reject all content with < character."""
        from scripts.normalize_conversations import should_include_message
        
        # Mathematical comparison should be allowed
        # Note: Current is_framework_injection has '<' pattern that's too broad
        # This test documents the limitation - after Round 4 fix it should pass
        result = should_include_message("user", "x < 5 in the equation")
        # After fix, this should be True (allowed)
        self.assertTrue(result, "Ordinary angle brackets in math should be allowed")

    def test_hermes_db_excludes_injections_from_output(self):
        """Should exclude framework injections from normalized Hermes messages."""
        from scripts.normalize_conversations import normalize_hermes_db
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create toy Hermes database with injected content
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
            cursor.execute("INSERT INTO messages VALUES (1, 'sess1', 'user', 'Normal question', 1700000001)")
            cursor.execute("INSERT INTO messages VALUES (2, 'sess1', 'user', '[System note: Task notification]', 1700000002)")
            cursor.execute("INSERT INTO messages VALUES (3, 'sess1', 'user', '# Skill: memory-dataset-engineering', 1700000003)")
            cursor.execute("INSERT INTO messages VALUES (4, 'sess1', 'assistant', 'Normal response', 1700000004)")
            conn.commit()
            conn.close()
            
            output_path = Path(tmpdir) / "normalized.jsonl"
            normalize_hermes_db(str(db_path), output_path, source_token="test_db")
            
            # Read normalized output
            messages = []
            with open(output_path) as f:
                for line in f:
                    messages.append(json.loads(line))
            
            # Should only have 2 messages (normal question + response), injections excluded
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["content"], "Normal question")
            self.assertEqual(messages[1]["content"], "Normal response")
            
            # Verify no injected content present
            all_content = [m["content"] for m in messages]
            self.assertNotIn("[System note:", "".join(all_content))
            self.assertNotIn("# Skill:", "".join(all_content))

    def test_jsonl_normalization_excludes_injections(self):
        """Should exclude framework injections from normalized JSONL messages."""
        from scripts.normalize_conversations import normalize_jsonl_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create toy JSONL with injected content
            source_file = Path(tmpdir) / "source.jsonl"
            with open(source_file, "w") as f:
                f.write(json.dumps({"role": "user", "content": "Normal query", "session_id": "s1"}) + "\n")
                f.write(json.dumps({"role": "user", "content": "[IMPORTANT: Agent instruction]", "session_id": "s1"}) + "\n")
                f.write(json.dumps({"role": "assistant", "content": "Normal answer", "session_id": "s1"}) + "\n")
            
            output_path = Path(tmpdir) / "normalized.jsonl"
            normalize_jsonl_file("test", source_file, output_path, source_token="test_source")
            
            # Read normalized output
            messages = []
            with open(output_path) as f:
                for line in f:
                    messages.append(json.loads(line))
            
            # Should only have 2 messages (query + answer), injection excluded
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["content"], "Normal query")
            self.assertEqual(messages[1]["content"], "Normal answer")


if __name__ == "__main__":
    unittest.main()
