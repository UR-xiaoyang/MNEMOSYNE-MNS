"""
P1 Round 4: Test precise injection filtering and path-free CLI output.

Verifies:
1. Ordinary angle brackets in math/natural language are preserved
2. Explicit injection markers are still filtered
3. CLI stdout contains no paths
"""
import unittest
import tempfile
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


class TestPreciseInjectionFilter(unittest.TestCase):
    """Test that injection filter is precise and doesn't over-filter."""

    def test_should_include_allows_mathematical_comparisons(self):
        """Should allow ordinary mathematical angle brackets."""
        from scripts.normalize_conversations import should_include_message
        
        # Mathematical comparisons should be allowed
        self.assertTrue(should_include_message("user", "x < 5 is a valid comparison"))
        self.assertTrue(should_include_message("user", "if temperature < 100 then"))
        self.assertTrue(should_include_message("user", "5 < 10 and 10 > 5"))
        
    def test_should_include_allows_natural_language_angle_brackets(self):
        """Should allow natural language with angle brackets."""
        from scripts.normalize_conversations import should_include_message
        
        # Natural language should be allowed
        self.assertTrue(should_include_message("user", "I need a value < threshold"))
        self.assertTrue(should_include_message("user", "Results show A < B in all cases"))
    
    def test_should_include_still_filters_explicit_html_doctype(self):
        """Should still filter explicit HTML DOCTYPE."""
        from scripts.normalize_conversations import should_include_message
        
        # Explicit HTML DOCTYPE should be filtered
        self.assertFalse(should_include_message("user", "<!DOCTYPE html><html><body>"))
        self.assertFalse(should_include_message("user", "<!DOCTYPE HTML PUBLIC"))
    
    def test_should_include_still_filters_system_notes(self):
        """Should still filter [System note:] injections."""
        from scripts.normalize_conversations import should_include_message
        
        self.assertFalse(should_include_message("user", "[System note: Task assignment]"))
        self.assertFalse(should_include_message("user", "[system note: memory context]"))
    
    def test_should_include_still_filters_skill_injections(self):
        """Should still filter # Skill: injections."""
        from scripts.normalize_conversations import should_include_message
        
        self.assertFalse(should_include_message("user", "# Skill: test-driven-development"))
        self.assertFalse(should_include_message("user", "# skill: memory-dataset-engineering"))
    
    def test_should_include_still_filters_important_markers(self):
        """Should still filter [IMPORTANT: injections."""
        from scripts.normalize_conversations import should_include_message
        
        self.assertFalse(should_include_message("user", "[IMPORTANT: The user launched this CLI]"))
        self.assertFalse(should_include_message("user", "[important: system instruction]"))
    
    def test_should_include_still_filters_assistant_injections(self):
        """Should still filter 'You are a' assistant injections."""
        from scripts.normalize_conversations import should_include_message
        
        self.assertFalse(should_include_message("user", "You are a helpful assistant"))
        self.assertFalse(should_include_message("user", "You are an AI coding assistant"))


class TestCLINoPathOutput(unittest.TestCase):
    """Test that CLI stdout contains no paths."""
    
    def test_normalize_cli_stdout_contains_no_output_path(self):
        """CLI stdout should not contain output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy JSONL input
            source_file = Path(tmpdir) / "source.jsonl"
            with open(source_file, "w") as f:
                f.write(json.dumps({"role": "user", "content": "test", "session_id": "s1"}) + "\n")
            
            # Create minimal inventory
            inventory_file = Path(tmpdir) / "inventory.json"
            with open(inventory_file, "w") as f:
                json.dump({
                    "sources": [{
                        "source": "test",
                        "source_token": "test_tok",
                        "format": "jsonl"
                    }]
                }, f)
            
            output_file = Path(tmpdir) / "output.jsonl"
            
            # Run CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/normalize_conversations.py",
                    "--inventory", str(inventory_file),
                    "--source", f"test_tok={source_file}",
                    "--output", str(output_file)
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # Verify success
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            
            # Verify stdout contains no paths
            stdout = result.stdout
            self.assertNotIn(str(output_file), stdout, "Output path leaked in stdout")
            self.assertNotIn(str(source_file), stdout, "Source path leaked in stdout")
            self.assertNotIn(str(inventory_file), stdout, "Inventory path leaked in stdout")
            self.assertNotIn(tmpdir, stdout, "Tempdir leaked in stdout")
            
            # Verify stdout contains summary info (not paths)
            self.assertIn("✓", stdout, "Should contain success indicator")
            self.assertIn("message", stdout.lower(), "Should mention messages")
    
    def test_normalize_cli_stdout_contains_no_inventory_path(self):
        """CLI stdout should not contain inventory file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal toy database
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at INTEGER)")
            cursor.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, created_at INTEGER)")
            cursor.execute("INSERT INTO sessions VALUES ('s1', 1700000000)")
            cursor.execute("INSERT INTO messages VALUES (1, 's1', 'user', 'test', 1700000001)")
            conn.commit()
            conn.close()
            
            # Create minimal inventory
            inventory_file = Path(tmpdir) / "inventory.json"
            with open(inventory_file, "w") as f:
                json.dump({
                    "sources": [{
                        "source": "hermes",
                        "source_token": "hermes_db",
                        "format": "sqlite"
                    }]
                }, f)
            
            output_file = Path(tmpdir) / "output.jsonl"
            
            # Run CLI with database source
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/normalize_conversations.py",
                    "--inventory", str(inventory_file),
                    "--source", f"hermes_db={db_path}",
                    "--output", str(output_file)
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # Verify success
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            
            # Verify stdout contains no database path
            stdout = result.stdout
            self.assertNotIn(str(db_path), stdout, "Database path leaked in stdout")
            self.assertNotIn(str(output_file), stdout, "Output path leaked in stdout")
            self.assertNotIn(str(inventory_file), stdout, "Inventory path leaked in stdout")
            self.assertNotIn(tmpdir, stdout, "Tempdir leaked in stdout")


if __name__ == "__main__":
    unittest.main()
