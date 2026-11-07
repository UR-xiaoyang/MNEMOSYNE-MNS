"""
Test suite for dataset_inventory.py

Tests the read-only source inventory functionality without copying private data.
"""
import unittest
import tempfile
import json
import os
import sqlite3
from pathlib import Path


class TestDatasetInventory(unittest.TestCase):
    """Test dataset inventory discovery and safe cataloging."""

    def test_empty_inventory_when_no_sources_configured(self):
        """Should return empty inventory when no environment variables set."""
        # This is the toy fixture case - no private data available
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "inventory.json"
            
            from scripts.dataset_inventory import run_inventory
            
            result = run_inventory(output_path=output_path, env_overrides={})
            
            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                inventory = json.load(f)
            
            self.assertEqual(inventory["source_count"], 0)
            self.assertEqual(len(inventory["sources"]), 0)
            self.assertIn("timestamp", inventory)
            self.assertEqual(inventory["version"], "1.0.0")

    def test_discovers_hermes_db_with_readonly_uri(self):
        """Should open Hermes DB with read-only URI and count records."""
        from scripts.dataset_inventory import run_inventory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a toy Hermes database
            db_path = Path(tmpdir) / "hermes_test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, data TEXT)")
            cursor.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id INTEGER, content TEXT)")
            cursor.execute("INSERT INTO sessions VALUES (1, 'session1')")
            cursor.execute("INSERT INTO sessions VALUES (2, 'session2')")
            cursor.execute("INSERT INTO messages VALUES (1, 1, 'msg1')")
            cursor.execute("INSERT INTO messages VALUES (2, 1, 'msg2')")
            cursor.execute("INSERT INTO messages VALUES (3, 2, 'msg3')")
            conn.commit()
            conn.close()
            
            output_path = Path(tmpdir) / "inventory.json"
            env = {"HERMES_STATE_DB": str(db_path)}
            
            result = run_inventory(output_path=output_path, env_overrides=env)
            
            self.assertEqual(result["source_count"], 1)
            source = result["sources"][0]
            self.assertEqual(source["source"], "hermes")
            self.assertEqual(source["format"], "sqlite")
            self.assertEqual(source["session_count"], 2)
            self.assertEqual(source["message_count"], 3)
            self.assertIn("tables", source["schema"])

    def test_discovers_jsonl_files_without_loading_content(self):
        """Should count JSONL lines without reading private content."""
        from scripts.dataset_inventory import run_inventory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a toy JSONL file
            jsonl_path = Path(tmpdir) / "claude_history.jsonl"
            with open(jsonl_path, 'w') as f:
                f.write('{"id": 1, "role": "user", "content": "sensitive data 1"}\n')
                f.write('{"id": 2, "role": "assistant", "content": "sensitive data 2"}\n')
                f.write('{"id": 3, "role": "user", "content": "sensitive data 3"}\n')
            
            output_path = Path(tmpdir) / "inventory.json"
            env = {"CLAUDE_HISTORY_DIR": str(tmpdir)}
            
            result = run_inventory(output_path=output_path, env_overrides=env)
            
            self.assertEqual(result["source_count"], 1)
            source = result["sources"][0]
            self.assertEqual(source["source"], "claude")
            self.assertEqual(source["format"], "jsonl")
            self.assertEqual(source["record_count"], 3)
            
            # Verify no content was copied into inventory
            inventory_json = json.dumps(result)
            self.assertNotIn("sensitive data", inventory_json)

    def test_skips_empty_files(self):
        """Should skip zero-byte files."""
        from scripts.dataset_inventory import run_inventory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty JSONL file
            empty_file = Path(tmpdir) / "empty.jsonl"
            empty_file.touch()
            
            output_path = Path(tmpdir) / "inventory.json"
            env = {"CLAUDE_HISTORY_DIR": str(tmpdir)}
            
            result = run_inventory(output_path=output_path, env_overrides=env)
            
            # Should skip empty file
            self.assertEqual(result["source_count"], 0)

    def test_handles_nonexistent_paths_gracefully(self):
        """Should not fail when configured paths don't exist."""
        from scripts.dataset_inventory import run_inventory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "inventory.json"
            env = {
                "HERMES_STATE_DB": "/nonexistent/path/hermes.db",
                "CLAUDE_HISTORY_DIR": "/nonexistent/claude",
            }
            
            result = run_inventory(output_path=output_path, env_overrides=env)
            
            # Should succeed with 0 sources
            self.assertEqual(result["source_count"], 0)


if __name__ == "__main__":
    unittest.main()
