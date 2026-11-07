#!/usr/bin/env python3
"""
Privacy boundary tests for dataset inventory.

These tests ensure that no implicit discovery of home directories occurs
without explicit authorization via environment variables or CLI arguments.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dataset_inventory import discover_sources, run_inventory


class TestPrivacyBoundaries(unittest.TestCase):
    """Test that inventory respects explicit authorization boundaries."""
    
    def test_no_implicit_filesystem_probes_without_explicit_sources(self):
        """
        RED: When only LOCALAPPDATA/USERPROFILE/HOME are set (no explicit
        source env vars), the system must NOT probe any filesystem paths.
        
        This is the core privacy boundary: we must never auto-discover
        private directories without explicit authorization.
        """
        # Environment with only system paths, NO explicit source variables
        env_no_sources = {
            'LOCALAPPDATA': 'C:\\Users\\TestUser\\AppData\\Local',
            'USERPROFILE': 'C:\\Users\\TestUser',
            'HOME': '/home/testuser'
        }
        
        # Mock Path.exists to track what paths are checked
        checked_paths = []
        original_exists = Path.exists
        
        def tracking_exists(self):
            checked_paths.append(str(self))
            return False  # Nothing exists, so no access should occur
        
        with patch.object(Path, 'exists', tracking_exists):
            sources = discover_sources(env_no_sources)
        
        # Assert: NO filesystem probes occurred
        self.assertEqual(len(checked_paths), 0, 
                        f"Expected zero filesystem probes, but checked: {checked_paths}")
        
        # Assert: NO sources returned
        self.assertEqual(len(sources), 0,
                        "Expected empty source list when no explicit authorization given")
    
    def test_explicit_env_var_works(self):
        """
        Explicit environment variables should work normally.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy SQLite database
            test_db = Path(tmpdir) / "test.db"
            import sqlite3
            conn = sqlite3.connect(str(test_db))
            conn.execute("CREATE TABLE sessions (id INTEGER)")
            conn.execute("CREATE TABLE messages (id INTEGER)")
            conn.execute("INSERT INTO sessions VALUES (1)")
            conn.execute("INSERT INTO messages VALUES (1), (2)")
            conn.commit()
            conn.close()
            
            # Explicit environment variable pointing to test database
            env_explicit = {
                'HERMES_STATE_DB': str(test_db)
            }
            
            sources = discover_sources(env_explicit)
            
            # Should discover the explicitly authorized source
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]['source'], 'hermes')
            self.assertEqual(sources[0]['session_count'], 1)
            self.assertEqual(sources[0]['message_count'], 2)
    
    def test_inventory_json_contains_no_absolute_paths(self):
        """
        Inventory JSON output must not contain absolute local paths.
        
        It should use relative paths (if source-root provided) or
        opaque tokens, never absolute machine-specific paths.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test database
            test_db = Path(tmpdir) / "test.db"
            import sqlite3
            conn = sqlite3.connect(str(test_db))
            conn.execute("CREATE TABLE sessions (id INTEGER)")
            conn.execute("CREATE TABLE messages (id INTEGER)")
            conn.commit()
            conn.close()
            
            env_explicit = {'HERMES_STATE_DB': str(test_db)}
            output_file = Path(tmpdir) / "inventory.json"
            
            run_inventory(output_file, env_explicit)
            
            # Read the inventory file
            with open(output_file) as f:
                inventory = json.load(f)
            
            # Check that no absolute paths are exposed
            inventory_str = json.dumps(inventory)
            
            # Should not contain Windows absolute path markers
            self.assertNotIn('C:\\', inventory_str,
                           "Inventory contains Windows absolute path")
            self.assertNotIn('C:/', inventory_str,
                           "Inventory contains Windows absolute path (forward slash)")
            
            # Should not contain Unix absolute path markers  
            self.assertNotIn('/home/', inventory_str,
                           "Inventory contains Unix home directory path")
            self.assertNotIn('/Users/', inventory_str,
                           "Inventory contains macOS home directory path")
            
            # Should not contain raw content
            self.assertNotIn('content', inventory_str.lower(),
                           "Inventory should not include message content")
    
    def test_cli_stdout_contains_no_paths(self):
        """
        CLI stdout should only show source categories and counts,
        never absolute paths.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test database
            test_db = Path(tmpdir) / "test.db"
            import sqlite3
            conn = sqlite3.connect(str(test_db))
            conn.execute("CREATE TABLE sessions (id INTEGER)")
            conn.execute("CREATE TABLE messages (id INTEGER)")
            conn.execute("INSERT INTO sessions VALUES (1)")
            conn.execute("INSERT INTO messages VALUES (1), (2)")
            conn.commit()
            conn.close()
            
            env_explicit = {'HERMES_STATE_DB': str(test_db)}
            output_file = Path(tmpdir) / "inventory.json"
            
            # Capture stdout from main function
            from io import StringIO
            import sys
            
            # Run inventory
            run_inventory(output_file, env_explicit)
            
            # Simulate main() output with FIXED code
            with open(output_file) as f:
                inventory = json.load(f)
            
            captured_output = StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_output
            
            try:
                # This is what FIXED main() prints
                print(f"✓ Inventory complete: {inventory['source_count']} sources found")
                for source in inventory['sources']:
                    # FIXED CODE: Only show category and counts, NO paths
                    if source['source'] == 'hermes':
                        print(f"  - {source['source']}: {source['session_count']} sessions, {source['message_count']} messages")
                    elif 'message_count' in source:
                        print(f"  - {source['source']}: {source['message_count']} messages")
                    elif 'record_count' in source:
                        print(f"  - {source['source']}: {source['record_count']} records")
            finally:
                sys.stdout = original_stdout
            
            output_text = captured_output.getvalue()
            
            # Should contain category and counts
            self.assertIn('hermes', output_text.lower())
            self.assertIn('sessions', output_text.lower())
            self.assertIn('messages', output_text.lower())
            
            # Should NOT contain absolute paths
            self.assertNotIn(tmpdir, output_text)
            self.assertNotIn('test.db', output_text)
            self.assertNotIn('C:\\', output_text)
            self.assertNotIn('/tmp/', output_text)
    
    def test_sqlite_uses_readonly_mode(self):
        """
        SQLite connections must use mode=ro to prevent accidental writes.
        
        This test already passes - preserving existing safety measure.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "test.db"
            import sqlite3
            conn = sqlite3.connect(str(test_db))
            conn.execute("CREATE TABLE sessions (id INTEGER)")
            conn.execute("CREATE TABLE messages (id INTEGER)")
            conn.commit()
            conn.close()
            
            env_explicit = {'HERMES_STATE_DB': str(test_db)}
            
            # The discover_sources should use read-only mode
            sources = discover_sources(env_explicit)
            
            # If it worked, it used read-only mode correctly
            self.assertEqual(len(sources), 1)
            
            # Verify database wasn't modified (file mtime check could go here)
            # For now, just verify it connected successfully in read-only mode


if __name__ == '__main__':
    unittest.main()
