#!/usr/bin/env python3
"""
Tests for normalize_conversations.py source mapping and path redaction.

Ensures:
1. CLI accepts explicit --source <token>=<path> mappings
2. Missing mappings are rejected with clear errors
3. Normalized output uses tokens, not absolute paths
4. CLI stdout contains no absolute paths
"""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import io

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from normalize_conversations import main, normalize_hermes_db


class TestNormalizeSourceMapping(unittest.TestCase):
    """Test source token mapping in normalize pipeline."""
    
    def setUp(self):
        """Create a toy SQLite database for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "toy.db"
        
        # Create minimal Hermes-like database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                created_at INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                ordinal INTEGER,
                role TEXT,
                content TEXT,
                created_at INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Insert test data
        cursor.execute(
            "INSERT INTO sessions VALUES (?, ?)",
            ("test-session-1", 1700000000)
        )
        
        cursor.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            ("msg-1", "test-session-1", 1, "user", "Test message", 1700000001)
        )
        
        conn.commit()
        conn.close()
        
        self.output_path = Path(self.temp_dir.name) / "normalized.jsonl"
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
    
    def test_normalize_with_explicit_source_mapping(self):
        """
        RED: Test that normalizer accepts explicit --source token=path mapping.
        
        This should fail because --source argument doesn't exist yet.
        """
        # Create inventory with only token
        inventory_path = Path(self.temp_dir.name) / "inventory.json"
        source_token = "abc123def456"
        
        inventory = {
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "source_count": 1,
            "sources": [{
                "source": "hermes",
                "source_token": source_token,
                "format": "sqlite",
                "size_bytes": self.db_path.stat().st_size,
                "session_count": 1,
                "message_count": 1
            }]
        }
        
        with open(inventory_path, 'w') as f:
            json.dump(inventory, f)
        
        # Run normalizer with explicit mapping
        with patch('sys.argv', [
            'normalize_conversations.py',
            '--inventory', str(inventory_path),
            '--output', str(self.output_path),
            '--source', f'{source_token}={self.db_path}'
        ]):
            main()
        
        # Verify output exists and is valid
        self.assertTrue(self.output_path.exists())
        
        with open(self.output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            
            msg = json.loads(lines[0])
            # source_file should be token, not absolute path
            self.assertEqual(msg["source_file"], source_token)
            self.assertNotIn(str(self.db_path), msg["source_file"])
    
    def test_normalize_rejects_missing_source_mapping(self):
        """
        RED: Test that normalizer rejects tokens without explicit path mapping.
        
        This should fail because error handling doesn't exist yet.
        """
        inventory_path = Path(self.temp_dir.name) / "inventory.json"
        
        inventory = {
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "source_count": 1,
            "sources": [{
                "source": "hermes",
                "source_token": "unmapped_token",
                "format": "sqlite"
            }]
        }
        
        with open(inventory_path, 'w') as f:
            json.dump(inventory, f)
        
        # Should raise clear error about missing mapping
        with patch('sys.argv', [
            'normalize_conversations.py',
            '--inventory', str(inventory_path),
            '--output', str(self.output_path)
        ]):
            with self.assertRaises(ValueError) as cm:
                main()
            
            error_msg = str(cm.exception)
            self.assertIn("unmapped_token", error_msg)
            self.assertIn("--source", error_msg)
    
    def test_normalized_output_contains_no_absolute_paths(self):
        """
        RED: Test that normalized JSONL contains no absolute paths.
        
        This should fail because current code writes absolute paths.
        """
        inventory_path = Path(self.temp_dir.name) / "inventory.json"
        source_token = "test_token_xyz"
        
        inventory = {
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "source_count": 1,
            "sources": [{
                "source": "hermes",
                "source_token": source_token,
                "format": "sqlite"
            }]
        }
        
        with open(inventory_path, 'w') as f:
            json.dump(inventory, f)
        
        with patch('sys.argv', [
            'normalize_conversations.py',
            '--inventory', str(inventory_path),
            '--output', str(self.output_path),
            '--source', f'{source_token}={self.db_path}'
        ]):
            main()
        
        # Read entire output and check for absolute path leakage
        with open(self.output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Should not contain the actual database path
            self.assertNotIn(str(self.db_path), content)
            self.assertNotIn(str(self.db_path.absolute()), content)
            
            # Should contain the token instead
            self.assertIn(source_token, content)
    
    def test_cli_stdout_contains_no_absolute_paths(self):
        """
        RED: Test that CLI stdout output contains no absolute paths.
        
        This should fail because current code prints absolute paths.
        """
        inventory_path = Path(self.temp_dir.name) / "inventory.json"
        source_token = "stdout_test_token"
        
        inventory = {
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "source_count": 1,
            "sources": [{
                "source": "hermes",
                "source_token": source_token,
                "format": "sqlite"
            }]
        }
        
        with open(inventory_path, 'w') as f:
            json.dump(inventory, f)
        
        # Capture stdout
        captured_output = io.StringIO()
        
        with patch('sys.argv', [
            'normalize_conversations.py',
            '--inventory', str(inventory_path),
            '--output', str(self.output_path),
            '--source', f'{source_token}={self.db_path}'
        ]):
            with patch('sys.stdout', captured_output):
                main()
        
        stdout_text = captured_output.getvalue()
        
        # Should not contain absolute path
        self.assertNotIn(str(self.db_path), stdout_text)
        self.assertNotIn(str(self.db_path.absolute()), stdout_text)
        
        # Should contain token or generic identifier
        # (Either token or just "hermes" is acceptable)
        self.assertTrue(
            source_token in stdout_text or "hermes" in stdout_text.lower(),
            f"Expected token or 'hermes' in output, got: {stdout_text}"
        )


if __name__ == "__main__":
    unittest.main()
