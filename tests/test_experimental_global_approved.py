"""Tests for experimental global memory approved persistence.

Validates that:
1. Experimental global memories use dedicated SQLite table
2. approved=1 field is persisted and validated
3. Only approved=1 entries are returned on reads
4. Unapproved/historical entries are not returned
5. Write API rejects approved=false/missing
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from memory_store import MemoryStore
from experimental_global_memory import ExperimentalGlobalMemory


class TestExperimentalGlobalApprovedPersistence(unittest.TestCase):
    """Test experimental global memory approved field persistence."""
    
    def setUp(self):
        """Create temporary database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        self.store = MemoryStore(self.db_path)
        self.exp_global = ExperimentalGlobalMemory(self.store)
    
    def tearDown(self):
        """Clean up temporary database."""
        try:
            os.unlink(self.db_path)
        except:
            pass
    
    def test_dedicated_table_created(self):
        """Experimental global memories use dedicated table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experimental_global_memories'"
        )
        self.assertIsNotNone(cursor.fetchone())
        conn.close()
    
    def test_approved_field_exists(self):
        """Dedicated table has approved field."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("PRAGMA table_info(experimental_global_memories)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        self.assertIn("approved", columns)
        self.assertIn("INTEGER", columns["approved"])
        conn.close()
    
    def test_write_approved_true_persists(self):
        """Writing with approved=True persists approved=1."""
        result = self.exp_global.write(
            memory_id="test-1",
            content="Test content",
            approved=True,
            source_session_id="sess-1"
        )
        
        self.assertEqual(result["approved"], True)
        
        # Verify in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT approved FROM experimental_global_memories WHERE memory_id = ?",
            (result["memory_id"],)
        )
        row = cursor.fetchone()
        self.assertEqual(row[0], 1)
        conn.close()
    
    def test_write_approved_false_rejected(self):
        """Writing with approved=False raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.exp_global.write(
                memory_id="test-2",
                content="Test content",
                approved=False
            )
        
        self.assertIn("approved must be True", str(ctx.exception))
    
    def test_duplicate_id_rejected(self):
        """Duplicate memory_id raises ValueError."""
        self.exp_global.write(
            memory_id="test-3",
            content="First",
            approved=True
        )
        
        with self.assertRaises(ValueError) as ctx:
            self.exp_global.write(
                memory_id="test-3",
                content="Second",
                approved=True
            )
        
        self.assertIn("already exists", str(ctx.exception))
    
    def test_list_all_only_approved(self):
        """list_all returns only approved=1 entries."""
        # Write approved entry
        self.exp_global.write(
            memory_id="approved-1",
            content="Approved content",
            approved=True
        )
        
        # Manually insert unapproved entry
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO experimental_global_memories
               (memory_id, content, approved, status, created_at, updated_at)
               VALUES (?, ?, 0, 'active', ?, ?)""",
            ("experimental_global:unapproved-1", "Unapproved content", 1.0, 1.0)
        )
        conn.commit()
        conn.close()
        
        # list_all should only return approved
        results = self.exp_global.list_all()
        memory_ids = [m["memory_id"] for m in results]
        
        self.assertEqual(len(memory_ids), 1)
        self.assertIn("experimental_global:approved-1", memory_ids)
        self.assertNotIn("experimental_global:unapproved-1", memory_ids)
    
    def test_retrieve_only_approved(self):
        """retrieve returns only approved=1 entries."""
        # Write approved entry
        self.exp_global.write(
            memory_id="approved-2",
            content="Approved retrievable",
            approved=True
        )
        
        # Manually insert unapproved entry
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO experimental_global_memories
               (memory_id, content, approved, status, created_at, updated_at)
               VALUES (?, ?, 0, 'active', ?, ?)""",
            ("experimental_global:unapproved-2", "Unapproved retrievable", 2.0, 2.0)
        )
        conn.commit()
        conn.close()
        
        # retrieve should only return approved
        results = self.exp_global.retrieve("test query")
        memory_ids = [m["memory_id"] for m in results]
        
        self.assertIn("experimental_global:approved-2", memory_ids)
        self.assertNotIn("experimental_global:unapproved-2", memory_ids)
    
    def test_namespace_prefix_added(self):
        """memory_id gets experimental_global: prefix."""
        result = self.exp_global.write(
            memory_id="bare-id",
            content="Test",
            approved=True
        )
        
        self.assertTrue(result["memory_id"].startswith("experimental_global:"))
        self.assertEqual(result["memory_id"], "experimental_global:bare-id")
    
    def test_namespace_prefix_preserved(self):
        """memory_id with existing prefix is preserved."""
        result = self.exp_global.write(
            memory_id="experimental_global:prefixed-id",
            content="Test",
            approved=True
        )
        
        self.assertEqual(result["memory_id"], "experimental_global:prefixed-id")
    
    def test_source_session_id_persisted(self):
        """source_session_id is stored and retrieved."""
        self.exp_global.write(
            memory_id="with-source",
            content="Test",
            approved=True,
            source_session_id="sess-abc"
        )
        
        results = self.exp_global.list_all()
        entry = next(m for m in results if "with-source" in m["memory_id"])
        
        self.assertEqual(entry["source_session_id"], "sess-abc")
    
    def test_empty_content_rejected(self):
        """Empty content raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.exp_global.write(
                memory_id="empty",
                content="",
                approved=True
            )
        
        self.assertIn("content is required", str(ctx.exception))
    
    def test_empty_memory_id_rejected(self):
        """Empty memory_id raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.exp_global.write(
                memory_id="",
                content="Test",
                approved=True
            )
        
        self.assertIn("memory_id is required", str(ctx.exception))
    
    def test_inactive_status_not_returned(self):
        """Entries with status != 'active' are not returned."""
        self.exp_global.write(
            memory_id="active-entry",
            content="Active",
            approved=True
        )
        
        # Manually insert inactive entry
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO experimental_global_memories
               (memory_id, content, approved, status, created_at, updated_at)
               VALUES (?, ?, 1, 'deleted', ?, ?)""",
            ("experimental_global:inactive", "Inactive", 1.0, 1.0)
        )
        conn.commit()
        conn.close()
        
        results = self.exp_global.list_all()
        memory_ids = [m["memory_id"] for m in results]
        
        self.assertIn("experimental_global:active-entry", memory_ids)
        self.assertNotIn("experimental_global:inactive", memory_ids)
    
    def test_list_all_ordered_by_created_at_desc(self):
        """list_all returns newest entries first."""
        import time
        
        self.exp_global.write("first", "First", approved=True)
        time.sleep(0.01)
        self.exp_global.write("second", "Second", approved=True)
        time.sleep(0.01)
        self.exp_global.write("third", "Third", approved=True)
        
        results = self.exp_global.list_all()
        
        # Should be in reverse chronological order
        self.assertIn("third", results[0]["memory_id"])
        self.assertIn("second", results[1]["memory_id"])
        self.assertIn("first", results[2]["memory_id"])


if __name__ == "__main__":
    unittest.main()
