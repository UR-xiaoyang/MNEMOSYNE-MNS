import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experimental_global_memory import ExperimentalGlobalMemory
from memory_store import MemoryStore


class ExperimentalGlobalMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.db")
        self.service = ExperimentalGlobalMemory(self.store)
    
    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()
    
    def test_write_requires_approved_true(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.write("test-id", "content", approved=False)
        self.assertIn("approved", str(ctx.exception).lower())
    
    def test_write_rejects_empty_memory_id(self):
        with self.assertRaises(ValueError):
            self.service.write("", "content", approved=True)
    
    def test_write_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            self.service.write("test-id", "", approved=True)
    
    def test_write_prefixes_memory_id_with_namespace(self):
        result = self.service.write("test-id", "Test content", approved=True)
        self.assertTrue(result["memory_id"].startswith("experimental_global:"))
        self.assertIn("test-id", result["memory_id"])
    
    def test_write_preserves_namespace_if_already_prefixed(self):
        result = self.service.write("experimental_global:existing", "Content", approved=True)
        self.assertEqual(result["memory_id"], "experimental_global:existing")
    
    def test_write_rejects_duplicate_memory_id(self):
        self.service.write("dup-test", "First", approved=True)
        with self.assertRaises(ValueError) as ctx:
            self.service.write("dup-test", "Second", approved=True)
        self.assertIn("already exists", str(ctx.exception))
    
    def test_write_stores_source_session_lineage(self):
        result = self.service.write("lineage-test", "Content", approved=True, source_session_id="sess-123")
        self.assertEqual(result["approved"], True)
        
        # Query the experimental_global_memories table directly
        import sqlite3
        conn = sqlite3.connect(self.store.db_path)
        try:
            cursor = conn.execute(
                "SELECT memory_id, source_session_id FROM experimental_global_memories WHERE memory_id = ?",
                (result["memory_id"],)
            )
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], "sess-123")
        finally:
            conn.close()
    
    def test_list_all_returns_only_experimental_entries(self):
        self.service.write("exp-1", "Experimental memory", approved=True)
        # Write a non-experimental memory directly
        from memory_schema import memory_from_api
        regular = memory_from_api("regular:item", "Regular content", scope_level="global", destination="local")
        self.store.upsert(regular)
        
        result = self.service.list_all()
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["memory_id"].startswith("experimental_global:"))
    
    def test_retrieve_filters_to_experimental_namespace(self):
        self.service.write("exp-db", "Database choice: SQLite", approved=True)
        from memory_schema import memory_from_api
        regular = memory_from_api("regular:db", "Database choice: PostgreSQL", scope_level="global", destination="local")
        self.store.upsert(regular)
        
        results = self.service.retrieve("database")
        exp_ids = [r["memory_id"] for r in results]
        self.assertTrue(all(mid.startswith("experimental_global:") for mid in exp_ids))
    
    def test_retrieve_respects_token_budget(self):
        for i in range(5):
            self.service.write(f"item-{i}", f"Long content item number {i}" * 20, approved=True)
        
        results = self.service.retrieve("content", max_tokens=100, tokens_per_memory=50)
        self.assertLessEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
