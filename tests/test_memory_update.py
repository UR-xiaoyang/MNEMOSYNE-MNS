import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from memory_schema import memory_from_api
from memory_store import MemoryStore


class MemoryUpdateTest(unittest.TestCase):
    def test_supersede_creates_new_version_and_hides_old_from_strict_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            old = memory_from_api("old", "old proxy configuration", destination="local")
            store.upsert(old)
            new = memory_from_api("new", "new proxy configuration", destination="local")
            updated = store.supersede("old", new)
            self.assertEqual(updated.memory_id, "new")
            self.assertEqual(updated.version, 2)
            self.assertEqual(updated.provenance.parent_memory_id, "old")
            loaded_old = store.get("old")
            self.assertEqual(loaded_old.validity.state, "superseded")
            self.assertEqual(loaded_old.validity.superseded_by, "new")
            candidates = store.search_candidates("proxy", scope_level="global", destination="local")
            self.assertEqual([item.memory_id for item in candidates], ["new"])
            store.close()


if __name__ == "__main__":
    unittest.main()
