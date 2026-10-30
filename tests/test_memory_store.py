import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from memory_schema import MemoryEntity, Scope, memory_from_api
from memory_store import MemoryStore


class MemoryStoreTest(unittest.TestCase):
    def test_upsert_get_and_update_preserve_entity_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            memory = memory_from_api("m-1", "configure proxy", destination="local")
            store.upsert(memory)
            loaded = store.get("m-1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.memory_id, "m-1")
            self.assertEqual(loaded.content, "configure proxy")

            updated = memory_from_api("m-1", "configure proxy with HTTPS_PROXY", destination="local")
            updated.version = 2
            store.upsert(updated)
            loaded = store.get("m-1")
            self.assertEqual(loaded.content, "configure proxy with HTTPS_PROXY")
            self.assertEqual(loaded.version, 2)
            store.close()

    def test_candidates_enforce_scope_egress_and_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            visible = memory_from_api("visible", "proxy", destination="local", scope_level="project")
            hidden_scope = memory_from_api("hidden-scope", "proxy", destination="local", scope_level="session")
            hidden_egress = memory_from_api("hidden-egress", "proxy", destination="cloud_openai", scope_level="project")
            superseded = memory_from_api("superseded", "proxy", destination="local", scope_level="project")
            superseded.validity.state = "superseded"
            for item in (visible, hidden_scope, hidden_egress, superseded):
                store.upsert(item)

            candidates = store.search_candidates("proxy", scope_level="project", destination="local")
            self.assertEqual([item.memory_id for item in candidates], ["visible"])
            store.close()

    def test_cjk_substring_search_prefers_matching_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.upsert(memory_from_api("m-proxy", "配置代理服务器", destination="local"))
            store.upsert(memory_from_api("m-ssh", "使用 SSH 连接服务器", destination="local"))
            candidates = store.search_candidates("代理", scope_level="global", destination="local", limit=1)
            self.assertEqual([item.memory_id for item in candidates], ["m-proxy"])
            store.close()


if __name__ == "__main__":
    unittest.main()
