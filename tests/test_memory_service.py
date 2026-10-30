import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from memory_service import MemoryService
from memory_store import MemoryStore


class FakePipeline:
    class Ranker:
        model_id = "TEST-RANKER"

    ranker = Ranker()

    def retrieve_and_rank(self, query_text, candidate_memories, **kwargs):
        return [
            {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "rank": index + 1,
                "score": 1.0 / (index + 1),
                "injected": True,
                "metadata": {"validity_state": memory.validity.state, "conflict_with": []},
            }
            for index, memory in enumerate(candidate_memories)
        ]

    def assemble_context(self, ranked, **kwargs):
        return ranked[:1]


class MemoryServiceTest(unittest.TestCase):
    def test_write_route_retrieve_and_propose_form_one_service_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            service = MemoryService(FakePipeline(), store)
            written = service.write("m1", "配置代理服务器", scope_level="global", destination="local")
            self.assertEqual(written.memory_id, "m1")

            outcome = service.retrieve_store("代理", scope_level="global", destination="local", top_k=5)
            self.assertEqual(outcome["decision"].primary_path, "engineering/code")
            self.assertEqual(outcome["assembled"][0]["memory_id"], "m1")

            proposals = service.propose_write("请记住：我偏好使用中文回复。", scope_level="project")
            self.assertTrue(proposals[0]["admissible"])
            self.assertIsNone(store.get("请记住：我偏好使用中文回复。"))
            store.close()

    def test_supersede_is_exposed_through_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            service = MemoryService(FakePipeline(), store)
            service.write("old", "旧配置", scope_level="global", destination="local")
            replacement = service.supersede("old", "new", "新配置", scope_level="global", destination="local")
            self.assertEqual(replacement.memory_id, "new")
            self.assertEqual(store.get("old").validity.state, "superseded")
            store.close()


if __name__ == "__main__":
    unittest.main()
