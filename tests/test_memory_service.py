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
                "token_cost": 3,
                "injected": True,
                "metadata": {"validity_state": memory.validity.state, "conflict_with": []},
            }
            for index, memory in enumerate(candidate_memories)
        ]

    last_assembly_kwargs = {}

    def assemble_context(self, ranked, **kwargs):
        self.last_assembly_kwargs = kwargs
        selected = ranked[:1]
        for item in selected:
            item["assembly_token_cost"] = item.get(
                "token_cost",
                kwargs.get("tokens_per_memory", 200),
            )
        return selected


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

    def test_retrieve_store_accepts_context_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            pipeline = FakePipeline()
            service = MemoryService(pipeline, store)
            service.write("m1", "配置代理服务器", scope_level="global", destination="local")
            outcome = service.retrieve_store(
                "代理",
                scope_level="global",
                destination="local",
                top_k=5,
                max_tokens=17,
                tokens_per_memory=3,
            )
            self.assertEqual(outcome["assembled"][0]["assembly_token_cost"], 3)
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
