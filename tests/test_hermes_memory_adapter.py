import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hermes_memory_adapter import HermesMemoryAdapter, MemoryAdapterConfig
from memory_service import MemoryService
from memory_store import MemoryStore
from shadow_mode import ShadowModeCollector


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
                "metadata": {
                    "validity_state": memory.validity.state,
                    "conflict_with": [],
                },
            }
            for index, memory in enumerate(sorted(candidate_memories, key=lambda item: item.memory_id))
        ]

    def assemble_context(self, ranked, **kwargs):
        return ranked[:2]


class HermesMemoryAdapterTest(unittest.TestCase):
    def test_before_and_after_turn_record_trace_without_committing_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            shadow = ShadowModeCollector(Path(tmp) / "shadow.db")
            service = MemoryService(FakePipeline(), store)
            service.write(
                "m1",
                "Configure proxy with http.proxy and https.proxy",
                scope_level="global",
                destination="local",
            )
            service.write(
                "m2",
                "Proxy fallback via SSH",
                scope_level="global",
                destination="local",
            )
            adapter = HermesMemoryAdapter(
                service,
                shadow,
                MemoryAdapterConfig(shadow_enabled=True, max_context_tokens=200),
            )

            context = adapter.before_turn(
                "How do I configure proxy?",
                session_id="s1",
                turn_id="t1",
                scope_level="global",
                destination="local",
            )
            self.assertIsNotNone(context.retrieval_event_id)
            self.assertEqual(context.assembled[0]["memory_id"], "m1")
            self.assertIn("http.proxy", context.context_text)

            trace = adapter.after_turn(
                context,
                "Configure proxy with http.proxy and https.proxy.",
            )
            self.assertEqual([item["memory_id"] for item in trace.citations], ["m1"])
            self.assertIn("m1", trace.citation_event_ids)
            metrics = shadow.compute_metrics("s1")
            self.assertEqual(metrics["injected_count"], 2)
            self.assertEqual(metrics["cited_count"], 1)
            self.assertEqual(len(shadow.get_session_traces("s1")["ignored_events"]), 1)
            self.assertIsNone(store.get("How do I configure proxy?"))

            correction_id = adapter.record_correction(
                trace,
                memory_id="m1",
                correction_type="outdated",
                feedback={"action": "supersede"},
            )
            self.assertIsNotNone(correction_id)
            self.assertEqual(shadow.compute_metrics("s1")["correction_count"], 1)
            shadow.close()
            store.close()

    def test_disabled_shadow_mode_is_observational_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            shadow = ShadowModeCollector(Path(tmp) / "shadow.db")
            service = MemoryService(FakePipeline(), store)
            service.write("m1", "Use SSH", scope_level="global", destination="local")
            adapter = HermesMemoryAdapter(
                service,
                shadow,
                MemoryAdapterConfig(shadow_enabled=False),
            )

            context = adapter.before_turn(
                "Use SSH",
                session_id="s1",
                turn_id="t1",
                scope_level="global",
                destination="local",
            )
            trace = adapter.after_turn(context, "Use SSH for the connection.")
            self.assertIsNone(context.retrieval_event_id)
            self.assertTrue(trace.citations)
            self.assertEqual(shadow.compute_metrics("s1")["injected_count"], 0)
            self.assertIsNone(
                adapter.record_correction(
                    trace,
                    memory_id="m1",
                    correction_type="outdated",
                    feedback={},
                )
            )
            shadow.close()
            store.close()


if __name__ == "__main__":
    unittest.main()
