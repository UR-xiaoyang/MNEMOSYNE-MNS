import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _MemoryProvider:
    pass


class _RecallStatus:
    def __init__(self, provider_label, count, glyph="🧠"):
        self.provider_label = provider_label
        self.count = count
        self.glyph = glyph


class _FakeClient:
    def __init__(self):
        self.calls = []

    def retrieve_store(self, query, *, session_id, turn_id, scope_level, destination, top_k):
        self.calls.append(("retrieve_store", query, session_id, turn_id, scope_level, destination, top_k))
        return {
            "results": [
                {
                    "memory_id": "m1",
                    "content": "Use the local proxy configuration.",
                    "rank": 1,
                    "score": 0.9,
                    "injected": True,
                }
            ],
            "retrieval_event_id": "evt-1",
        }

    def write_proposals(self, text, *, scope_level):
        self.calls.append(("write_proposals", text, scope_level))
        return {"proposals": [{"kind": "preference", "admissible": True}]}

    def detect_citations(self, response, injected_memories):
        self.calls.append(("detect_citations", response, injected_memories))
        return {"citations": [{"memory_id": "m1", "span": "proxy", "confidence": 0.9}]}

    def log_citation(self, payload):
        self.calls.append(("log_citation", payload))
        return {"event_id": "citation-1"}


class MnemosyneProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        agent_module = types.ModuleType("agent")
        memory_provider_module = types.ModuleType("agent.memory_provider")
        memory_provider_module.MemoryProvider = _MemoryProvider
        memory_provider_module.RecallStatus = _RecallStatus
        memory_provider_module.spawn_context_thread = lambda target, **kwargs: None
        agent_module.memory_provider = memory_provider_module
        sys.modules["agent"] = agent_module
        sys.modules["agent.memory_provider"] = memory_provider_module
        path = Path(__file__).resolve().parents[1] / "plugins" / "memory" / "mnemosyne" / "__init__.py"
        spec = importlib.util.spec_from_file_location("mnemosyne_provider", path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules["mnemosyne_provider"] = cls.module
        spec.loader.exec_module(cls.module)

    def test_prefetch_returns_ranked_context_and_recall_status(self):
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            provider.on_turn_start(3, "proxy")
            provider.queue_prefetch("proxy", session_id="s1")
            context = provider.prefetch("proxy", session_id="s1")

        self.assertIn("Use the local proxy configuration.", context)
        self.assertEqual(provider.recall_status().count, 1)
        self.assertEqual(fake.calls[0][0], "retrieve_store")
        self.assertEqual(fake.calls[0][3], "turn-3")

    def test_tools_only_generate_write_proposals_and_never_commit(self):
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            payload = json.loads(
                provider.handle_tool_call(
                    "mnemosyne_write_proposals",
                    {"text": "Please remember that I prefer Chinese replies."},
                )
            )

        self.assertTrue(payload["proposals"][0]["admissible"])
        self.assertEqual([call[0] for call in fake.calls], ["write_proposals"])
        self.assertNotIn("mnemosyne_remember", [schema["name"] for schema in provider.get_tool_schemas()])

    def test_sync_turn_records_citation_only_when_shadow_enabled(self):
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            provider.on_turn_start(3, "proxy")
            provider.queue_prefetch("proxy", session_id="s1")
            provider.prefetch("proxy", session_id="s1")
            old = __import__("os").environ.get("MNEMOSYNE_SHADOW_MODE")
            __import__("os").environ["MNEMOSYNE_SHADOW_MODE"] = "1"
            try:
                provider.sync_turn("query", "Use the local proxy configuration.", session_id="s1")
            finally:
                if old is None:
                    __import__("os").environ.pop("MNEMOSYNE_SHADOW_MODE", None)
                else:
                    __import__("os").environ["MNEMOSYNE_SHADOW_MODE"] = old

        log_calls = [call for call in fake.calls if call[0] == "log_citation"]
        self.assertEqual(len(log_calls), 1)
        self.assertEqual(log_calls[0][1]["retrieval_event_id"], "evt-1")

        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            provider.sync_turn("remember this", "acknowledged", session_id="s1")

        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
