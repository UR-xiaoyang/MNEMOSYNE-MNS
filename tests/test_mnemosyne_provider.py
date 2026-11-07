import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
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

    def retrieve_context_recent(self, *, session_id, project_id="", max_tokens=2000, tokens_per_memory=200):
        self.calls.append(("retrieve_context_recent", session_id, project_id, max_tokens, tokens_per_memory))
        return {
            "results": [
                {
                    "content": "Use the local proxy configuration.",
                }
            ],
            "token_cost": 100,
        }

    def retrieve_experimental_global(self, query, *, max_tokens=1000, tokens_per_memory=200):
        self.calls.append(("retrieve_experimental_global", query, max_tokens, tokens_per_memory))
        return {
            "results": [
                {
                    "memory_id": "m1",
                    "content": "Use the local proxy configuration.",
                    "rank": 1,
                }
            ],
            "retrieval_event_id": "evt-1",
        }

    def retrieve_store(self, query, *, session_id, turn_id, scope_level, destination, top_k, project_id=""):
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

    def record_context_turn(self, user_text, assistant_text, **payload):
        self.calls.append(("record_context_turn", {"user_text": user_text, "assistant_text": assistant_text, **payload}))
        return {"memory_id": "session:s1:turn-3"}
    def record_context_checkpoint(self, **payload):
        self.calls.append(("record_context_checkpoint", payload))
        return {"memory_id": "session:s1:checkpoint:abc"}


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
        self.assertEqual(fake.calls[0][0], "retrieve_context_recent")
        self.assertEqual(fake.calls[0][1], "s1")

    def test_prefetch_refreshes_when_current_query_differs_from_cached_query(self):
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            provider.on_turn_start(1, "proxy")
            provider.queue_prefetch("proxy", session_id="s1")
            provider.prefetch("SSH setup", session_id="s1")

        retrievals = [call for call in fake.calls if call[0] == "retrieve_context_recent"]
        # Session mode calls retrieve_context_recent twice (different queries trigger refresh)
        self.assertEqual(len(retrievals), 2)

    def test_session_switch_clears_cached_context_and_rebinds_session(self):
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            provider.on_turn_start(1, "proxy")
            provider.queue_prefetch("proxy", session_id="s1")
            provider.on_session_switch("s2", reset=True)
            provider.on_turn_start(1, "SSH setup")
            context = provider.prefetch("SSH setup", session_id="s2")

        self.assertIn("Use the local proxy configuration.", context)
        retrievals = [call for call in fake.calls if call[0] == "retrieve_context_recent"]
        self.assertEqual(len(retrievals), 2)
        self.assertEqual(retrievals[-1][1], "s2")

    def test_tools_only_generate_write_proposals_and_never_commit(self):
        import os
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            # Set experimental_global mode to enable proposals tool
            old_mode = os.environ.get("MNEMOSYNE_MEMORY_MODE")
            os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
            try:
                provider.initialize("s1", hermes_home=tmp)
                payload = json.loads(
                    provider.handle_tool_call(
                        "mnemosyne_write_proposals",
                        {"text": "Please remember that I prefer Chinese replies."},
                    )
                )
            finally:
                if old_mode is None:
                    os.environ.pop("MNEMOSYNE_MEMORY_MODE", None)
                else:
                    os.environ["MNEMOSYNE_MEMORY_MODE"] = old_mode

        self.assertTrue(payload["proposals"][0]["admissible"])
        self.assertEqual([call[0] for call in fake.calls], ["write_proposals"])
        self.assertNotIn("mnemosyne_remember", [schema["name"] for schema in provider.get_tool_schemas()])

    def test_sync_turn_records_citation_only_when_shadow_enabled(self):
        import os
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            # Use experimental_global mode which provides retrieval_event_id for citations
            old_mode = os.environ.get("MNEMOSYNE_MEMORY_MODE")
            old_shadow = os.environ.get("MNEMOSYNE_EXPERIMENTAL_SHADOW")
            os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
            os.environ["MNEMOSYNE_EXPERIMENTAL_SHADOW"] = "1"
            try:
                provider.initialize("s1", hermes_home=tmp)
                provider.on_turn_start(3, "proxy")
                provider.queue_prefetch("proxy", session_id="s1")
                provider.prefetch("proxy", session_id="s1")
                provider.sync_turn("query", "Use the local proxy configuration.", session_id="s1")
            finally:
                if old_mode is None:
                    os.environ.pop("MNEMOSYNE_MEMORY_MODE", None)
                else:
                    os.environ["MNEMOSYNE_MEMORY_MODE"] = old_mode
                if old_shadow is None:
                    os.environ.pop("MNEMOSYNE_EXPERIMENTAL_SHADOW", None)
                else:
                    os.environ["MNEMOSYNE_EXPERIMENTAL_SHADOW"] = old_shadow

        log_calls = [call for call in fake.calls if call[0] == "log_citation"]
        self.assertEqual(len(log_calls), 1)

        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            provider.sync_turn("remember this", "acknowledged", session_id="s1")

        context_calls = [call for call in fake.calls if call[0] == "record_context_turn"]
        citation_calls = [call for call in fake.calls if call[0] == "log_citation"]
        self.assertEqual(len(context_calls), 1)
        self.assertEqual(citation_calls, [])
    def test_pre_compress_creates_session_checkpoint_without_long_term_promotion(self):
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp)
            provider.on_turn_start(4, "Continue the SQLite design")
            summary = provider.on_pre_compress([
                {"role": "user", "content": "Use SQLite for the session context. " * 500},
                {"role": "assistant", "content": "Confirmed."},
            ], require_checkpoint=True)

        checkpoints = [call for call in fake.calls if call[0] == "record_context_checkpoint"]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0][1]["session_id"], "s1")
        self.assertEqual(checkpoints[0][1]["turn_id"], "turn-4")
        self.assertLessEqual(len(checkpoints[0][1]["transcript"]), 12_000)
        self.assertIn("checkpoint stored", summary)
        self.assertNotIn("write_proposals", [call[0] for call in fake.calls])

    def test_citation_client_maps_provider_memory_ids_to_api_ids(self):
        client = self.module.MnemosyneHTTPClient(base_url="http://example.test")
        captured = {}

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"citations": []}

        def request(_method, _url, **kwargs):
            captured.update(kwargs)
            return Response()

        with patch("requests.request", side_effect=request):
            client.detect_citations("response", [{"memory_id": "m1", "content": "proxy", "rank": 1}])

        self.assertEqual(captured["json"]["injected_memories"], [{"id": "m1", "content": "proxy"}])

    def test_citation_api_accepts_legacy_memory_id_payload(self):
        from pydantic import BaseModel

        class CitationMemory(BaseModel):
            id: str = ""
            memory_id: str = ""
            content: str

            @property
            def resolved_id(self):
                return self.id or self.memory_id

        self.assertEqual(CitationMemory(memory_id="m1", content="proxy").resolved_id, "m1")

    def test_sync_turn_sends_citation_payload_accepted_by_api_contract(self):
        import os
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            # Use experimental_global mode for citations
            old_mode = os.environ.get("MNEMOSYNE_MEMORY_MODE")
            old_shadow = os.environ.get("MNEMOSYNE_EXPERIMENTAL_SHADOW")
            os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
            os.environ["MNEMOSYNE_EXPERIMENTAL_SHADOW"] = "1"
            try:
                provider.initialize("s1", hermes_home=tmp)
                provider.on_turn_start(3, "proxy")
                provider.queue_prefetch("proxy", session_id="s1")
                provider.prefetch("proxy", session_id="s1")
                provider.sync_turn("query", "Use the local proxy configuration.", session_id="s1")
            finally:
                if old_mode is None:
                    os.environ.pop("MNEMOSYNE_MEMORY_MODE", None)
                else:
                    os.environ["MNEMOSYNE_MEMORY_MODE"] = old_mode
                if old_shadow is None:
                    os.environ.pop("MNEMOSYNE_EXPERIMENTAL_SHADOW", None)
                else:
                    os.environ["MNEMOSYNE_EXPERIMENTAL_SHADOW"] = old_shadow

        citation = next(call for call in fake.calls if call[0] == "log_citation")[1]
        self.assertIn("citation", citation)
        self.assertEqual(citation["memory_id"], "m1")
        self.assertEqual(citation["original_rank"], 1)

    def test_sync_turn_records_session_context_without_promoting_long_term_memory(self):
        fake = _FakeClient()
        provider = self.module.MnemosyneMemoryProvider(client_factory=lambda **_: fake)
        with tempfile.TemporaryDirectory() as tmp:
            provider.initialize("s1", hermes_home=tmp, cwd="D:/project/MNEMOSYNE-MNS")
            provider.on_turn_start(3, "Use SQLite for the local store")
            provider.sync_turn("Use SQLite for the local store", "I will use SQLite.", session_id="s1")

        writes = [call for call in fake.calls if call[0] == "record_context_turn"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1]["session_id"], "s1")
        self.assertEqual(writes[0][1]["turn_id"], "turn-3")
        self.assertEqual(writes[0][1]["project_id"], "D:/project/MNEMOSYNE-MNS")
        self.assertNotIn("write_proposals", [call[0] for call in fake.calls])


if __name__ == "__main__":
    unittest.main()
