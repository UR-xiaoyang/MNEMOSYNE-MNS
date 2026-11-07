"""Tests for Provider mode awareness and contract compliance.

Validates that the Provider:
1. Calls /context/recent in session mode (not /retrieve_store)
2. Calls /context/recent + /experimental/global-memories/retrieve in experimental_global mode
3. Returns empty tool schemas in session mode
4. Only exposes proposals tool in experimental_global mode
5. Uses correct headers for each mode
6. Handles unknown mode safely
7. Shadow/citation requires explicit MNEMOSYNE_EXPERIMENTAL_SHADOW=1
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

# Mock agent.memory_provider before importing the provider
mock_agent_module = Mock()
mock_memory_provider_module = Mock()
sys.modules['agent'] = mock_agent_module
sys.modules['agent.memory_provider'] = mock_memory_provider_module

# Create mock classes
class MockMemoryProvider:
    pass

class MockRecallStatus:
    def __init__(self, name, count):
        self.name = name
        self.count = count

def mock_spawn_context_thread(target, name=None):
    # Return None to trigger synchronous execution
    return None

mock_memory_provider_module.MemoryProvider = MockMemoryProvider
mock_memory_provider_module.RecallStatus = MockRecallStatus
mock_memory_provider_module.spawn_context_thread = mock_spawn_context_thread

# Add plugins to path
sys.path.insert(0, str(Path(__file__).parent.parent / "plugins" / "memory" / "mnemosyne"))

# Import after mocking
import __init__ as provider_module

# Patch the base class
provider_module.MemoryProvider = MockMemoryProvider
provider_module.RecallStatus = MockRecallStatus
provider_module.spawn_context_thread = mock_spawn_context_thread

MnemosyneMemoryProvider = provider_module.MnemosyneMemoryProvider


class FakeClient:
    """Fake client that tracks calls."""
    
    def __init__(self, *args, **kwargs):
        self.calls = []
        
    def retrieve_context_recent(self, **kwargs):
        self.calls.append(("retrieve_context_recent", kwargs))
        return {"results": [{"content": "Session context"}], "token_cost": 100}
    
    def retrieve_experimental_global(self, query, **kwargs):
        self.calls.append(("retrieve_experimental_global", {"query": query, **kwargs}))
        return {"results": [{"content": "Global memory", "memory_id": "exp:1", "rank": 1}], "retrieval_event_id": "evt-123"}
    
    def retrieve_store(self, query, **kwargs):
        self.calls.append(("retrieve_store", {"query": query, **kwargs}))
        return {"results": [], "retrieval_event_id": None}
    
    def record_context_turn(self, *args, **kwargs):
        self.calls.append(("record_context_turn", kwargs))
        return {"memory_id": "ctx-turn-1"}
    
    def detect_citations(self, *args, **kwargs):
        self.calls.append(("detect_citations", {}))
        return {"citations": []}
    
    def log_citation(self, *args, **kwargs):
        self.calls.append(("log_citation", {}))
        return {"event_id": "citation-1"}
    
    def write_proposals(self, *args, **kwargs):
        self.calls.append(("write_proposals", {}))
        return {"proposals": []}
    
    def record_context_checkpoint(self, *args, **kwargs):
        self.calls.append(("record_context_checkpoint", {}))
        return {"memory_id": "checkpoint-1"}


class TestProviderModeAwareness(unittest.TestCase):
    """Test Provider mode-aware behavior."""
    
    def setUp(self):
        """Clear environment before each test."""
        for key in ["MNEMOSYNE_MEMORY_MODE", "MNEMOSYNE_EXPERIMENTAL_SHADOW", "MNEMOSYNE_SHADOW_MODE"]:
            os.environ.pop(key, None)
    
    def test_session_mode_default(self):
        """Session mode is default when MNEMOSYNE_MEMORY_MODE is unset."""
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        self.assertEqual(provider._memory_mode, "session")
    
    def test_session_mode_explicit(self):
        """Session mode when MNEMOSYNE_MEMORY_MODE=session."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        self.assertEqual(provider._memory_mode, "session")
    
    def test_experimental_global_mode(self):
        """Experimental global mode when MNEMOSYNE_MEMORY_MODE=experimental_global."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        self.assertEqual(provider._memory_mode, "experimental_global")
    
    def test_unknown_mode_fallback(self):
        """Unknown mode safely falls back to session."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "invalid_mode"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        self.assertEqual(provider._memory_mode, "session")
    
    def test_session_mode_calls_context_recent_only(self):
        """Session mode calls /context/recent, not /retrieve_store."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        provider._pending_query = "test query"
        
        client = provider._client
        provider._prefetch()
        
        # Should call retrieve_context_recent
        call_names = [c[0] for c in client.calls]
        self.assertIn("retrieve_context_recent", call_names)
        self.assertNotIn("retrieve_store", call_names)
        self.assertNotIn("retrieve_experimental_global", call_names)
    
    def test_experimental_global_calls_both(self):
        """Experimental global mode calls /context/recent + /experimental/global-memories/retrieve."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        provider._pending_query = "test query"
        
        client = provider._client
        provider._prefetch()
        
        # Should call both
        call_names = [c[0] for c in client.calls]
        self.assertIn("retrieve_context_recent", call_names)
        self.assertIn("retrieve_experimental_global", call_names)
        self.assertNotIn("retrieve_store", call_names)
    
    def test_session_mode_header(self):
        """Session mode uses '[Previous conversation context]' header."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        provider._pending_query = "test"
        
        provider._prefetch()
        context = provider.prefetch("test")
        
        self.assertIn("[Previous conversation context]", context)
        self.assertNotIn("[MNEMOSYNE Context]", context)
    
    def test_experimental_global_two_headers(self):
        """Experimental global mode has two distinct headers."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        provider._pending_query = "test"
        
        provider._prefetch()
        context = provider.prefetch("test")
        
        self.assertIn("[Previous conversation context]", context)
        self.assertIn("[Experimental cross-conversation memory]", context)
    
    def test_session_mode_no_tool_schemas(self):
        """Session mode returns empty tool schemas."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        schemas = provider.get_tool_schemas()
        self.assertEqual(schemas, [])
    
    def test_experimental_global_exposes_proposals_tool(self):
        """Experimental global mode exposes mnemosyne_write_proposals."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        schemas = provider.get_tool_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "mnemosyne_write_proposals")
    
    def test_session_mode_rejects_proposal_tool_call(self):
        """Session mode rejects mnemosyne_write_proposals calls."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        result = provider.handle_tool_call("mnemosyne_write_proposals", {"text": "test"})
        self.assertIn("error", result)
        self.assertIn("only available in experimental_global", result)
    
    def test_shadow_requires_experimental_switch(self):
        """Shadow/citation requires MNEMOSYNE_EXPERIMENTAL_SHADOW=1, not MNEMOSYNE_SHADOW_MODE."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        os.environ["MNEMOSYNE_SHADOW_MODE"] = "1"  # Old variable should not trigger
        
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        client = provider._client
        provider.sync_turn("user input", "assistant response")
        
        # Should record turn but NOT call detect_citations
        call_names = [c[0] for c in client.calls]
        self.assertIn("record_context_turn", call_names)
        self.assertNotIn("detect_citations", call_names)
    
    def test_shadow_enabled_with_experimental_switch(self):
        """Shadow/citation enabled with MNEMOSYNE_EXPERIMENTAL_SHADOW=1."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
        os.environ["MNEMOSYNE_EXPERIMENTAL_SHADOW"] = "1"
        
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        provider._pending_query = "test"
        provider._prefetch()  # Set up retrieval_event_id
        
        client = provider._client
        provider.sync_turn("user input", "assistant response")
        
        # Should call both record_context_turn and detect_citations
        call_names = [c[0] for c in client.calls]
        self.assertIn("record_context_turn", call_names)
        self.assertIn("detect_citations", call_names)
    
    def test_system_prompt_session_mode(self):
        """Session mode system prompt describes conversation context."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        prompt = provider.system_prompt_block()
        self.assertIn("Conversation Context", prompt)
        self.assertIn("Previous conversation history", prompt)
        self.assertNotIn("memories are injected", prompt.lower())
    
    def test_system_prompt_experimental_global(self):
        """Experimental global mode system prompt mentions both session and global."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "experimental_global"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        
        prompt = provider.system_prompt_block()
        self.assertIn("MNEMOSYNE Memory", prompt)
        self.assertIn("Session context", prompt)
        self.assertIn("approved global memories", prompt)
    
    def test_session_mode_zero_injected_memories(self):
        """Session mode sets injected_memories to empty list."""
        os.environ["MNEMOSYNE_MEMORY_MODE"] = "session"
        provider = MnemosyneMemoryProvider(client_factory=FakeClient)
        provider.initialize("session-123")
        provider._pending_query = "test"
        
        provider._prefetch()
        
        self.assertEqual(provider._injected_memories, [])
        self.assertIsNone(provider._retrieval_event_id)


if __name__ == "__main__":
    unittest.main()
