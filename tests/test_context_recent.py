import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from context_memory import ContextMemoryService
from memory_store import MemoryStore


class ContextRecentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.db")
        self.service = ContextMemoryService(self.store)
    
    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()
    
    def test_retrieve_recent_requires_session_id(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.retrieve_recent(session_id="")
        self.assertIn("session_id", str(ctx.exception))
    
    def test_retrieve_recent_returns_chronological_order(self):
        self.service.record_turn(session_id="s1", project_id="p1", turn_id="t1", user_text="First", assistant_text="OK")
        self.service.record_turn(session_id="s1", project_id="p1", turn_id="t2", user_text="Second", assistant_text="Sure")
        self.service.record_turn(session_id="s1", project_id="p1", turn_id="t3", user_text="Third", assistant_text="Done")
        
        result = self.service.retrieve_recent(session_id="s1", project_id="p1")
        contents = [r["content"] for r in result["results"]]
        
        self.assertEqual(len(contents), 3)
        self.assertIn("First", contents[0])
        self.assertIn("Second", contents[1])
        self.assertIn("Third", contents[2])
    
    def test_retrieve_recent_excludes_other_sessions(self):
        self.service.record_turn(session_id="s1", project_id="p1", user_text="Session 1", assistant_text="OK")
        self.service.record_turn(session_id="s2", project_id="p1", user_text="Session 2", assistant_text="OK")
        
        result = self.service.retrieve_recent(session_id="s1", project_id="p1")
        
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("Session 1", result["results"][0]["content"])
        self.assertNotIn("Session 2", result["results"][0]["content"])
    
    def test_retrieve_recent_excludes_checkpoints(self):
        self.service.record_turn(session_id="s1", project_id="p1", user_text="Turn", assistant_text="OK")
        self.service.record_checkpoint(session_id="s1", turn_id="t1", project_id="p1", transcript="Checkpoint text")
        
        result = self.service.retrieve_recent(session_id="s1", project_id="p1")
        
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("Turn", result["results"][0]["content"])
        self.assertNotIn("Checkpoint", result["results"][0]["content"])
    
    def test_retrieve_recent_respects_token_budget(self):
        for i in range(10):
            self.service.record_turn(
                session_id="s1", project_id="p1", turn_id=f"t{i}",
                user_text=f"Question {i}" * 10, assistant_text=f"Answer {i}" * 10
            )
        
        result = self.service.retrieve_recent(session_id="s1", project_id="p1", max_tokens=100, tokens_per_memory=50)
        
        # Token estimation is approximate, so we check it's reasonably close to the budget
        self.assertLessEqual(result["token_cost"], 150)  # Allow some margin
        self.assertGreater(len(result["results"]), 0)
        self.assertLess(len(result["results"]), 10)  # Should not return all 10 turns
    
    def test_retrieve_recent_returns_empty_for_new_session(self):
        result = self.service.retrieve_recent(session_id="new-session", project_id="p1")
        
        self.assertEqual(result["results"], [])
        self.assertEqual(result["token_cost"], 0)
    
    def test_retrieve_recent_includes_created_at_timestamps(self):
        self.service.record_turn(session_id="s1", project_id="p1", user_text="Test", assistant_text="OK")
        
        result = self.service.retrieve_recent(session_id="s1", project_id="p1")
        
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("created_at", result["results"][0])
        self.assertIsInstance(result["results"][0]["created_at"], float)


if __name__ == "__main__":
    unittest.main()
