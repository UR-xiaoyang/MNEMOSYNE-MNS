import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from context_memory import ContextMemoryService
from memory_store import MemoryStore


class ContextMemoryServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.db")
        self.service = ContextMemoryService(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_session_context_is_recalled_only_inside_same_session(self):
        self.service.record_turn(
            session_id="s1", project_id="p1", user_text="We chose SQLite", assistant_text="Noted."
        )
        same = self.service.retrieve_context("database choice", session_id="s1", project_id="p1")
        other = self.service.retrieve_context("database choice", session_id="s2", project_id="p1")

        self.assertTrue(any(item["memory_id"].startswith("session:s1:") for item in same["results"]))
        self.assertFalse(any(item["memory_id"].startswith("session:s1:") for item in other["results"]))

    def test_context_is_not_promoted_to_global_without_explicit_commit(self):
        self.service.record_turn(
            session_id="s1", project_id="p1", user_text="Temporary build status", assistant_text="Acknowledged."
        )
        global_context = self.service.retrieve_context("temporary build status", session_id="s1", project_id="p1", scope_level="global")
        self.assertFalse(any(item["memory_id"].startswith("session:s1:") for item in global_context["results"]))

    def test_record_turn_is_idempotent_for_turn_id(self):
        first = self.service.record_turn(
            session_id="s1", project_id="p1", turn_id="t1", user_text="Use SQLite", assistant_text="Okay."
        )
        second = self.service.record_turn(
            session_id="s1", project_id="p1", turn_id="t1", user_text="Use SQLite", assistant_text="Okay."
        )
        self.assertEqual(first, second)
        rows = self.store.search_candidates("SQLite", scope_level="session", destination="local", session_id="s1")
        self.assertEqual(len(rows), 1)

    def test_checkpoint_is_idempotent_and_session_scoped(self):
        first = self.service.record_checkpoint(
            session_id="s1", turn_id="t5", project_id="p1", transcript="Decision: use SQLite."
        )
        second = self.service.record_checkpoint(
            session_id="s1", turn_id="t5", project_id="p1", transcript="Decision: use SQLite."
        )
        same = self.service.retrieve_context("SQLite", session_id="s1", project_id="p1")
        other = self.service.retrieve_context("SQLite", session_id="s2", project_id="p1")

        self.assertEqual(first, second)
        self.assertTrue(any(item["memory_id"] == first for item in same["results"]))
        self.assertFalse(any(item["memory_id"] == first for item in other["results"]))

    def test_new_checkpoint_supersedes_prior_checkpoint_in_same_session(self):
        old = self.service.record_checkpoint(
            session_id="s1", turn_id="t1", project_id="p1", transcript="Decision: use SQLite."
        )
        current = self.service.record_checkpoint(
            session_id="s1", turn_id="t2", project_id="p1", transcript="Decision: use PostgreSQL."
        )
        context = self.service.retrieve_context("Decision", session_id="s1", project_id="p1", top_k=10)

        self.assertNotEqual(old, current)
        self.assertEqual(self.store.get(old).validity.state, "superseded")
        self.assertEqual([item["memory_id"] for item in context["results"]], [current])

    def test_context_budget_limits_injected_items(self):
        for i in range(5):
            self.service.record_turn(
                session_id="s1", project_id="p1", turn_id=f"t{i}",
                user_text=f"Important database decision number {i}", assistant_text="Recorded."
            )
        context = self.service.retrieve_context("database decision", session_id="s1", project_id="p1", max_tokens=20, tokens_per_memory=10)
        self.assertLessEqual(len(context["results"]), 2)
        self.assertLessEqual(context["token_cost"], 20)


if __name__ == "__main__":
    unittest.main()
