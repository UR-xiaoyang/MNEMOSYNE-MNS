import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from context_assembly import assemble_context


def result(memory_id, content, score, rank, state="active", conflict_with=None, token_cost=None):
    item = {
        "memory_id": memory_id,
        "content": content,
        "score": score,
        "rank": rank,
        "injected": True,
        "metadata": {
            "validity_state": state,
            "conflict_with": conflict_with or [],
        },
    }
    if token_cost is not None:
        item["token_cost"] = token_cost
    return item


class ContextAssemblyTest(unittest.TestCase):
    def test_budget_limits_injected_context(self):
        ranked = [
            result("m1", "first", 0.9, 1, token_cost=4),
            result("m2", "second", 0.8, 2, token_cost=4),
            result("m3", "third", 0.7, 3, token_cost=4),
        ]
        assembled = assemble_context(ranked, max_tokens=8, tokens_per_memory=4)
        self.assertEqual([item["memory_id"] for item in assembled], ["m1", "m2"])
        self.assertTrue(all(item["injected"] for item in assembled))

    def test_near_duplicate_keeps_highest_ranked_memory(self):
        ranked = [
            result("m1", "configure proxy with git config", 0.9, 1),
            result("m2", "configure proxy with git config", 0.8, 2),
            result("m3", "use ssh keys", 0.7, 3),
        ]
        assembled = assemble_context(ranked, max_tokens=100, tokens_per_memory=10)
        self.assertEqual([item["memory_id"] for item in assembled], ["m1", "m3"])

    def test_conflicted_and_superseded_memories_are_not_injected(self):
        ranked = [
            result("bad", "old rule", 0.95, 1, state="conflicted"),
            result("old", "superseded rule", 0.9, 2, state="superseded"),
            result("good", "current rule", 0.8, 3),
        ]
        assembled = assemble_context(ranked, max_tokens=100, tokens_per_memory=10)
        self.assertEqual([item["memory_id"] for item in assembled], ["good"])

    def test_conflict_links_exclude_linked_memory(self):
        ranked = [
            result("m1", "one version", 0.9, 1, conflict_with=["m2"]),
            result("m2", "another version", 0.8, 2, state="conflicted"),
            result("m3", "unrelated fact", 0.7, 3),
        ]
        assembled = assemble_context(ranked, max_tokens=100, tokens_per_memory=10)
        self.assertEqual([item["memory_id"] for item in assembled], ["m3"])


if __name__ == "__main__":
    unittest.main()
