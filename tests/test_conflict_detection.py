import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conflict_detection import detect_conflict, find_conflicts
from memory_schema import memory_from_api


class ConflictDetectionTest(unittest.TestCase):
    def test_explicit_negation_conflict_is_detected(self):
        left = memory_from_api("m1", "必须使用 Python 3.11", destination="local")
        right = memory_from_api("m2", "不要使用 Python 3.11", destination="local")
        proposal = detect_conflict(left, right)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.reason, "opposite_negation")
        self.assertGreaterEqual(proposal.confidence, 0.8)

    def test_unrelated_memories_are_not_conflicted(self):
        left = memory_from_api("m1", "配置 git 代理", destination="local")
        right = memory_from_api("m2", "使用 SSH 登录服务器", destination="local")
        self.assertIsNone(detect_conflict(left, right))

    def test_same_content_is_not_conflicted(self):
        left = memory_from_api("m1", "必须使用 Python 3.11", destination="local")
        right = memory_from_api("m2", "必须使用 Python 3.11", destination="local")
        self.assertIsNone(detect_conflict(left, right))

    def test_find_conflicts_returns_only_conflicting_pairs(self):
        base = memory_from_api("base", "必须使用 Python 3.11", destination="local")
        other = memory_from_api("other", "不要使用 Python 3.11", destination="local")
        unrelated = memory_from_api("unrelated", "使用 SSH 登录", destination="local")
        proposals = find_conflicts([base, other, unrelated])
        self.assertEqual([(p.left_id, p.right_id) for p in proposals], [("base", "other")])


if __name__ == "__main__":
    unittest.main()
