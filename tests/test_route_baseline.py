import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from route_baseline import route_query


class RouteBaselineTest(unittest.TestCase):
    def test_engineering_query_routes_to_code_path(self):
        decision = route_query("如何配置 git 的代理服务器？")
        self.assertEqual(decision.primary_path, "engineering/code")
        self.assertIn("engineering/code", decision.candidate_paths)
        self.assertFalse(decision.force_slow_fallback)
        self.assertGreaterEqual(decision.confidence_proxy, 0.5)

    def test_preference_query_routes_to_user_preference(self):
        decision = route_query("请记住我偏好使用中文回复。")
        self.assertEqual(decision.primary_path, "user/preference")
        self.assertEqual(decision.query_kind, "preference")
        self.assertFalse(decision.force_slow_fallback)

    def test_reference_query_forces_slow_fallback(self):
        decision = route_query("继续上次关于记忆调度的方案")
        self.assertTrue(decision.force_slow_fallback)
        self.assertEqual(decision.fallback_reason, "reference_or_history")
        self.assertIn("task/state", decision.candidate_paths)

    def test_constraint_query_has_high_authority_route(self):
        decision = route_query("以后禁止把私有数据上传到云端")
        self.assertEqual(decision.primary_path, "user/constraint")
        self.assertTrue(decision.requires_high_authority)
        self.assertEqual(decision.query_kind, "constraint")

    def test_ambiguous_query_is_conservative(self):
        decision = route_query("那个怎么办？")
        self.assertEqual(decision.primary_path, "unclassified")
        self.assertTrue(decision.force_slow_fallback)
        self.assertEqual(decision.fallback_reason, "low_route_confidence")


if __name__ == "__main__":
    unittest.main()
