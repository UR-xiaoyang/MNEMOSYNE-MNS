import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from write_gate import extract_write_proposals


class WriteGateTest(unittest.TestCase):
    def test_explicit_preference_becomes_high_confidence_project_candidate(self):
        proposals = extract_write_proposals("请记住：我偏好使用中文回复。", scope_level="project")
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.kind, "preference")
        self.assertEqual(proposal.scope_level, "project")
        self.assertGreaterEqual(proposal.confidence, 0.8)
        self.assertTrue(proposal.admissible)

    def test_question_and_request_are_not_long_term_candidates(self):
        proposals = extract_write_proposals("如何配置代理服务器？请帮我现在执行。")
        self.assertEqual(proposals, [])

    def test_constraint_is_kept_as_explicit_candidate(self):
        proposals = extract_write_proposals("以后不要把私有数据上传到 Git。", scope_level="global")
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].kind, "constraint")
        self.assertTrue(proposals[0].admissible)
        self.assertIn("不要", proposals[0].content)

    def test_credentials_are_rejected(self):
        fake_key = "sk-test-" + "abcdefghijklmnopqrstuvwxyz123456"
        proposals = extract_write_proposals("请记住 API key: " + fake_key)
        self.assertEqual(len(proposals), 1)
        self.assertFalse(proposals[0].admissible)
        self.assertEqual(proposals[0].rejection_reason, "sensitive_content")

    def test_transient_task_state_is_session_scoped_and_not_admitted(self):
        proposals = extract_write_proposals("现在正在运行一次临时训练任务。", scope_level="project")
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].kind, "task_state")
        self.assertEqual(proposals[0].scope_level, "session")
        self.assertFalse(proposals[0].admissible)


if __name__ == "__main__":
    unittest.main()
