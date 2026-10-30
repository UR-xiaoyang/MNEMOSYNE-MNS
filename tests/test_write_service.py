import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from write_gate import proposal_payload


class WriteServiceContractTest(unittest.TestCase):
    def test_proposal_payload_is_json_serializable_and_does_not_commit(self):
        payload = proposal_payload("请记住：我偏好使用中文回复。", scope_level="project")
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["kind"], "preference")
        self.assertTrue(payload[0]["admissible"])
        self.assertEqual(payload[0]["scope_level"], "project")
        self.assertNotIn("memory_id", payload[0])


if __name__ == "__main__":
    unittest.main()
