import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from shadow_mode import ShadowModeCollector
from shadow_runtime import record_retrieval


class ShadowRuntimeTest(unittest.TestCase):
    def test_record_retrieval_returns_event_id_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = ShadowModeCollector(Path(tmp) / "trace.db")
            event_id = record_retrieval(
                collector,
                enabled=True,
                session_id="s1",
                turn_id="t1",
                query={"text": "proxy"},
                candidates=[{"memory_id": "m1", "rank": 1, "score": 0.8, "injected": True}],
                metadata={"ranker_model_id": "RANK-HYBRID-01"},
            )
            self.assertIsInstance(event_id, str)
            self.assertEqual(collector.compute_metrics("s1")["injected_count"], 1)
            collector.close()

    def test_record_retrieval_is_noop_when_disabled(self):
        event_id = record_retrieval(
            None,
            enabled=False,
            session_id="s1",
            turn_id="t1",
            query={"text": "proxy"},
            candidates=[],
            metadata={},
        )
        self.assertIsNone(event_id)


if __name__ == "__main__":
    unittest.main()
