import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from shadow_mode import ShadowModeCollector
from shadow_runtime import record_citation, record_correction, record_retrieval


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

    def test_citation_and_correction_are_recorded_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = ShadowModeCollector(Path(tmp) / "trace.db")
            retrieval_id = record_retrieval(
                collector, enabled=True, session_id="s1", turn_id="t1", query={"text": "proxy"},
                candidates=[{"memory_id": "m1", "rank": 1, "score": 0.8, "injected": True}], metadata={}
            )
            citation_id = record_citation(
                collector, enabled=True, session_id="s1", turn_id="t1", retrieval_event_id=retrieval_id,
                memory_id="m1", citation={"span": "proxy", "confidence": 0.9}, original_rank=1
            )
            correction_id = record_correction(
                collector, enabled=True, session_id="s1", turn_id="t2", citation_event_id=citation_id,
                memory_id="m1", correction_type="outdated", feedback={"text": "old"}
            )
            self.assertIsInstance(citation_id, str)
            self.assertIsInstance(correction_id, str)
            metrics = collector.compute_metrics("s1")
            self.assertEqual(metrics["cited_count"], 1)
            self.assertEqual(metrics["correction_count"], 1)
            collector.close()


if __name__ == "__main__":
    unittest.main()
