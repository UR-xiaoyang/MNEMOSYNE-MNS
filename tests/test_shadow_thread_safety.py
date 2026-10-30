import tempfile
import threading
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shadow_mode import ShadowModeCollector


class ShadowThreadSafetyTest(unittest.TestCase):
    def test_metrics_can_be_read_from_worker_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = ShadowModeCollector(Path(tmp) / "trace.db")
            collector.log_retrieval("s1", "t1", {"text": "q"}, [], {})
            result = []
            error = []

            def read_metrics():
                try:
                    result.append(collector.compute_metrics("s1"))
                except Exception as exc:
                    error.append(exc)

            thread = threading.Thread(target=read_metrics)
            thread.start()
            thread.join()
            self.assertEqual(error, [])
            self.assertEqual(result[0]["injected_count"], 1)
            collector.close()


if __name__ == "__main__":
    unittest.main()
