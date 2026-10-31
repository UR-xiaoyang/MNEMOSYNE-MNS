import tempfile
import threading
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shadow_mode import ShadowModeCollector


class ShadowThreadSafetyTest(unittest.TestCase):
    def test_concurrent_trace_writes_are_serialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = ShadowModeCollector(Path(tmp) / "trace.db")
            errors = []

            def write(index):
                try:
                    collector.log_retrieval(
                        "s1",
                        f"t{index}",
                        {"text": "q"},
                        [{"memory_id": f"m{index}", "injected": True}],
                        {},
                    )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(collector.compute_metrics("s1")["injected_count"], 8)
            collector.close()

        with tempfile.TemporaryDirectory() as tmp:
            collector = ShadowModeCollector(Path(tmp) / "trace.db")
            collector.log_retrieval(
                "s1",
                "t1",
                {"text": "q"},
                [{"memory_id": "m1", "injected": True}],
                {},
            )
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
