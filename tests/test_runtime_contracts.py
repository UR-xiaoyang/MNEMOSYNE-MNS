import math
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from memory_schema import memory_from_api
from memory_pipeline import resolve_model_path
from structured_features import build_structured_features


class RuntimeContractsTest(unittest.TestCase):
    def test_structured_features_have_training_shape_and_are_finite(self):
        features, names = build_structured_features(
            ["configure proxy in git config"],
            ["configure proxy with git config and http.proxy"],
        )
        self.assertEqual(features.shape, (1, 12))
        self.assertEqual(len(names), 12)
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[0, names.index("lexical_jaccard")], 0.0)

    def test_structured_features_are_deterministic(self):
        first, names_first = build_structured_features(
            ["continue the proxy setup"],
            ["previously we configured HTTP_PROXY"],
        )
        second, names_second = build_structured_features(
            ["continue the proxy setup"],
            ["previously we configured HTTP_PROXY"],
        )
        self.assertEqual(names_first, names_second)
        np.testing.assert_array_equal(first, second)

    def test_api_memory_preserves_external_id_without_randomness(self):
        first = memory_from_api("memory-42", "Use the project proxy", destination="local")
        second = memory_from_api("memory-42", "Use the project proxy", destination="local")
        self.assertEqual(first.memory_id, "memory-42")
        self.assertEqual(first.memory_id, second.memory_id)
        self.assertEqual(first.content, second.content)
        self.assertEqual(first.usage.retrieved_count, 0)
        self.assertEqual(first.authority.source, "imported")

    def test_bundle_export_contains_vectorizer_for_training_parity(self):
        from export_ranker_bundle import export_ranker
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            destination = Path(tmp) / "bundle"
            source.mkdir()
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
            vectorizer.fit(["proxy configuration", "ssh configuration"])
            artifact = {
                "scaler": StandardScaler().fit(np.ones((2, 13))),
                "ranker": LogisticRegression().fit(np.ones((2, 13)), [0, 1]),
                "feature_names": ["e5_cosine"] + [f"feature_{i}" for i in range(12)],
                "e5_model": "local-test-model",
                "vectorizer": vectorizer,
            }
            joblib.dump(artifact, source / "hybrid_ranker.joblib")
            (source / "metadata.json").write_text(
                '{"model_id":"MNEMOSYNE-MNS-RANK-HYBRID-01"}', encoding="utf-8"
            )
            export_ranker(source, destination)
            exported = joblib.load(destination / "hybrid_ranker.joblib")
            self.assertIsNotNone(exported.get("vectorizer"))



if __name__ == "__main__":
    unittest.main()
