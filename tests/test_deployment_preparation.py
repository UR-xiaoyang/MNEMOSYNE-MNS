import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_deployment import prepare_deployment


class DeploymentPreparationTest(unittest.TestCase):
    def test_prepare_deployment_creates_manifest_and_required_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            vectorizer_path = root / "vectorizer.joblib"
            e5_source = root / "e5"
            destination = root / "deploy"
            source.mkdir()
            e5_source.mkdir()

            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
            vectorizer.fit(["proxy configuration", "ssh configuration"])
            joblib.dump(vectorizer, vectorizer_path)
            ranker = LogisticRegression().fit(np.ones((2, 13)), [0, 1])
            artifact = {
                "scaler": StandardScaler().fit(np.ones((2, 13))),
                "ranker": ranker,
                "feature_names": ["e5_cosine"] + [f"feature_{i}" for i in range(12)],
                "e5_model": "local-test-model",
            }
            joblib.dump(artifact, source / "hybrid_ranker.joblib")
            (source / "metadata.json").write_text(
                '{"model_id":"MNEMOSYNE-MNS-RANK-HYBRID-01"}', encoding="utf-8"
            )
            for filename in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
                (e5_source / filename).write_bytes(filename.encode("utf-8"))

            manifest = prepare_deployment(source, vectorizer_path, e5_source, destination)

            self.assertEqual(manifest["model_id"], "MNEMOSYNE-MNS-RANK-HYBRID-01")
            self.assertTrue((destination / "model_bundle" / "hybrid_ranker.joblib").is_file())
            self.assertTrue((destination / "e5_model" / "model.safetensors").is_file())
            manifest_path = destination / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(loaded["files"]), 5)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in loaded["files"]))


if __name__ == "__main__":
    unittest.main()
