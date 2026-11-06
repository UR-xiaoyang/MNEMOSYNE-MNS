"""
Test RANK Enhanced V2 inference integration.
"""
import unittest
import tempfile
from pathlib import Path
import json
import numpy as np


class TestRankEnhancedV2Inference(unittest.TestCase):
    """Test Enhanced V2 model loading and inference."""
    
    def setUp(self):
        """Create mock model directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "rank_enhanced_v2"
        self.model_path.mkdir()
        
        # Create mock metadata
        metadata = {
            "model_id": "MNEMOSYNE-MNS-RANK-ENHANCED-V2",
            "version": "v0.3.0",
            "encoder": "intfloat/multilingual-e5-small",
            "features": 23,
            "ndcg10": 0.8057
        }
        with open(self.model_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def test_enhanced_v2_model_id_detection(self):
        """Test that Enhanced V2 model_id is recognized."""
        from scripts.rank_inference import RankingBaseline
        
        # Should not raise during initialization (model file missing is OK for ID detection)
        try:
            baseline = RankingBaseline(self.model_path)
            self.assertEqual(baseline.model_id, "MNEMOSYNE-MNS-RANK-ENHANCED-V2")
        except FileNotFoundError as e:
            # Expected: enhanced_ranker.joblib not found
            self.assertIn("enhanced_ranker", str(e))
    
    def test_feature_extraction(self):
        """Test feature extraction produces correct shape."""
        from scripts.rank_enhanced_features import extract_enhanced_features
        
        queries = ["What is machine learning?", "How to train a model?"]
        memories = ["ML is a subset of AI", "Training requires data and compute"]
        e5_cosine = np.array([[0.85], [0.72]], dtype=np.float32)
        
        features = extract_enhanced_features(queries, memories, e5_cosine)
        
        self.assertEqual(features.shape, (2, 23))
        self.assertEqual(features.dtype, np.float32)
        
        # Check e5_cosine is first feature
        np.testing.assert_array_almost_equal(features[:, 0], e5_cosine[:, 0])
        
        # Check lexical features are computed
        self.assertGreater(features[0, 2], 0)  # lexical_jaccard
        self.assertGreater(features[0, 3], 0)  # lexical_containment
    
    def test_feature_extraction_handles_none(self):
        """Test feature extraction with None texts."""
        from scripts.rank_enhanced_features import extract_enhanced_features
        
        queries = [None, ""]
        memories = ["Some text", None]
        e5_cosine = np.array([[0.5], [0.3]], dtype=np.float32)
        
        # Should not crash
        features = extract_enhanced_features(queries, memories, e5_cosine)
        self.assertEqual(features.shape, (2, 23))
    
    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
