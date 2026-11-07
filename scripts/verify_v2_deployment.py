#!/usr/bin/env python3
"""
Production deployment verification for MNEMOSYNE-MNS V2.

This script validates the complete deployment chain:
1. Model files exist and are loadable
2. Feature extraction works correctly
3. Inference produces valid scores
4. Pipeline integrates properly
5. Server can start with V2

Usage:
    python scripts/verify_v2_deployment.py
    python scripts/verify_v2_deployment.py --verbose
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def check_model_files():
    """Verify V2 model files exist."""
    print("1. Checking model files...")
    
    required_files = [
        PROJECT_ROOT / "deploy/rank_enhanced_v2/enhanced_ranker.joblib",
        PROJECT_ROOT / "deploy/rank_enhanced_v2/metadata.json",
        PROJECT_ROOT / "deploy/rank_enhanced_v2/validation_report.json",
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"   ✗ Missing: {file_path}")
            return False
        print(f"   ✓ {file_path.name}")
    
    return True


def check_model_loading():
    """Verify V2 model can be loaded."""
    print("\n2. Testing model loading...")
    
    try:
        from rank_inference import RankingBaseline
        
        ranker = RankingBaseline(PROJECT_ROOT / "deploy/rank_enhanced_v2", device="cpu")
        
        if ranker.model_id != "MNEMOSYNE-MNS-RANK-ENHANCED-V2":
            print(f"   ✗ Wrong model_id: {ranker.model_id}")
            return False
        
        print(f"   ✓ Model ID: {ranker.model_id}")
        print(f"   ✓ Encoder: {ranker.encoder_name}")
        print(f"   ✓ NDCG@10: {ranker.metadata.get('ndcg10', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False


def check_inference():
    """Verify inference produces valid scores."""
    print("\n3. Testing inference...")
    
    try:
        from rank_inference import RankingBaseline
        import numpy as np
        
        ranker = RankingBaseline(PROJECT_ROOT / "deploy/rank_enhanced_v2", device="cpu")
        
        queries = [
            "如何配置代理服务器？",
            "What is memory ranking?"
        ]
        memories = [
            "在 git config 中设置 http.proxy 和 https.proxy",
            "Memory ranking sorts retrieved memories by relevance score"
        ]
        
        scores = ranker.score(queries, memories)
        
        if not isinstance(scores, np.ndarray):
            print(f"   ✗ Scores not ndarray: {type(scores)}")
            return False
        
        if scores.shape != (2,):
            print(f"   ✗ Wrong shape: {scores.shape}")
            return False
        
        if not all(np.isfinite(scores)):
            print(f"   ✗ Non-finite scores: {scores}")
            return False
        
        print(f"   ✓ Scores shape: {scores.shape}")
        print(f"   ✓ Score values: [{scores[0]:.4f}, {scores[1]:.4f}]")
        print(f"   ✓ All scores finite")
        
        return True
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pipeline():
    """Verify pipeline integration."""
    print("\n4. Testing pipeline integration...")
    
    try:
        from memory_pipeline import create_pipeline
        
        pipeline = create_pipeline("v2")
        
        if pipeline.ranker.model_id != "MNEMOSYNE-MNS-RANK-ENHANCED-V2":
            print(f"   ✗ Wrong pipeline model: {pipeline.ranker.model_id}")
            return False
        
        print(f"   ✓ Pipeline created with V2")
        print(f"   ✓ Ranker: {pipeline.ranker.model_id}")
        
        return True
    except Exception as e:
        print(f"   ✗ Pipeline creation failed: {e}")
        return False


def check_deployment_config():
    """Verify deployment configuration."""
    print("\n5. Testing deployment config...")
    
    try:
        from deployment_config import get_ranker_config, create_production_pipeline
        
        config = get_ranker_config("v2")
        
        if config["model_id"] != "MNEMOSYNE-MNS-RANK-ENHANCED-V2":
            print(f"   ✗ Wrong config model_id: {config['model_id']}")
            return False
        
        if config["ndcg10"] != 0.8057:
            print(f"   ✗ Wrong NDCG@10: {config['ndcg10']}")
            return False
        
        print(f"   ✓ Config: {config['model_id']}")
        print(f"   ✓ NDCG@10: {config['ndcg10']}")
        
        # Test production pipeline creation
        prod_pipeline = create_production_pipeline("v2")
        print(f"   ✓ Production pipeline created")
        
        return True
    except Exception as e:
        print(f"   ✗ Deployment config failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("MNEMOSYNE-MNS V2 Deployment Verification")
    print("=" * 60)
    
    start_time = time.time()
    
    checks = [
        ("Model Files", check_model_files),
        ("Model Loading", check_model_loading),
        ("Inference", check_inference),
        ("Pipeline", check_pipeline),
        ("Deployment Config", check_deployment_config),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            results.append((name, False))
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\nPassed: {passed_count}/{total}")
    print(f"Elapsed: {elapsed:.2f}s")
    
    if passed_count == total:
        print("\n🎉 All checks passed! V2 is ready for production.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
