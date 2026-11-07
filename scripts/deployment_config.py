#!/usr/bin/env python3
"""
Production deployment configuration for MNEMOSYNE-MNS RANK Enhanced V2.

Usage:
    from deployment_config import get_ranker_config, create_production_pipeline
    
    config = get_ranker_config("v2")
    pipeline = create_production_pipeline(config)
"""

from pathlib import Path
from typing import Literal

RankerVersion = Literal["hybrid", "v2"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANKER_CONFIGS = {
    "hybrid": {
        "model_id": "MNEMOSYNE-MNS-RANK-HYBRID-01",
        "path": PROJECT_ROOT / "deploy/model_bundle",
        "ndcg10": 0.5796,
        "features": 13,
        "description": "Baseline hybrid ranker (E5 + 12 structured features)",
    },
    "v2": {
        "model_id": "MNEMOSYNE-MNS-RANK-ENHANCED-V2",
        "path": PROJECT_ROOT / "deploy/rank_enhanced_v2",
        "ndcg10": 0.8057,
        "features": 23,
        "description": "Enhanced V2 ranker with hard negative mining (+39% NDCG@10)",
    },
}


def get_ranker_config(version: RankerVersion = "v2") -> dict:
    """
    Get ranker configuration for deployment.
    
    Args:
        version: "hybrid" (baseline) or "v2" (enhanced, default)
    
    Returns:
        Configuration dict with model_id, path, metrics
    """
    if version not in RANKER_CONFIGS:
        raise ValueError(f"Unknown ranker version: {version}. Available: {list(RANKER_CONFIGS.keys())}")
    
    config = RANKER_CONFIGS[version].copy()
    
    # Validate model path exists
    if not config["path"].exists():
        raise FileNotFoundError(
            f"Ranker model not found at {config['path']}. "
            f"Run deployment setup first."
        )
    
    return config


def create_production_pipeline(
    ranker_version: RankerVersion = "v2",
    device: str = "cpu",
    enable_faiss: bool = False,
):
    """
    Create production-ready MemoryPipeline with specified ranker.
    
    Args:
        ranker_version: "hybrid" or "v2"
        device: "cpu" or "cuda"
        enable_faiss: Enable FAISS semantic recall
    
    Returns:
        Configured MemoryPipeline instance
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    
    from memory_pipeline import MemoryPipeline
    
    config = get_ranker_config(ranker_version)
    
    faiss_config = {}
    if enable_faiss:
        faiss_config = {
            "faiss_index_path": PROJECT_ROOT / "deploy/faiss_index.bin",
            "e5_model_path": PROJECT_ROOT / "deploy/e5_model",
        }
    
    pipeline = MemoryPipeline(
        ranker_path=config["path"],
        device=device,
        enable_faiss=enable_faiss,
        **faiss_config,
    )
    
    return pipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MNEMOSYNE-MNS Ranker Deployment Config")
    parser.add_argument(
        "--version",
        choices=["hybrid", "v2"],
        default="v2",
        help="Ranker version to deploy",
    )
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    
    args = parser.parse_args()
    
    config = get_ranker_config(args.version)
    print(f"✓ Ranker: {config['model_id']}")
    print(f"✓ Path: {config['path']}")
    print(f"✓ NDCG@10: {config['ndcg10']:.4f}")
    print(f"✓ Features: {config['features']}")
    print(f"✓ {config['description']}")
    
    if args.test:
        print("\nRunning smoke test...")
        pipeline = create_production_pipeline(args.version)
        
        from memory_schema import MemoryEntity, Scope, Authority, Validity, Retrieval, Usage, Egress, Provenance
        import time
        
        now = time.time()
        test_memories = [
            MemoryEntity(
                memory_id="m1",
                content="在 git config 中设置 http.proxy 和 https.proxy 可以配置代理",
                created_at=now,
                updated_at=now,
                version=1,
                scope=Scope(level="global"),
                authority=Authority(class_="user_persistent", priority=70, source="user_explicit", reliability=0.9),
                validity=Validity(state="active"),
                retrieval=Retrieval(path="tech/git/config"),
                usage=Usage(),
                egress=Egress(allowed_destinations=["local"]),
                provenance=Provenance(created_by="user"),
            ),
            MemoryEntity(
                memory_id="m2",
                content="Python uses pip for package management",
                created_at=now,
                updated_at=now,
                version=1,
                scope=Scope(level="global"),
                authority=Authority(class_="user_persistent", priority=70, source="user_explicit", reliability=0.9),
                validity=Validity(state="active"),
                retrieval=Retrieval(path="tech/python/package"),
                usage=Usage(),
                egress=Egress(allowed_destinations=["local"]),
                provenance=Provenance(created_by="user"),
            ),
        ]
        
        results = pipeline.retrieve_and_rank(
            query_text="如何配置代理？",
            candidate_memories=test_memories,
            top_k=2,
        )
        
        print(f"\n✅ Test passed! Retrieved {len(results)} ranked memories")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['memory_id']}] score={result['score']:.4f}")
