"""
评估双路召回系统（FTS + FAISS）的性能

对比指标：
- Recall@100: 相关记忆的召回率
- Latency: 召回延迟
- Memory usage: 内存占用
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import psutil

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_store import MemoryStore
from dual_recall_store import DualRecallStore
from memory_schema import MemoryEntity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_recall(
    db_path: str | Path,
    faiss_index_path: str | Path | None,
    e5_model_path: str | Path,
    test_queries: list[str],
    ground_truth: dict[str, list[str]] | None = None,
    device: str = "cpu",
) -> dict:
    """
    评估双路召回性能
    
    Args:
        db_path: SQLite 数据库路径
        faiss_index_path: FAISS 索引路径（如果为 None，仅使用 FTS）
        e5_model_path: E5 模型路径
        test_queries: 测试查询列表
        ground_truth: 查询 -> 相关 memory_id 列表的映射（用于计算召回率）
        device: 计算设备
    
    Returns:
        评估结果字典
    """
    process = psutil.Process()
    
    # Baseline: FTS only (with strict matching - no fallback to all memories)
    print("\n=== Evaluating FTS Baseline ===")
    
    from memory_store import _from_payload, _fts_query
    import json
    
    fts_store = MemoryStore(db_path)
    
    fts_results = []
    fts_latencies = []
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    for query in test_queries:
        start_time = time.perf_counter()
        
        # Use strict FTS matching without fallback to all memories
        # This matches the _search_fts implementation in DualRecallStore
        fetch_limit = 1000
        rows = fts_store.connection.execute(
            """
            SELECT m.payload_json
            FROM memories_fts f
            JOIN memories m ON m.memory_id = f.memory_id
            WHERE memories_fts MATCH ?
            ORDER BY m.updated_at DESC
            LIMIT ?
            """,
            (_fts_query(query), fetch_limit),
        ).fetchall()
        
        candidates = [_from_payload(json.loads(row["payload_json"])) for row in rows]
        
        # If FTS fails, try LIKE as fallback
        if not candidates:
            like_query = f"%{query.strip()}%"
            rows = fts_store.connection.execute(
                "SELECT payload_json FROM memories WHERE content LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (like_query, fetch_limit),
            ).fetchall()
            candidates = [_from_payload(json.loads(row["payload_json"])) for row in rows]
        
        # Filter by scope
        candidates = [
            c for c in candidates
            if c.hard_filter("global", "local", False)
        ][:100]
        
        latency = (time.perf_counter() - start_time) * 1000  # ms
        fts_latencies.append(latency)
        fts_results.append([c.memory_id for c in candidates])
    
    memory_after_fts = process.memory_info().rss / 1024 / 1024  # MB
    fts_store.close()
    
    # Dual recall: FTS + FAISS
    print("\n=== Evaluating Dual Recall (FTS + FAISS) ===")
    dual_store = DualRecallStore(
        db_path,
        faiss_index_path=faiss_index_path,
        e5_model_path=e5_model_path,
        device=device,
    )
    
    dual_results = []
    dual_latencies = []
    memory_before_dual = process.memory_info().rss / 1024 / 1024  # MB
    
    for query in test_queries:
        start_time = time.perf_counter()
        candidates = dual_store.search_candidates(
            query,
            scope_level="global",
            destination="local",
            limit=100,
            strict_validity=False,
            fts_top_k=50,
            faiss_top_k=50,
        )
        latency = (time.perf_counter() - start_time) * 1000  # ms
        dual_latencies.append(latency)
        dual_results.append([c.memory_id for c in candidates])
    
    memory_after_dual = process.memory_info().rss / 1024 / 1024  # MB
    dual_store.close()
    
    # Compute metrics
    fts_avg_latency = sum(fts_latencies) / len(fts_latencies)
    dual_avg_latency = sum(dual_latencies) / len(dual_latencies)
    fts_memory_usage = memory_after_fts - memory_before
    dual_memory_usage = memory_after_dual - memory_before_dual
    
    # Recall metrics (if ground truth provided)
    recall_metrics = {}
    if ground_truth:
        fts_recalls = []
        dual_recalls = []
        query_details = []
        
        for query in test_queries:
            if query in ground_truth:
                gt_ids = ground_truth[query]
                idx = test_queries.index(query)
                
                # FTS recall
                fts_retrieved = set(fts_results[idx])
                fts_relevant = len(fts_retrieved & set(gt_ids))
                fts_recall = fts_relevant / len(gt_ids) if gt_ids else 0.0
                fts_recalls.append(fts_recall)
                
                # Dual recall
                dual_retrieved = set(dual_results[idx])
                dual_relevant = len(dual_retrieved & set(gt_ids))
                dual_recall = dual_relevant / len(gt_ids) if gt_ids else 0.0
                dual_recalls.append(dual_recall)
                
                query_details.append({
                    "query": query,
                    "ground_truth": gt_ids,
                    "fts_recall": fts_recall,
                    "dual_recall": dual_recall,
                    "fts_found": fts_relevant,
                    "dual_found": dual_relevant,
                    "improvement": dual_recall - fts_recall,
                })
        
        fts_avg = sum(fts_recalls) / len(fts_recalls) if fts_recalls else 0.0
        dual_avg = sum(dual_recalls) / len(dual_recalls) if dual_recalls else 0.0
        
        recall_metrics = {
            "fts_recall@100": fts_avg,
            "dual_recall@100": dual_avg,
            "recall_improvement_absolute": dual_avg - fts_avg,
            "recall_improvement_percent": (
                (dual_avg - fts_avg) / fts_avg * 100
                if fts_avg > 0
                else (dual_avg * 100 if dual_avg > 0 else 0.0)
            ),
            "query_details": query_details,
        }
    
    # Coverage comparison
    fts_avg_coverage = sum(len(r) for r in fts_results) / len(fts_results)
    dual_avg_coverage = sum(len(r) for r in dual_results) / len(dual_results)
    
    results = {
        "test_queries_count": len(test_queries),
        "fts_baseline": {
            "avg_latency_ms": fts_avg_latency,
            "avg_coverage": fts_avg_coverage,
            "memory_usage_mb": fts_memory_usage,
        },
        "dual_recall": {
            "avg_latency_ms": dual_avg_latency,
            "avg_coverage": dual_avg_coverage,
            "memory_usage_mb": dual_memory_usage,
            "faiss_enabled": dual_store.faiss_enabled,
        },
        "comparison": {
            "latency_increase_ms": dual_avg_latency - fts_avg_latency,
            "latency_increase_percent": (
                (dual_avg_latency - fts_avg_latency) / fts_avg_latency * 100
                if fts_avg_latency > 0 else 0.0
            ),
            "coverage_increase": dual_avg_coverage - fts_avg_coverage,
            "memory_increase_mb": dual_memory_usage - fts_memory_usage,
        },
        **recall_metrics,
    }
    
    return results


def create_synthetic_test_queries(db_path: str | Path, count: int = 10) -> list[str]:
    """从数据库中创建合成测试查询"""
    import sqlite3
    
    conn = sqlite3.connect(str(Path(db_path).expanduser()))
    rows = conn.execute("SELECT content FROM memories ORDER BY RANDOM() LIMIT ?", (count,)).fetchall()
    conn.close()
    
    # Extract keywords from memory content as queries
    queries = []
    for row in rows:
        content = row[0]
        # Simple: take first 50 chars or first sentence
        query = content[:50].strip()
        if "。" in query:
            query = query.split("。")[0] + "。"
        queries.append(query)
    
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dual recall system performance")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "memory.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--faiss-index",
        type=Path,
        default=PROJECT_ROOT / "deploy/faiss_index.bin",
        help="Path to FAISS index",
    )
    parser.add_argument(
        "--e5-model",
        type=Path,
        default=PROJECT_ROOT / "deploy/e5_model",
        help="Path to E5 model",
    )
    parser.add_argument(
        "--test-queries",
        type=Path,
        help="Path to JSON file with test queries (list of strings)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Path to JSON file with ground truth (query -> [memory_ids])",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/retrieve_dual_recall_evaluation.json",
        help="Output path for evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for encoding",
    )
    
    args = parser.parse_args()
    
    # Load or create test queries
    if args.test_queries and args.test_queries.exists():
        test_queries = json.loads(args.test_queries.read_text(encoding="utf-8"))
    else:
        print("No test queries provided, creating synthetic queries from database...")
        test_queries = create_synthetic_test_queries(args.db_path, count=10)
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth and args.ground_truth.exists():
        ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    
    # Run evaluation
    results = evaluate_recall(
        db_path=args.db_path,
        faiss_index_path=args.faiss_index if args.faiss_index.exists() else None,
        e5_model_path=args.e5_model,
        test_queries=test_queries,
        ground_truth=ground_truth,
        device=args.device,
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.output}")
    
    # Check if recall improvement meets requirement
    if "recall_improvement_percent" in results:
        improvement = results["recall_improvement_percent"]
        print(f"\n✓ Recall improvement: {improvement:.1f}%")
        if improvement >= 10.0:
            print("✓ Requirement met: Recall improvement >= 10%")
        else:
            print(f"✗ Requirement not met: Recall improvement {improvement:.1f}% < 10%")
            
        # Print query details
        if "query_details" in results:
            print(f"\nQuery-level breakdown:")
            for detail in results["query_details"]:
                print(f"  {detail['query'][:30]:30s} | FTS: {detail['fts_recall']:.2f} | Dual: {detail['dual_recall']:.2f} | Δ: {detail['improvement']:+.2f}")


if __name__ == "__main__":
    main()
