"""
创建更完整的测试数据集以验证双路召回效果

使用策略：
1. 词汇匹配查询（纯 FTS 可以召回）
2. 语义匹配查询（需要 FAISS 语义理解）
3. 混合查询（同时需要词汇和语义）
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_schema import (
    MemoryEntity, Scope, Authority, Validity, Retrieval, Usage, Egress, Provenance
)
from memory_store import MemoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_test_dataset(db_path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    """
    创建测试数据集：插入多样化的记忆和查询
    
    Returns:
        (test_queries, ground_truth): 测试查询列表和 ground truth 映射
    """
    store = MemoryStore(db_path)
    
    # Create diverse test memories
    test_memories = [
        # 技术类 - 词汇匹配
        {
            "memory_id": "test:001",
            "content": "如何在 Python 中使用 FAISS 库进行向量检索？首先安装 faiss-cpu，然后创建索引。",
            "tags": ["python", "faiss", "vector-search"],
        },
        {
            "memory_id": "test:002",
            "content": "机器学习中的召回率（Recall）指的是在所有相关文档中，实际检索到的相关文档的比例。",
            "tags": ["machine-learning", "metrics", "recall"],
        },
        {
            "memory_id": "test:003",
            "content": "Transformer 模型使用自注意力机制来处理序列数据，广泛应用于 NLP 任务。",
            "tags": ["transformer", "nlp", "attention"],
        },
        # 配置类 - 语义匹配
        {
            "memory_id": "test:004",
            "content": "要设置代理服务器，需要配置 HTTP_PROXY 和 HTTPS_PROXY 环境变量。",
            "tags": ["proxy", "configuration", "network"],
        },
        {
            "memory_id": "test:005",
            "content": "数据库连接池可以提高性能，避免频繁建立和关闭连接的开销。",
            "tags": ["database", "performance", "connection-pool"],
        },
        # 概念类 - 语义匹配
        {
            "memory_id": "test:006",
            "content": "分布式系统中的 CAP 定理指出，一致性、可用性和分区容错性不能同时满足。",
            "tags": ["distributed-systems", "cap-theorem", "theory"],
        },
        {
            "memory_id": "test:007",
            "content": "索引是数据库中用于加速查询的数据结构，类似于书籍的目录。",
            "tags": ["database", "index", "query-optimization"],
        },
        # 实践类 - 混合匹配
        {
            "memory_id": "test:008",
            "content": "使用 Docker 部署应用时，建议使用多阶段构建来减小镜像体积。",
            "tags": ["docker", "deployment", "best-practices"],
        },
        {
            "memory_id": "test:009",
            "content": "Git 分支管理策略：主分支保持稳定，开发在 feature 分支进行，通过 PR 合并。",
            "tags": ["git", "workflow", "version-control"],
        },
        {
            "memory_id": "test:010",
            "content": "REST API 设计原则：使用 HTTP 方法表示操作，URL 表示资源，状态码表示结果。",
            "tags": ["rest-api", "design", "web-development"],
        },
        # 问题解决类 - 语义匹配
        {
            "memory_id": "test:011",
            "content": "内存泄漏通常由未释放的对象引用导致，使用 profiler 工具可以定位问题。",
            "tags": ["debugging", "memory-leak", "profiling"],
        },
        {
            "memory_id": "test:012",
            "content": "当 API 响应缓慢时，应该检查数据库查询、网络延迟和缓存策略。",
            "tags": ["performance", "api", "troubleshooting"],
        },
        # 数据结构类 - 词汇匹配
        {
            "memory_id": "test:013",
            "content": "哈希表提供 O(1) 的平均查找时间，但需要处理哈希冲突。",
            "tags": ["data-structures", "hash-table", "algorithms"],
        },
        {
            "memory_id": "test:014",
            "content": "B树是一种自平衡的树结构，常用于数据库和文件系统的索引。",
            "tags": ["data-structures", "btree", "index"],
        },
        # 工具类 - 混合匹配
        {
            "memory_id": "test:015",
            "content": "Pytest 是 Python 的测试框架，支持 fixtures、参数化测试和插件扩展。",
            "tags": ["testing", "pytest", "python"],
        },
    ]
    
    timestamp = time.time()
    for idx, mem in enumerate(test_memories):
        memory = MemoryEntity(
            memory_id=mem["memory_id"],
            content=mem["content"],
            created_at=timestamp + idx,
            updated_at=timestamp + idx,
            version=1,
            scope=Scope(level="global"),
            authority=Authority(class_="system_rule", priority=80, source="tool_verified", reliability=1.0),
            validity=Validity(state="active"),
            retrieval=Retrieval(path="test/data", tags=mem.get("tags", [])),
            usage=Usage(retrieved_count=0, cited_count=0),
            egress=Egress(allowed_destinations=["local"]),
            provenance=Provenance(created_by="import", import_source="test_dataset"),
        )
        store.upsert(memory)
    
    store.close()
    
    # Define test queries with ground truth
    test_queries = [
        # 词汇精确匹配（FTS 擅长）
        "FAISS 库",
        "召回率 Recall",
        "哈希表",
        # 语义匹配（需要 FAISS）
        "如何配置网络代理",  # 应该召回 test:004（代理服务器）
        "怎样提升数据库性能",  # 应该召回 test:005（连接池）、test:007（索引）
        "分布式架构设计原则",  # 应该召回 test:006（CAP）
        "程序运行慢怎么办",  # 应该召回 test:012（API 响应缓慢）
        "代码调试技巧",  # 应该召回 test:011（内存泄漏）
        # 混合匹配
        "Docker 容器部署",  # 词汇 + 语义
        "Git 版本管理",  # 词汇 + 语义
    ]
    
    ground_truth = {
        "FAISS 库": ["test:001"],
        "召回率 Recall": ["test:002"],
        "哈希表": ["test:013"],
        "如何配置网络代理": ["test:004"],
        "怎样提升数据库性能": ["test:005", "test:007"],
        "分布式架构设计原则": ["test:006"],
        "程序运行慢怎么办": ["test:012"],
        "代码调试技巧": ["test:011"],
        "Docker 容器部署": ["test:008"],
        "Git 版本管理": ["test:009"],
    }
    
    return test_queries, ground_truth


def main() -> None:
    db_path = PROJECT_ROOT / "memory.db"
    
    print("Creating test dataset...")
    test_queries, ground_truth = create_test_dataset(db_path)
    
    # Save test queries and ground truth
    queries_path = PROJECT_ROOT / "reports/test_queries.json"
    ground_truth_path = PROJECT_ROOT / "reports/ground_truth.json"
    
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text(json.dumps(test_queries, indent=2, ensure_ascii=False), encoding="utf-8")
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"✓ Created {len(test_queries)} test queries")
    print(f"✓ Saved to {queries_path}")
    print(f"✓ Ground truth saved to {ground_truth_path}")
    
    # Count memories
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    
    print(f"✓ Total memories in database: {count}")


if __name__ == "__main__":
    main()
