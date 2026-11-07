"""
创建语义召回测试集：查询与记忆内容无词汇重叠，需要语义理解

策略：
1. 查询使用同义词、上位词、不同表述
2. 记忆内容使用技术术语、具体实现
3. 测试纯语义匹配能力
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


def create_semantic_test_dataset(db_path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    """
    创建语义测试数据集：强制需要语义理解才能召回
    
    Returns:
        (test_queries, ground_truth): 测试查询列表和 ground truth 映射
    """
    store = MemoryStore(db_path)
    
    # Create memories with technical/specific language
    test_memories = [
        # 网络配置（用技术术语）
        {
            "memory_id": "sem:001",
            "content": "配置 HTTP_PROXY 和 HTTPS_PROXY 环境变量可以让程序通过中间服务器转发请求。",
            "semantic_topics": ["网络", "代理", "配置"],
        },
        {
            "memory_id": "sem:002",
            "content": "设置 git config http.proxy 和 https.proxy 参数来让版本控制系统使用中间服务器。",
            "semantic_topics": ["Git", "代理", "配置"],
        },
        # 性能优化（用具体实现）
        {
            "memory_id": "sem:003",
            "content": "建立连接池避免频繁创建 TCP 连接，可以显著降低数据库访问延迟。",
            "semantic_topics": ["数据库", "性能", "连接池"],
        },
        {
            "memory_id": "sem:004",
            "content": "使用 B-tree 或 Hash index 可以将 O(n) 的全表扫描降低到 O(log n) 或 O(1)。",
            "semantic_topics": ["数据库", "索引", "查询优化"],
        },
        {
            "memory_id": "sem:005",
            "content": "实现缓存层（Redis/Memcached）可以避免重复计算和数据库查询。",
            "semantic_topics": ["缓存", "性能", "Redis"],
        },
        # 调试技巧（用工具和方法）
        {
            "memory_id": "sem:006",
            "content": "使用 pdb.set_trace() 或 breakpoint() 可以在代码执行时暂停并检查变量状态。",
            "semantic_topics": ["调试", "Python", "断点"],
        },
        {
            "memory_id": "sem:007",
            "content": "通过 logging 模块记录关键路径的变量值，帮助定位问题发生的位置。",
            "semantic_topics": ["调试", "日志", "排错"],
        },
        {
            "memory_id": "sem:008",
            "content": "memory_profiler 可以追踪每行代码的内存分配，找出未释放的对象。",
            "semantic_topics": ["调试", "内存泄漏", "profiling"],
        },
        # 架构设计（用理论和模式）
        {
            "memory_id": "sem:009",
            "content": "CAP 定理表明分布式系统无法同时保证 Consistency、Availability 和 Partition tolerance。",
            "semantic_topics": ["分布式", "CAP", "理论"],
        },
        {
            "memory_id": "sem:010",
            "content": "采用微服务架构可以让不同模块独立部署和扩展，但会增加网络通信开销。",
            "semantic_topics": ["架构", "微服务", "分布式"],
        },
        {
            "memory_id": "sem:011",
            "content": "使用消息队列（Kafka/RabbitMQ）实现异步通信，解耦服务之间的直接依赖。",
            "semantic_topics": ["架构", "消息队列", "解耦"],
        },
        # 容器部署（用操作和工具）
        {
            "memory_id": "sem:012",
            "content": "多阶段构建（multi-stage build）可以分离编译环境和运行环境，减小最终镜像大小。",
            "semantic_topics": ["Docker", "构建", "优化"],
        },
        {
            "memory_id": "sem:013",
            "content": "使用 Docker Compose 定义多容器应用的服务、网络和数据卷配置。",
            "semantic_topics": ["Docker", "编排", "配置"],
        },
        {
            "memory_id": "sem:014",
            "content": "Kubernetes 通过 Deployment、Service 和 Ingress 管理容器的生命周期和流量路由。",
            "semantic_topics": ["Kubernetes", "容器编排", "部署"],
        },
        # 机器学习评估（用公式和概念）
        {
            "memory_id": "sem:015",
            "content": "Recall = TP / (TP + FN)，衡量模型找回所有正样本的能力。",
            "semantic_topics": ["机器学习", "评估指标", "召回率"],
        },
        {
            "memory_id": "sem:016",
            "content": "Precision = TP / (TP + FP)，衡量模型预测为正的样本中真正是正的比例。",
            "semantic_topics": ["机器学习", "评估指标", "精确率"],
        },
        {
            "memory_id": "sem:017",
            "content": "F1-score 是 Precision 和 Recall 的调和平均数，平衡两个指标。",
            "semantic_topics": ["机器学习", "评估指标", "F1"],
        },
        # API 设计（用原则和规范）
        {
            "memory_id": "sem:018",
            "content": "GET 用于查询、POST 创建、PUT 更新、DELETE 删除，遵循 HTTP 语义。",
            "semantic_topics": ["REST", "API", "HTTP方法"],
        },
        {
            "memory_id": "sem:019",
            "content": "使用 200/201 表示成功，400 表示客户端错误，500 表示服务器错误。",
            "semantic_topics": ["HTTP", "状态码", "错误处理"],
        },
        {
            "memory_id": "sem:020",
            "content": "实现幂等性确保重复请求不会产生副作用，使用 ETag 支持条件请求。",
            "semantic_topics": ["API设计", "幂等性", "并发控制"],
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
            retrieval=Retrieval(path="semantic/test", tags=mem.get("semantic_topics", [])),
            usage=Usage(retrieved_count=0, cited_count=0),
            egress=Egress(allowed_destinations=["local"]),
            provenance=Provenance(created_by="import", import_source="semantic_test"),
        )
        store.upsert(memory)
    
    store.close()
    
    # Define queries with NO lexical overlap
    # 查询用日常语言，记忆用技术术语
    test_queries = [
        # 网络相关（日常表述 vs 技术术语）
        "怎么让程序走跳板机访问外网",  # -> sem:001 (HTTP_PROXY)
        "如何让 git 通过中转服务器连接",  # -> sem:002 (git proxy)
        
        # 性能相关（问题描述 vs 解决方案）
        "数据库查询太慢怎么办",  # -> sem:003 (连接池), sem:004 (索引), sem:005 (缓存)
        "系统响应时间长如何改善",  # -> sem:005 (缓存)
        
        # 调试相关（需求描述 vs 工具方法）
        "怎样查看程序运行时的变量值",  # -> sem:006 (pdb), sem:007 (logging)
        "如何找到内存没释放的地方",  # -> sem:008 (memory_profiler)
        
        # 架构相关（问题 vs 理论/模式）
        "大型系统怎么设计才能容错",  # -> sem:009 (CAP), sem:010 (微服务)
        "多个服务之间如何解除耦合",  # -> sem:011 (消息队列)
        
        # 容器相关（目标 vs 实现）
        "如何减小容器镜像的体积",  # -> sem:012 (multi-stage build)
        "怎么管理多个容器服务",  # -> sem:013 (Docker Compose), sem:014 (Kubernetes)
        
        # 评估指标（需求 vs 公式/概念）
        "如何衡量模型找全样本的能力",  # -> sem:015 (Recall)
        "怎么评估预测结果的准确程度",  # -> sem:016 (Precision), sem:017 (F1)
        
        # API 设计（需求 vs 规范）
        "接口应该用什么方法表示操作",  # -> sem:018 (HTTP methods)
        "请求失败了该返回什么代码",  # -> sem:019 (status codes)
    ]
    
    ground_truth = {
        "怎么让程序走跳板机访问外网": ["sem:001"],
        "如何让 git 通过中转服务器连接": ["sem:002"],
        "数据库查询太慢怎么办": ["sem:003", "sem:004", "sem:005"],
        "系统响应时间长如何改善": ["sem:005"],
        "怎样查看程序运行时的变量值": ["sem:006", "sem:007"],
        "如何找到内存没释放的地方": ["sem:008"],
        "大型系统怎么设计才能容错": ["sem:009", "sem:010"],
        "多个服务之间如何解除耦合": ["sem:011"],
        "如何减小容器镜像的体积": ["sem:012"],
        "怎么管理多个容器服务": ["sem:013", "sem:014"],
        "如何衡量模型找全样本的能力": ["sem:015"],
        "怎么评估预测结果的准确程度": ["sem:016", "sem:017"],
        "接口应该用什么方法表示操作": ["sem:018"],
        "请求失败了该返回什么代码": ["sem:019"],
    }
    
    return test_queries, ground_truth


def main() -> None:
    db_path = PROJECT_ROOT / "memory.db"
    
    print("Creating semantic test dataset...")
    test_queries, ground_truth = create_semantic_test_dataset(db_path)
    
    # Save test queries and ground truth
    queries_path = PROJECT_ROOT / "reports/semantic_test_queries.json"
    ground_truth_path = PROJECT_ROOT / "reports/semantic_ground_truth.json"
    
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text(json.dumps(test_queries, indent=2, ensure_ascii=False), encoding="utf-8")
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"✓ Created {len(test_queries)} semantic test queries")
    print(f"✓ Saved to {queries_path}")
    print(f"✓ Ground truth saved to {ground_truth_path}")
    
    # Count memories
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM memories WHERE memory_id LIKE 'sem:%'").fetchone()[0]
    conn.close()
    
    print(f"✓ Total semantic test memories: {count}")


if __name__ == "__main__":
    main()
