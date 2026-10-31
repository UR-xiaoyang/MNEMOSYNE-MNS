# MNEMOSYNE-MNS API 使用指南

## 快速开始

### 1. 完整示例：端到端记忆检索

```python
from memory_pipeline import create_pipeline
from memory_schema import synthesize_native_metadata_for_alpsbench

# 创建 pipeline（使用推荐的 hybrid ranker）
pipeline = create_pipeline("hybrid")

# 准备候选记忆（示例：从 AlpsBench 或真实记忆库加载）
candidate_memories = [
    synthesize_native_metadata_for_alpsbench(
        "在 git config 中设置 http.proxy 和 https.proxy",
        index=0,
        dataset="user_memories"
    ),
    synthesize_native_metadata_for_alpsbench(
        "SSH 到 jx@172.29.2.51 训练服务器",
        index=1,
        dataset="user_memories"
    ),
    synthesize_native_metadata_for_alpsbench(
        "Python 3.9+ 需要 pip install torch transformers",
        index=2,
        dataset="user_memories"
    ),
]

# 检索并排序
results = pipeline.retrieve_and_rank(
    query_text="如何配置代理服务器？",
    candidate_memories=candidate_memories,
    scope_level="global",
    destination="local",
    top_k=5
)

# 装配上下文（预算约束）
assembled = pipeline.assemble_context(
    ranked_results=results,
    max_tokens=2000,
    tokens_per_memory=200
)

# 使用结果
for memory in assembled:
    print(f"Rank {memory['rank']}: {memory['content'][:50]}...")
    print(f"  Score: {memory['score']:.4f}")
    print(f"  Authority: {memory['metadata']['authority_class']}")
```

### Persistent MemoryStore

The service exposes a local SQLite-backed store:

```bash
# Write or update a memory
curl -X POST http://localhost:8000/memories \
  -H 'Content-Type: application/json' \
  -d '{"id":"m-1","content":"配置代理服务器","scope_level":"global","destination":"local"}'

# Search the store, then rank the visible candidates
curl -X POST http://localhost:8000/retrieve_store \
  -H 'Content-Type: application/json' \
  -d '{"query":"代理","memories":[],"top_k":5,"scope_level":"global","destination":"local"}'
```

`MemoryStore` applies scope, egress, superseded/deprecated and strict-conflict filters before ranking. In Docker, mount `/app/data` so `memory.db` survives container replacement.

### ROUTE / WRITE / UPDATE

```bash
# Route only; does not retrieve or write
curl -X POST http://localhost:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"text":"继续上次关于记忆调度的方案","scope_level":"session"}'

# Generate a proposal; does not commit a memory
curl -X POST http://localhost:8000/write_proposals \
  -H 'Content-Type: application/json' \
  -d '{"text":"请记住：我偏好使用中文回复。","scope_level":"project"}'

# Explicitly supersede an existing memory
curl -X POST http://localhost:8000/memories/supersede \
  -H 'Content-Type: application/json' \
  -d '{"old_id":"m-old","new_id":"m-new","content":"新的代理配置","scope_level":"global","destination":"local"}'
```

`/write_proposals` is intentionally proposal-only. Long-term writes require an explicit `/memories` call; superseding requires the dedicated update endpoint.

```python
from shadow_mode import ShadowModeCollector
from citation_detection import detect_citations_in_response, detect_ignored_memories

# 初始化收集器
collector = ShadowModeCollector(db_path="~/.hermes/shadow_traces.db")

# 检索阶段
results = pipeline.retrieve_and_rank(...)
assembled = pipeline.assemble_context(results, max_tokens=2000)

# 记录检索事件
retrieval_event = collector.log_retrieval(
    session_id="session_123",
    turn_id="turn_1",
    query={"text": "如何配置代理？", "scope": {"level": "global"}},
    candidates=[
        {
            "memory_id": r["memory_id"],
            "rank": r["rank"],
            "score": r["score"],
            "injected": r["injected"],
        }
        for r in results
    ],
    metadata={"ranker_model_id": "RANK-HYBRID-01"}
)

# 模型生成响应后，检测引用
response = "可以在 git config 中设置 http.proxy..."  # 模型响应

injected = [{"memory_id": m["memory_id"], "content": m["content"], "rank": m["rank"]} for m in assembled]
citations = detect_citations_in_response(response, injected)

# 记录引用事件
for citation in citations:
    collector.log_citation(
        session_id="session_123",
        turn_id="turn_1",
        retrieval_event_id=retrieval_event.event_id,
        memory_id=citation["memory_id"],
        citation={"span": citation["span"], "confidence": citation["confidence"]},
        original_rank=next(m["rank"] for m in injected if m["memory_id"] == citation["memory_id"])
    )

# 检测未被引用的记忆
ignored = detect_ignored_memories(injected, citations)
for ign in ignored:
    collector.log_ignored(
        session_id="session_123",
        turn_id="turn_1",
        retrieval_event_id=retrieval_event.event_id,
        memory_id=ign["memory_id"],
        rank=ign["rank"],
        injected=True
    )

# 用户纠正（如果发生）
if user_corrected:
    collector.log_correction(
        session_id="session_123",
        turn_id="turn_2",
        citation_event_id=citation_event_id,
        memory_id="memory_id",
        correction_type="outdated",
        feedback={"text": "这个配置过期了", "action": "supersede"}
    )

collector.close()
```

### Hermes Agent turn adapter

The explicit Python adapter is documented in [docs/HERMES-INTEGRATION.md](HERMES-INTEGRATION.md). The standalone Hermes `MemoryProvider` plugin lives under `plugins/memory/mnemosyne/` and exposes prefetch, Shadow citation logging and proposal-only writes without automatically committing resident memory.

### Shadow Mode HTTP endpoints

Set `MNEMOSYNE_SHADOW_MODE=1` to enable local trace writes. It is disabled by default.

- `GET /shadow/metrics?session_id=...` — citation/correction counters
- `POST /shadow/citations` — record a cited memory
- `POST /shadow/corrections` — record a user correction

A `/retrieve` response includes `retrieval_event_id` when shadow mode is enabled and session/turn IDs are supplied. Citation and correction endpoints return `409` while shadow mode is disabled.



```python
from shadow_mode import ShadowModeCollector

collector = ShadowModeCollector(db_path="~/.hermes/shadow_traces.db")

# 计算某个 session 的指标
metrics = collector.compute_metrics(session_id="session_123")
print(f"Citation rate: {metrics['citation_rate']:.2%}")
print(f"Correction rate: {metrics['correction_rate']:.2%}")

# 获取完整 traces
traces = collector.get_session_traces("session_123")
print(f"Retrieval events: {len(traces['retrieval_events'])}")
print(f"Citation events: {len(traces['citation_events'])}")
print(f"Correction events: {len(traces['correction_events'])}")

collector.close()
```

## API 参考

### MemoryPipeline

**初始化：**
```python
pipeline = MemoryPipeline(
    ranker_path="runs/rank_hybrid_v1",  # 模型路径
    device="cpu",                        # "cpu" 或 "cuda"
    strict_validity=False                # 是否严格过滤 stale/conflicted
)
```

**检索与排序：**
```python
results = pipeline.retrieve_and_rank(
    query_text: str,                     # 查询文本
    candidate_memories: list[MemoryEntity],  # 候选记忆池
    scope_level: str = "global",         # "session" | "project" | "global"
    destination: str = "local",          # "local" | "cloud_anthropic" | ...
    path_hint: str = "",                 # 分类路径提示
    requires_high_authority: bool = False,  # 只返回高权威记忆
    top_k: int = 10,                     # 返回前 K 个
    extra_features: np.ndarray | None = None  # [N x 12] 结构化特征
)
# 返回: list[dict] 包含 memory_id, content, rank, score, metadata
```

**上下文装配：**
```python
assembled = pipeline.assemble_context(
    ranked_results: list[dict],          # retrieve_and_rank() 输出
    max_tokens: int = 4000,              # token 预算
    tokens_per_memory: int = 200         # 每条记忆的 token 估计
)
# 返回: list[dict] injected=True 的记忆
```

### ShadowModeCollector

**初始化：**
```python
collector = ShadowModeCollector(db_path="~/.hermes/shadow_traces.db")
```

**记录事件：**
```python
# 检索
retrieval_event = collector.log_retrieval(session_id, turn_id, query, candidates, metadata)

# 引用
citation_event = collector.log_citation(session_id, turn_id, retrieval_event_id, memory_id, citation, ...)

# 纠正
correction_event = collector.log_correction(session_id, turn_id, citation_event_id, memory_id, correction_type, feedback)

# 忽略
ignored_event = collector.log_ignored(session_id, turn_id, retrieval_event_id, memory_id, rank, injected)
```

**查询：**
```python
metrics = collector.compute_metrics(session_id=None)  # None = 全局指标
traces = collector.get_session_traces(session_id)
```

### Citation Detection

```python
from citation_detection import detect_citations_in_response, detect_ignored_memories

# 检测引用
citations = detect_citations_in_response(
    response: str,                       # 模型响应
    injected_memories: list[dict],       # [{"memory_id": ..., "content": ..., "rank": ...}]
    min_overlap_words: int = 3,          # 最少重叠词数
    min_overlap_ratio: float = 0.3       # 最小重叠比例
)
# 返回: [{"memory_id": ..., "span": ..., "confidence": ..., "explicit_reference": bool}]

# 检测被忽略的记忆
ignored = detect_ignored_memories(
    injected_memories: list[dict],
    citations: list[dict]
)
# 返回: [{"memory_id": ..., "rank": ..., "injected": True}]
```

## 模型选择

| Model | 特征 | NDCG@10 | 用途 |
|---|---|---:|---|
| **hybrid** | E5 + 结构化 | 0.5796 | **推荐用于生产** |
| extended | E5 + 结构化 + 原生 | TBD | 测试中，可能更好 |
| e5 | 纯语义 | 0.5146 | 快速原型 |
| tree | 非线性结构化 | 0.4278 | 基线对比 |

**创建 pipeline：**
```python
pipeline = create_pipeline("hybrid")    # 推荐
pipeline = create_pipeline("extended")  # 测试中
pipeline = create_pipeline("e5")        # 轻量级
pipeline = create_pipeline("tree")      # 基线
```

## 系统集成

### Hermes Agent 集成点

1. **查询路由后**：确定 scope 和 destination
2. **召回后**：获得候选记忆池
3. **注入前**：调用 `pipeline.retrieve_and_rank()` + `assemble_context()`
4. **响应生成后**：检测 citations
5. **用户纠正时**：记录 correction 事件

### 配置建议

```yaml
memory_retrieval:
  ranker_model: "hybrid"           # 或 "extended"
  top_k: 20                        # 排序前 20
  max_context_tokens: 4000         # 预算
  strict_validity: false           # 允许 stale 记忆
  
  shadow_mode:
    enabled: false                 # 用户 opt-in
    db_path: "~/.hermes/shadow_traces.db"
    collect_citations: true
    collect_corrections: true
```

## 性能指标

| 操作 | 延迟（CPU） | 说明 |
|---|---:|---|
| 特征提取（100 候选） | ~500ms | E5 编码 |
| 排序 | <10ms | 线性代数 |
| 硬过滤 | <1ms | 简单规则 |
| Citation 检测 | ~50ms | 启发式 |
| **端到端（100 候选）** | **~600ms** | 可接受 |

GPU 加速可将编码降至 ~100ms。

## 下一步

- ✅ 基础 pipeline 完成
- 🔄 扩展 ranker 训练中（预计 45 分钟）
- ⏳ Shadow mode 部署（需 Hermes agent 支持）
- ⏳ 真实 trace 评估
- ⏳ 其他专家模块（RETRIEVE, WRITE, UPDATE）

## 故障排查

**问题：找不到模型文件**
```python
# 确保模型已训练
Path("runs/rank_hybrid_v1/hybrid_ranker.joblib").exists()
Path("runs/rank_hybrid_v1/metadata.json").exists()
```

**问题：特征维度不匹配**
```python
# 检查 extra_features 维度
# 标准结构化特征：12 维
# 原生特征：14 维
# Extended ranker 需要：12 + 14 = 26 维（不包含 E5）
```

**问题：Shadow mode 数据库锁定**
```python
# 确保调用 collector.close()
collector.close()
```

## 许可与数据政策

- 代码：MIT License
- 训练数据：AlpsBench（Apache 2.0）
- Shadow traces：本地存储，用户数据，不上传
- 模型权重：计划发布到 Hugging Face
