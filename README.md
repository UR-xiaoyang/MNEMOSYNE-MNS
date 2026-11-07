# MNEMOSYNE-MNS

MNEMOSYNE-MNS（Memory Navigation Scheduler）是一个可插拔的外置记忆导航与调度系统。

## 设计目标

- 用外部记忆实体库承载长期记忆；
- 通过分类树、快速索引和慢速索引完成分层召回；
- 组合可独立训练、版本化和替换的专家模型；
- 在作用域、权限、版本和 token 预算约束下装配上下文；
- 保留完整 trace，支持回放、评估和持续训练。

## 专家模块

```text
ROUTE       查询路由
WRITE       记忆提取
UPDATE      记忆更新
RETRIEVE    多路召回
RANK        候选重排序
CONFLICT    冲突检测
ASSEMBLE    上下文装配
AUDIT       后台审计
EGRESS      本地/云端投递策略
```

## 仓库边界

本仓库只保存：

- 训练和推理代码；
- 配置与 schema；
- 测试；
- 架构和实验文档；
- 可复现的构建脚本。

以下内容**不进入 Git 仓库**：

- 私有对话数据集；
- Claude、Codex、OpenCode、Hermes 原始历史；
- 数据库、日志和本地 agent 状态；
- 训练缓存和实验输出；
- 模型权重。

模型权重计划单独发布到 Hugging Face，并在对应模型卡中记录训练数据范围、许可证和限制。

## 当前状态

**✓ V2 增强版已部署（Production-Ready）**

RANK Enhanced V2 已完成生产部署，相比基线版本NDCG@10提升39%。

### 核心功能

- ✅ **端到端检索 Pipeline** — 硬过滤 → 语义+结构化+原生特征 → Ranking → 预算装配
- ✅ **RANK Enhanced V2** — 23维特征 + 硬负样本挖掘（**NDCG@10 = 0.8057, +39%**）
- ✅ **训练好的 Ranker** — Hybrid基线（NDCG@10 = 0.5796）作为回退选项
- ✅ **Shadow Mode** — 本地 trace 收集与生产指标计算
- ✅ **Citation 检测** — 启发式引用检测
- ✅ **RETRIEVE 基础层** — SQLite 记忆存储、FTS/子串召回、scope/egress/validity 过滤
- ✅ **记忆写入 API** — `/memories` 持久化写入，`/retrieve_store` 从本地库召回并排序
- ✅ **模型 bundle** — TF-IDF vectorizer、ranker、E5 目录可离线打包
- ✅ **Context Assembly** — token budget、近重复去重、conflicted/superseded 过滤和冲突链排除
- ✅ **Session Working Context** — 每轮对话写入 session scope，并在下一轮按 session/project/global 层级召回；不会自动升级为长期记忆
- ✅ **ROUTE baseline** — 分类路径、authority 需求、低置信度慢回退信号
- ✅ **Hermes MemoryProvider** — 独立 plugin provider，支持 prefetch、Shadow citation 和 proposal-only WRITE

### 使用示例

```python
from memory_pipeline import create_pipeline

# 创建 V2 增强版 pipeline (推荐)
pipeline = create_pipeline("v2")

# 或使用基线版本
# pipeline = create_pipeline("hybrid")

# 检索并排序
results = pipeline.retrieve_and_rank(
    query_text="如何配置代理？",
    candidate_memories=memory_pool,
    scope_level="global",
    top_k=5
)

# 装配上下文
assembled = pipeline.assemble_context(results, max_tokens=2000)

# 使用结果
for memory in assembled:
    print(f"Rank {memory['rank']}: {memory['content']}")
```

**V2 部署文档**: [DEPLOYMENT_V2.md](DEPLOYMENT_V2.md)  
完整 API 文档：[docs/API-USAGE.md](docs/API-USAGE.md)  
Hermes 适配器文档：[docs/HERMES-INTEGRATION.md](docs/HERMES-INTEGRATION.md)

### 持久化记忆 API

```bash
# 启动服务 (默认使用 V2)
export MNEMOSYNE_RANKER_VERSION=v2  # 或 hybrid
python -m uvicorn docker.server:app --host 0.0.0.0 --port 8000

# 写入记忆
curl -X POST http://localhost:8000/memories \
  -H 'Content-Type: application/json' \
  -d '{"id":"m-1","content":"配置代理服务器","scope_level":"global","destination":"local"}'

# 检索排序
curl -X POST http://localhost:8000/retrieve_store \
  -H 'Content-Type: application/json' \
  -d '{"query":"代理","memories":[],"top_k":5,"scope_level":"global","destination":"local"}'
```

容器将 SQLite 数据库保存在 `/app/data/memory.db`；生产部署应挂载持久卷。

- **端到端延迟**：~600ms / 100 候选（CPU）
- **NDCG@10**：0.5796（AlpsBench 验证集）
- **CPU-only**：无需 GPU

## 快速开始

### 环境要求

```bash
python >= 3.9
torch >= 2.0
transformers >= 4.30
sklearn
numpy
joblib
```

### 评估冻结 E5 基线

```bash
python3 scripts/evaluate_rank_embedding.py \
  --feature-dir <path_to_features> \
  --output-dir runs/rank_embedding_e5_v1
```

### 训练其他基线

```bash
# TF-IDF + 结构特征基线
python3 scripts/train_rank_baseline.py

# 非线性树模型
python3 scripts/train_rank_tree.py

# 混合模型（E5 + 结构特征）
python3 scripts/train_rank_hybrid.py
```

### 统一推理接口

```python
from scripts.rank_inference import RankingBaseline

# 加载模型
baseline = RankingBaseline(model_path="runs/rank_embedding_e5_v1")

# 打分
queries = ["如何配置代理服务器？"]
memories = ["在 git config 中设置 http.proxy 和 https.proxy"]
scores = baseline.score(queries, memories)
```

## 实验报告

- [TF-IDF + 结构特征基线](docs/rank-baseline-report.md)
- [Pairwise 排序基线](docs/rank-pairwise-baseline-v1.md)
- [特征消融实验](docs/rank-pairwise-ablation-v1.md)
- [非线性树模型](docs/rank-tree-baseline-v1.md)
- [冻结 E5 语义基线](docs/rank-e5-frozen-v1.md)
- [Baseline 选择报告](docs/baseline-selection-report.md)

## 从 Hugging Face 获取模型 bundle

推荐模型 bundle 已发布到：

```text
https://huggingface.co/UR-xiaoyang/mnemosyne-mns-rank-hybrid-01
```

下载并部署：

```bash
huggingface-cli download UR-xiaoyang/mnemosyne-mns-rank-hybrid-01 \
  --local-dir deploy

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MNEMOSYNE_E5_MODEL=$PWD/deploy/e5_model \
python3 -m uvicorn docker.server:app --host 0.0.0.0 --port 8000
```

仓库只包含公开 AlpsBench 训练得到的 RANK bundle；私有对话数据、私有历史和本地 SQLite 数据库不在其中。


模型权重和 E5 模型目录不进入 Git。先准备本地部署 bundle：

```bash
python3 scripts/export_ranker_bundle.py \
  --source runs/rank_hybrid_v1 \
  --vectorizer <path_to_features>/rerank_tfidf_vectorizer.joblib \
  --destination deploy/model_bundle
# 将本地 E5 snapshot 复制到 deploy/e5_model/

docker build -t mnemosyne-mns:portable .
docker run --rm -p 8000:8000 mnemosyne-mns:portable
```

完整说明：[docs/PORTABLE-DEPLOYMENT.md](docs/PORTABLE-DEPLOYMENT.md)。Docker 镜像使用离线模型目录，不在启动时联网下载。`cpp/mnemosyne_inference.cpp` 仍是 ONNX Runtime 的参考集成，尚未作为独立 Ranker 发布；它需要单独验证 tokenizer 与 Hybrid 特征的数值一致性。

### Native C++ session gateway

默认 session-mode 的 HTTP 数据平面现有原生 C++ 实现：认证、SQLite/WAL 写入、FTS 更新和 `/context/recent` 的按时间读取不再依赖 Python、PyTorch 或 Transformers。它与现有 Python gateway 共用 `memories`/`memories_fts` schema，并保持 Hermes provider 使用的 `/health`、`/context/turn`、`/context/checkpoint` 和 `/context/recent` 合约。

Hybrid Ranker、experimental-global、Shadow 数据导出仍保留在 Python 侧，直到 ONNX tokenizer 和模型特征获得端到端一致性验证；不能为了迁移而降低检索质量。构建、切换、回滚与性能验证见 [docs/NATIVE-GATEWAY.md](docs/NATIVE-GATEWAY.md)。

## 验证

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py docker/server.py
```

当前 runtime/deployment/memory-store contract 测试覆盖结构化特征、API 记忆 ID 稳定性、模型 bundle 的 vectorizer 保留、路径解析和 SQLite 召回过滤。Docker build 需要 Docker daemon；C++ build 需要 ONNX Runtime 和 tokenizer 依赖。

## 隐私与安全

私有数据只保留在本地工作环境，不通过 Git 提交。提交前必须执行凭据扫描、暂存区审查和测试。
