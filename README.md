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

**✓ 可用版本已交付（Production-Ready API）**

RANK 专家的完整 API 已实现并可用于生产环境集成。

### 核心功能

- ✅ **端到端检索 Pipeline** — 硬过滤 → 语义+结构化+原生特征 → Ranking → 预算装配
- ✅ **训练好的 Ranker** — Hybrid 模型（NDCG@10 = 0.5796）
- ✅ **Shadow Mode** — 本地 trace 收集与生产指标计算
- ✅ **Citation 检测** — 启发式引用检测
- ✅ **记忆实体 Schema** — 完整元数据（scope / authority / validity / usage）

### 使用示例

```python
from memory_pipeline import create_pipeline

# 创建 pipeline
pipeline = create_pipeline("hybrid")

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

完整 API 文档：[docs/API-USAGE.md](docs/API-USAGE.md)

### 性能

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

## 可移植部署

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

完整说明：[docs/PORTABLE-DEPLOYMENT.md](docs/PORTABLE-DEPLOYMENT.md)。Docker 镜像使用离线模型目录，不在启动时联网下载。C++/ONNX 文件目前是参考集成，尚未在本机编译验证，不宣称为独立可执行发布物。

## 验证

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py docker/server.py
```

当前 runtime contract 测试覆盖结构化特征、API 记忆 ID 稳定性、模型 bundle 的 vectorizer 保留和路径解析。Docker build 需要 Docker daemon；C++ build 需要 ONNX Runtime 和 tokenizer 依赖。

## 隐私与安全

私有数据只保留在本地工作环境，不通过 Git 提交。提交前必须执行凭据扫描、暂存区审查和测试。
