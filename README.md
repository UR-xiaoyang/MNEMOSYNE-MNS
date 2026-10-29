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

**RANK 专家的 CPU baseline 已完成。**

已训练并评估的基线模型：

| Model | Type | NDCG@10 | Status |
|---|---|---:|---|
| TF-IDF cosine | Lexical | 0.3916 | Baseline |
| Pairwise logistic | Linear structured | 0.4090 | Baseline |
| Pairwise tree | Nonlinear structured | 0.4278 | Baseline |
| Frozen E5 | Semantic | 0.5146 | Baseline |
| **Hybrid E5+features** | **Semantic + structured** | **0.5796** | **✓ Recommended** |

推荐基线：**MNEMOSYNE-MNS-RANK-HYBRID-01**

- Components: Frozen E5 (`intfloat/multilingual-e5-small`) + 12 structured features
- Training: AlpsBench public train split, pairwise logistic ranker
- Validation: 312 queries, NDCG@10 = 0.5796, Recall@1 = 0.4071
- Inference: CPU, no GPU required
- Training time: ~41 minutes
- Multilingual: Chinese + English

详细报告：[docs/baseline-selection-report.md](docs/baseline-selection-report.md)

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

## 数据政策

当前所有基线训练只使用 AlpsBench 公开数据集的训练和验证切分。私有弱监督标签已排除在训练和模型选择之外，仅作域迁移诊断。

## 下一步工作

1. 完成混合 ranker 训练与评估
2. 集成 scope/path/validity 特征
3. 在真实 `retrieved → cited → corrected` trace 上评估
4. Shadow mode 部署
5. 其他专家模块（RETRIEVE、WRITE、UPDATE）

## 隐私与安全

私有数据只保留在本地工作环境，不通过 Git 提交。提交前必须执行凭据扫描、暂存区审查和测试。
