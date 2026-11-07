# P3 完成报告：弱监督迭代优化闭环

## 执行摘要

P3 Phase 已完成：建立"无人工金标"的弱监督迭代优化闭环，实现完整的 iteration manifest 构建、offline baseline vs candidate 对比、disagreement 采样和自动化迭代门。所有实现均使用 toy fixtures 测试通过，未运行真实数据或完整训练。

## 核心交付物

### 1. Iteration Manifest Builder (`scripts/iterative_rank_loop.py`, 282 行)

**功能**：
- ✅ 显式来源授权 (`--input-dir` 必需，拒绝 None)
- ✅ 时序切分 (按 `query_timestamp` 排序后切分 train/val/holdout)
- ✅ Unknown 标签处理 (`sample_weight=0.0` 排除监督 loss)
- ✅ Citation/Correction 分离统计
- ✅ 隐私保护 (manifest 仅文件名，无绝对路径/文本内容)
- ✅ 数据门 (`insufficient_evidence` / `pass_with_warning` / `pass`)

**CLI**:
```bash
python scripts/iterative_rank_loop.py \
  --input-dir <EXPLICIT_DIR> \
  --output-dir ./training/iter_001 \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --min-samples 100
```

### 2. Offline Replay Evaluator (`scripts/replay_rank_iterations.py`, 209 行)

**功能**：
- ✅ 候选池严格一致性验证 (query sets + candidate sets)
- ✅ Query-level 指标 (Recall@k, MRR, NDCG@k)
- ✅ Paired bootstrap 比较 (复用 `evaluate_ranking.py`)
- ✅ 95% 置信区间输出
- ✅ 隐私保护 (结果无文件路径)

**CLI**:
```bash
python scripts/replay_rank_iterations.py \
  --baseline ./predictions/baseline.jsonl \
  --candidate ./predictions/candidate.jsonl \
  --output ./comparison.json \
  --n-bootstrap 3000
```

### 3. Active Disagreement Sampler (`scripts/active_disagreement_sampler.py`, 209 行)

**功能**：
- ✅ Baseline vs Candidate 排序分歧检测
- ✅ 分歧类型分类 (`low_confidence`, `large_rank_flip`, `score_disagreement`)
- ✅ 仅输出 IDs 和 reason (无 query/memory 文本)
- ✅ 按分歧强度排序 (rank_diff 主序, score_diff 次序)
- ✅ 隐私保护 (无路径、无文本内容)

**CLI**:
```bash
python scripts/active_disagreement_sampler.py \
  --baseline ./predictions/baseline.jsonl \
  --candidate ./predictions/candidate.jsonl \
  --output ./disagreements.jsonl \
  --n-samples 100
```

## 测试覆盖

### P3 新增测试 (19 个)

**`tests/test_iterative_rank_loop.py`** (8 tests):
- ✅ 显式来源授权要求
- ✅ 时序切分创建 (train/val/holdout 文件存在)
- ✅ Unknown 标签 `sample_weight=0.0`
- ✅ Citation/Correction 分离统计
- ✅ 数据门 insufficient_samples
- ✅ 数据门 unknown_ratio 警告
- ✅ Manifest 无真实路径
- ✅ Manifest 无 query/memory 文本

**`tests/test_replay_rank_iterations.py`** (6 tests):
- ✅ 候选池一致性验证 (query sets 必须相同)
- ✅ Query-level 指标计算
- ✅ Paired bootstrap 使用
- ✅ `(query_id, pair_id)` 唯一性验证
- ✅ `pair_id` 前缀匹配 `query_id`
- ✅ 输出无路径

**`tests/test_active_disagreement_sampler.py`** (5 tests):
- ✅ Baseline/Candidate 分歧采样
- ✅ 输出仅含 IDs 和 reason (无文本)
- ✅ 输出无路径
- ✅ Low confidence 分类
- ✅ Large rank flip 分类

### 全量测试结果

```bash
python -m pytest tests/ -q
# 263 passed, 2 warnings in 41.98s

python -m unittest discover -s tests -q
# Ran 243 tests in 14.803s
# OK

python -m py_compile scripts/*.py docker/server.py
# (all passed)
```

## 文档更新

### 新增文档

**`docs/iterative-training-protocol-v1.md`**:
- 无人工金标版本完整说明
- 证据等级分层 (strong/weak/insufficient)
- 三个 Pipeline 组件使用说明
- 迭代门详细规则 (data/offline/shadow)
- 失败模式和防护机制
- 示例迭代工作流
- 报告要求

### 已有文档

- ✅ `docs/shadow-mode-trace-schema.md` (P2 已更新)
- ✅ `P2_COMPLETION_REPORT.md` (P2 完成报告)

## 迭代门设计

### Data Gate (训练前)

**检查项**：
- 最小样本数 (train + val >= 100)
- Unknown 比例 (> 90% 警告)
- 时序先后正确
- Session isolation
- Secret scan

**结果**：
- `insufficient_evidence`: 停止，不训练
- `pass_with_warning`: 继续但记录警告
- `pass`: 进入 offline 评估

### Offline Gate (训练后)

**检查项**：
- 主指标 (NDCG@10) 95% CI 不退化
- Correction-risk 不升
- 硬约束违规 = 0
- 所有 query group 可报告

**结果**：
- `pass`: 进入 shadow mode
- `fail`: 停止迭代，保留 baseline
- `insufficient_evidence`: 不提升，收集更多数据

### Shadow Gate (线上监控)

**检查项** (滑动窗口 7-14 天)：
- Citation proxy 不恶化
- Context pollution 不升
- Correction events 不升
- 延迟在预算内
- 样本量 >= 统计最小值

**结果**：
- `pass`: 提升为 production baseline
- `fail`: 回滚，记录失败原因
- `insufficient_evidence`: 延长 shadow 期

## 硬约束验证

✅ **未访问真实数据**：所有测试使用 `tempfile.TemporaryDirectory()` toy fixtures  
✅ **仅 toy fixtures**：无真实 shadow DB、私有历史或完整训练  
✅ **无路径泄露**：manifest/report/CLI stdout 测试断言无 `C:\`, `/home/`, `/Users/`  
✅ **Unknown 语义**：`label="unknown"` 显式保留，`sample_weight=0.0` 排除监督  
✅ **Citation/Correction 分离**：分别统计，不混成单一 positive 标签  
✅ **不训练删除模型**：仅排序/召回/路由，不自动删除记忆  
✅ **不绕过硬过滤**：scope/authority/validity/冲突/预算始终规则硬约束  
✅ **TDD 流程**：19 个测试全部先 FAIL 后 GREEN  
✅ **不提交/push**：仅本地工件  

## 技术设计亮点

### 1. Unknown 标签显式处理

```python
def process_sample(sample: dict) -> dict:
    """Add sample_weight based on label."""
    sample = sample.copy()
    if sample.get("label") == "unknown":
        sample["sample_weight"] = 0.0  # Excluded from supervised loss
    elif "sample_weight" not in sample:
        sample["sample_weight"] = 1.0
    return sample
```

**Why**: 未引用不等于不相关；保留 unknown 用于未来分析，但不作为 negative 训练。

### 2. 严格候选池验证

```python
# Verify identical query pools
if baseline_queries != candidate_queries:
    raise ValueError(f"Query pools don't match. ...")

# Verify identical candidate sets per query
for query_id in baseline_queries:
    baseline_pairs = baseline_by_query[query_id]
    candidate_pairs = candidate_by_query[query_id]
    if baseline_pairs != candidate_pairs:
        raise ValueError(f"Query {query_id} has different candidate sets")
```

**Why**: Paired bootstrap 要求完全相同的候选池；不同池的指标不可比。

### 3. 时序切分防泄漏

```python
# Sort by query timestamp (temporal ordering)
samples.sort(key=lambda x: x["query_timestamp"])

# Calculate split indices
n = len(samples)
train_end = int(n * train_ratio)
val_end = train_end + int(n * val_ratio)

train_samples = samples[:train_end]
val_samples = samples[train_end:val_end]
holdout_samples = samples[val_end:]
```

**Why**: 随机切分会让同一 session 的前后轮出现在 train 和 test 中，导致记忆泄漏。

### 4. 隐私分层过滤

```python
# Remove text fields if they exist (privacy)
sample.pop("query_text", None)
sample.pop("memory_text", None)
sample.pop("query_text_redacted", None)
sample.pop("memory_text_redacted", None)
```

**Why**: Disagreement sampler 仅需 IDs 定位样本；文本内容不应出现在输出。

## 与 P2 的集成

P3 直接消费 P2 导出的 shadow samples：

1. P2 `build_shadow_training_data.py` 生成 `samples.jsonl` (含 citation/correction/unknown 标签)
2. P3 `iterative_rank_loop.py` 读取 `samples.jsonl`，按时间切分，赋权重
3. P3 `replay_rank_iterations.py` 对比 baseline/candidate 在相同候选池上的排序
4. P3 `active_disagreement_sampler.py` 找出分歧样本供下一轮高价值优化

## 下一步条件

可以使用公开 benchmark/显式授权 shadow 数据的条件：

1. ✅ **Toy-only 测试全部通过** — 19 个 P3 测试 + 263 个全量测试
2. ✅ **三个脚本实现完整** — iteration loop + replay + disagreement sampler
3. ✅ **迭代门逻辑实现** — data/offline/shadow gates with insufficient_evidence
4. ✅ **隐私边界验证** — 无路径、无文本、无真实数据访问
5. ✅ **文档完整** — iterative-training-protocol-v1.md

### 实际数据使用前的额外要求

- [ ] 显式授权真实 shadow DB 路径 (环境变量或 CLI 参数)
- [ ] 真实 AlpsBench 下载和验证 (公开 benchmark)
- [ ] 真实 E5 embeddings 目录存在 (frozen baseline)
- [ ] 完整 offline evaluation 运行 (baseline vs toy candidate)
- [ ] Shadow mode 部署测试 (日志收集但不影响真实上下文)

**不要因为没有人工标签就宣称训练完成或模型已提升** — P3 仅建立了迭代闭环基础，真实提升需要实际运行迭代并通过所有门。

## 常见问题

**Q: 没有人工标注数据，模型怎么训练？**

A: 使用弱监督信号 (citation, correction, session continuation)。它们不是 ground truth，但可以作为有噪声的训练信号。Unknown 标签用 `sample_weight=0.0` 排除监督 loss，保留用于未来分析。

**Q: Correction 标签怎么处理？**

A: Correction 是独立的"错误使用/风险"信号，不与 citation positive 混成单一标签。可以用多任务学习 (citation head + correction risk head) 或分别训练。

**Q: 如何防止自我强化？**

A: 时序 holdout (test 严格晚于 train)、query-group paired bootstrap、baseline disagreement 监控、硬约束始终规则优先。

**Q: 数据不足怎么办？**

A: 返回 `insufficient_evidence`，不自动降低阈值、不把 unknown 转 negative、不伪造指标。诚实报告样本不足。

**Q: 什么时候可以上线新模型？**

A: 必须通过 data gate + offline gate + shadow gate。任何一个失败或 insufficient_evidence 都不能提升。

## 总结

P3 完成了"无人工金标的弱监督迭代优化闭环"全部交付物：

- 3 个核心脚本 (iteration loop, replay, disagreement sampler)
- 19 个新增测试 (全部通过)
- 完整文档 (iterative-training-protocol-v1.md)
- 自动化迭代门 (data/offline/shadow with insufficient_evidence)
- 隐私保护验证 (无路径、无文本、无真实数据)

下一阶段可以使用公开 benchmark 和显式授权 shadow 数据进行真实迭代实验，但必须遵守迭代门规则，不得因缺乏人工标签而跳过验证或自动上线。
