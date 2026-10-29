# MNEMOSYNE-MNS 下一阶段进度报告

**日期：** 2026-10-29  
**阶段：** 系统原生特征集成与 Shadow Mode 准备  
**状态：** 基础设施完成，准备进入实际部署

---

## 已完成工作

### 1. 记忆实体 Schema 设计 ✓

**文件：** `docs/memory-entity-schema-v1.md`

定义了完整的记忆实体元数据结构，包括：

- **Scope（作用域）**：session / project / global，支持可见性控制
- **Authority（权威性）**：user_current / user_persistent / system_rule / inferred / permanent_constraint
- **Validity（有效性）**：active / stale / conflicted / superseded / deprecated
- **Retrieval（检索）**：分类路径、标签、embedding 模型版本
- **Usage（使用统计）**：retrieved / cited / corrected 计数，citation_rate / correction_rate
- **Egress（出口控制）**：allowed_destinations、sensitivity（public / private / credential / pii）
- **Provenance（来源）**：创建者、导入来源、父记忆链

**原生特征提取：** 14 个系统原生特征，包括 scope 匹配、authority 优先级、validity 状态、usage 信号、path 一致性等。

**硬过滤规则：** 定义了 scope / egress / validity / authority 的硬性过滤条件，在 ranking 之前执行。

### 2. Python 实现 ✓

**文件：** `scripts/memory_schema.py`

- 完整的 `MemoryEntity` 数据类（Python dataclasses）
- `extract_native_features()` 函数：从 query 和 memory 提取 14 维原生特征
- `synthesize_native_metadata_for_alpsbench()` 函数：为 AlpsBench 数据合成合理的元数据（用于扩展 baseline 训练）
- `hard_filter()` 方法：检查记忆是否通过硬过滤

### 3. Shadow Mode Trace Schema 设计 ✓

**文件：** `docs/shadow-mode-trace-schema.md`

定义了 4 种事件类型的完整 schema：

1. **Retrieval Event**：query、candidates、ranking scores、features、injected memories
2. **Citation Event**：模型响应中引用了哪些记忆
3. **Correction Event**：用户纠正了哪些被引用的记忆
4. **Ignored Event**：哪些高排名记忆被注入但未被引用

**生产指标：**
- Citation rate（被引用率）
- Correction rate（被纠正率）
- Top-1 / Top-5 citation rate
- MRR citation（引用位置的倒数排名均值）
- Time to correction（引用到纠正的时间）

**存储：** SQLite 本地数据库 schema，4 张表 + 索引

**隐私保护：** 所有数据本地存储，不上传云端；credentials/PII 自动脱敏

### 4. Shadow Mode Collector 实现 ✓

**文件：** `scripts/shadow_mode.py`

- `ShadowModeCollector` 类：管理本地 SQLite trace 数据库
- `log_retrieval()` / `log_citation()` / `log_correction()` / `log_ignored()`：4 种事件记录方法
- `get_session_traces()`：检索某个 session 的所有 trace
- `compute_metrics()`：从 trace 计算生产指标
- 自动创建表结构和索引
- UUID 事件 ID 和 timestamp 跟踪

### 5. Git 提交 ✓

**Commit：** `d2e8e8d6759123635beff298bc3fd28a68b026e5`  
**文件：**
- `docs/memory-entity-schema-v1.md`
- `docs/shadow-mode-trace-schema.md`
- `scripts/memory_schema.py`
- `scripts/shadow_mode.py`

所有代码已验证并推送到远程仓库。

---

## 下一步工作

### 短期（接下来 1-2 天）

#### 1. 扩展 Hybrid Ranker 以支持原生特征

**任务：**
- 为 AlpsBench 数据合成原生元数据（使用 `synthesize_native_metadata_for_alpsbench()`）
- 提取 14 个原生特征
- 训练扩展版 Hybrid Ranker：13 个现有特征 + 14 个原生特征 = 27 特征
- 在公开验证集上评估是否有提升

**预期结果：**
- `MNEMOSYNE-MNS-RANK-HYBRID-02`（带原生特征）
- 如果 NDCG@10 > 0.5796，则更新推荐 baseline
- 如果无明显提升，说明合成元数据不够真实，需等待真实 trace

#### 2. 实现 Citation 检测启发式

**任务：**
- 在模型响应中检测是否引用了某条记忆
- 启发式规则：
  - 响应中出现记忆内容的关键短语
  - "根据..."、"之前提到..."等显式引用
  - 语义相似度超过阈值
- 实现 `detect_citations_in_response(response, injected_memories)` 函数

#### 3. Hermes Agent 集成点设计

**任务：**
- 定义 Hermes agent 中哪些地方需要调用 RANK 模型
- 设计记忆召回 → ranking → 硬过滤 → 装配 → 注入的完整管道
- 确定 shadow mode 开关和配置（默认关闭，用户 opt-in）
- 设计如何从 agent runtime 获取真实的 scope / authority / validity 元数据

### 中期（接下来 1-2 周）

#### 4. Shadow Mode 部署

**任务：**
- 在真实 Hermes agent 会话中启用 shadow mode
- 收集 100+ 会话的真实 trace
- 分析 citation rate、correction rate、latency
- 与 AlpsBench 验证集指标对比

**阻塞点：**
- 需要 Hermes agent 有记忆召回功能（当前可能尚未完整实现）
- 需要用户授权启用 shadow mode

#### 5. 真实 Trace 评估与模型迭代

**任务：**
- 用收集的真实 trace 重新评估所有 baseline
- 识别失败案例：高排名但未被引用、被引用但被纠正
- 分析原因：scope 错误、过时、冗余、误解
- 用真实 trace 重新训练 RANK 模型

#### 6. 其他专家模块

**优先级：**
1. **RETRIEVE**（多路召回）：分类树路由 + 快速索引 + 慢速索引
2. **WRITE**（记忆提取）：从对话中提取值得保存的记忆
3. **UPDATE**（记忆更新）：检测冲突、supersede、merge
4. **ASSEMBLE**（上下文装配）：预算分配、去重、冲突解决

---

## 当前架构状态

### 已实现模块

```text
[数据准备] ✓ 完成
  ↓
[特征工程] ✓ 完成（词法 + 结构 + 语义）
  ↓
[RANK Baseline] ✓ 完成（5 个模型，推荐 RANK-HYBRID-01）
  ↓
[统一推理接口] ✓ 完成
  ↓
[原生特征 Schema] ✓ 完成
  ↓
[Shadow Mode 基础设施] ✓ 完成
```

### 待实现模块

```text
[扩展 Ranker（原生特征）] → 进行中
  ↓
[Citation 检测] → 待实现
  ↓
[Hermes Agent 集成] → 待设计
  ↓
[Shadow Mode 部署] → 待用户授权
  ↓
[真实 Trace 评估] → 取决于 shadow mode
  ↓
[RETRIEVE / WRITE / UPDATE / ASSEMBLE] → 未开始
```

---

## 技术债务与风险

### 当前技术债务

1. **合成元数据的真实性**：为 AlpsBench 合成的 scope / authority / validity 是启发式的，可能与真实分布不符
2. **Citation 检测准确性**：当前尚未实现，需要启发式规则 + 可能的语义相似度，准确率未知
3. **AlpsBench 与真实场景的差距**：静态 query-memory 对 vs 动态会话上下文

### 风险

1. **Hermes Agent 记忆功能未就绪**：如果 agent 当前没有记忆召回功能，shadow mode 无法部署
2. **用户隐私顾虑**：即使本地存储，用户可能不愿启用 trace 收集
3. **原生特征无增益**：如果合成元数据训练的模型在真实 trace 上无提升，需要收集更多真实数据后重新训练

---

## 交付物清单（当前阶段）

- [x] 记忆实体 Schema 文档
- [x] 记忆实体 Python 实现
- [x] 原生特征提取函数（14 维）
- [x] 硬过滤规则实现
- [x] Shadow mode trace schema 文档
- [x] Shadow mode collector 实现（SQLite）
- [x] 生产指标计算函数
- [x] Git 提交并推送
- [ ] 扩展 Ranker 训练（原生特征）
- [ ] Citation 检测实现
- [ ] Hermes Agent 集成设计
- [ ] Shadow mode 部署计划

---

## 建议的下一步行动

**立即执行（无阻塞）：**

1. 训练扩展版 Hybrid Ranker（加入 14 个原生特征）
2. 实现 citation 检测启发式
3. 编写 Hermes Agent 集成设计文档

**需要决策/授权：**

1. 是否在真实 Hermes Agent 会话中启用 shadow mode（需用户 opt-in）
2. 如何获取真实的 scope / authority / validity 元数据（需 Hermes runtime 支持）
3. 是否继续训练其他专家模块（RETRIEVE / WRITE / UPDATE），还是先完成 RANK 的完整闭环

**建议优先级：**

```text
P0: 扩展 Ranker 训练 + Citation 检测 + 集成设计
P1: Shadow mode 部署（取决于 agent 就绪程度和用户授权）
P2: 真实 trace 评估与模型迭代
P3: 其他专家模块
```

---

**报告完成时间：** 2026-10-29  
**下次更新：** 完成扩展 Ranker 训练后
