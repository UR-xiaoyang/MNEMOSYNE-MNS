# MNEMOSYNE-MNS MoE 性能优化路线图

## 📊 总体目标

**核心指标提升**：
- RANK NDCG@10: 0.5796 → **0.65+**
- RETRIEVE Recall@100: ~0.70 → **0.85+**
- 端到端延迟: 600ms → **< 800ms**
- Citation Rate: ~30% → **40%+**

## 🗂️ 任务看板概览

### 工作流 1: 数据扩充与清洗（3 个任务）

| Task ID | 任务 | 状态 | 优先级 | 依赖 |
|---------|------|------|--------|------|
| t_b5527692 | 爬取公开数据集（MS MARCO, NQ, BEIR） | ✅ Ready | P10 | - |
| t_08f15f4f | 提取 Agent 历史记录 | ✅ Ready | P9 | - |
| t_e8541a95 | 构建统一训练集 v2 | ⏸️ Blocked | P10 | ↑ 两个任务完成 |

**预期收益**：训练数据从 312 queries → **20K+ queries**

---

### 工作流 2: 核心模块优化（4 个任务）

| Task ID | 任务 | 状态 | 优先级 | 依赖 | 预期提升 |
|---------|------|------|--------|------|---------|
| t_2db59121 | RETRIEVE 双路召回（FTS+FAISS） | ✅ Ready | P9 | - | Recall +10-15% |
| t_325856c1 | RANK 增强特征 + Listwise | ⏸️ Blocked | P10 | ↑ 统一数据集 | NDCG +3-5% |
| t_ca0b1566 | RANK GPU Cross-Encoder 微调 | ⏸️ Blocked | P8 | ↑ Enhanced RANK | NDCG +5-8% |
| t_67a1263c | ROUTE 路由分类器 | ⏸️ Blocked | P7 | ↑ Agent 历史 | 准确率 +5-10% |

**关键路径**：数据 → Enhanced RANK → Cross-Encoder

---

### 工作流 3: 新增专家模块（4 个任务）

| Task ID | 任务 | 状态 | 优先级 | 依赖 | 功能 |
|---------|------|------|--------|------|------|
| t_0092caa0 | AUDIT 质量监控 | ✅ Ready | P8 | - | 自动告警/回滚 |
| t_eaac914c | CONFLICT NLI 冲突检测 | ✅ Ready | P6 | - | 精度 +10-20% |
| t_733e3453 | WRITE 序列标注提取 | ⏸️ Blocked | P5 | ↑ Agent 历史 | Recall +15-25% |
| t_a19a98f4 | P3 在线学习闭环 | ⏸️ Blocked | P9 | ↑ RANK + AUDIT | 持续优化 |

**新能力**：质量监控、语义冲突、智能提取、在线学习

---

### 工作流 4: 架构演进（2 个任务）

| Task ID | 任务 | 状态 | 优先级 | 依赖 |
|---------|------|------|--------|------|
| t_32afac62 | 子 MoE 架构探索 | ✅ Ready | P4 | - |
| t_6802055d | 端到端集成测试 | ⏸️ Blocked | P9 | ↑ 多个核心模块 |

---

## 🚀 执行计划

### Phase 1: 基础设施（Week 1-2）✅ 可立即启动

**并行任务**：
1. **t_b5527692**: 爬取公开数据集（MS MARCO, Natural Questions, BEIR）
2. **t_08f15f4f**: 提取 Agent 历史记录（~/.hermes/, shadow_traces.db）
3. **t_2db59121**: 实现双路召回（FTS + FAISS）
4. **t_0092caa0**: 实现 AUDIT 专家
5. **t_eaac914c**: 实现 CONFLICT NLI 检测

**预期输出**：
- 20K+ 训练样本
- 召回率提升 10-15%
- 质量监控上线
- 冲突检测精度提升

---

### Phase 2: 核心优化（Week 3-4）⏸️ 等待 Phase 1

**串行依赖**：
1. **t_e8541a95**: 整合数据集（依赖前两个数据任务）
2. **t_325856c1**: RANK 增强特征 + Listwise（依赖统一数据集）
3. **t_67a1263c**: ROUTE 分类器（依赖 Agent 历史）
4. **t_733e3453**: WRITE 序列标注（依赖 Agent 历史）

**预期输出**：
- RANK NDCG@10: 0.5796 → **0.62+**
- ROUTE 准确率提升
- WRITE Recall 提升

---

### Phase 3: GPU 训练（Week 5-6）⏸️ 需远程服务器

**任务**：
1. **t_ca0b1566**: Cross-Encoder 微调（远程 GPU：jx@172.29.2.51）
   - 训练时间：~3-5 小时
   - 验证：Offline Gate + Shadow Gate
   
**预期输出**：
- RANK NDCG@10: 0.62 → **0.65+**
- 可能需要模型蒸馏（控制延迟）

---

### Phase 4: 闭环与集成（Week 7-8）⏸️ 等待核心模块

**任务**：
1. **t_a19a98f4**: 启动 P3 在线学习闭环
2. **t_6802055d**: 端到端集成测试
3. **t_32afac62**: 子 MoE 架构探索（可选）

**预期输出**：
- 完整迭代闭环运行
- 所有模块协同验证
- 性能持续提升机制

---

## 📈 性能演进预测

| Milestone | RANK NDCG@10 | RETRIEVE Recall@100 | 端到端延迟 | Citation Rate |
|-----------|--------------|---------------------|------------|--------------|
| **Baseline** | 0.5796 | ~0.70 | 600ms | ~30% |
| Phase 1 完成 | 0.5796 | **0.80** | 650ms | 32% |
| Phase 2 完成 | **0.62** | 0.82 | 700ms | 35% |
| Phase 3 完成 | **0.65** | 0.85 | 750ms | 38% |
| Phase 4 完成 | **0.67+** | **0.85+** | **< 800ms** | **40%+** |

---

## ⚠️ 风险与缓解

### 数据风险
- **风险**：公开数据集与真实场景不匹配
- **缓解**：优先使用 Agent 历史，公开数据作为补充

### 延迟风险
- **风险**：Cross-Encoder 推理延迟过高
- **缓解**：知识蒸馏 / 模型量化 / 候选截断

### 质量风险
- **风险**：新模型可能退化
- **缓解**：严格执行 data/offline/shadow 三重门，AUDIT 实时监控

### 资源风险
- **风险**：远程 GPU 不可用
- **缓解**：Phase 1-2 可纯 CPU 完成，Phase 3 灵活调度

---

## 🎯 关键验证点

### Data Gate（数据质量门）
- [ ] 样本量 >= 20K queries
- [ ] Citation 标签覆盖率 >= 30%
- [ ] 隐私扫描通过（无路径/token 泄露）
- [ ] 时序完整性（query_timestamp 单调）

### Offline Gate（离线评估门）
- [ ] NDCG@10 95% CI 不退化
- [ ] Paired bootstrap P(candidate > baseline) > 0.95
- [ ] Correction-risk 不升
- [ ] 所有 query group 可报告

### Shadow Gate（线上监控门）
- [ ] Citation rate 不恶化（7-14 天滑动窗口）
- [ ] Correction rate 不升
- [ ] Context pollution < 10%
- [ ] 延迟 P99 < 预算

---

## 📝 下一步行动

### 立即可启动（无依赖）
1. **数据爬取**：t_b5527692（MS MARCO, NQ, BEIR）
2. **Agent 历史提取**：t_08f15f4f（~/.hermes/, shadow_traces.db）
3. **双路召回**：t_2db59121（FTS + FAISS）
4. **AUDIT 专家**：t_0092caa0（质量监控）
5. **CONFLICT NLI**：t_eaac914c（语义冲突）

### 推荐执行策略

**选项 A：全面并行**（推荐）
- 同时启动 Phase 1 的 5 个任务
- 预计 1-2 周完成
- 为 Phase 2 准备完整基础设施

**选项 B：关键路径优先**
- 优先完成数据扩充（t_b5527692, t_08f15f4f）
- 然后串行执行 RANK 优化链
- 适合资源受限场景

**选项 C：增量验证**
- 每完成一个任务立即验证
- 性能提升不达预期时调整方案
- 最保守，但风险最低

---

## 📞 需要明确的问题

1. **Agent 历史访问权限**：
   - `~/.hermes/conversations/` 路径是否正确？
   - Shadow traces DB 是否有足够数据？

2. **远程 GPU 可用性**：
   - jx@172.29.2.51 可用时间段？
   - 是否需要排队/预约？

3. **执行策略偏好**：
   - 选择 A/B/C 中哪一个？
   - 是否需要人工审批某些决策点？

---

## 🏁 最终交付物

### 代码与模型
- [ ] `runs/rank_enhanced_v2/` - 增强 RANK 模型
- [ ] `runs/rank_crossencoder_v1/` - GPU Cross-Encoder
- [ ] `runs/route_classifier_v1/` - ROUTE 分类器
- [ ] `runs/write_tagger_v1/` - WRITE 序列标注
- [ ] `deploy/faiss_index.bin` - FAISS 索引
- [ ] `audit_expert.py` - AUDIT 监控
- [ ] `conflict_expert.py` - CONFLICT NLI

### 文档与报告
- [ ] 数据集统计报告
- [ ] 模型性能对比报告
- [ ] 端到端集成测试报告
- [ ] P3 迭代闭环运行日志
- [ ] 最终性能提升总结

### 生产部署
- [ ] Docker 镜像更新
- [ ] API 文档更新
- [ ] 部署脚本更新
- [ ] 监控仪表盘

---

**总计**：13 个任务，预计 6-8 周完成，目标性能提升 15-20%。
