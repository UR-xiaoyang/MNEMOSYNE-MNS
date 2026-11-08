# Shadow模式说明与当前策略

## 什么是Shadow模式？

Shadow模式是一种**无风险的生产验证策略**，用于在真实环境中测试新模型，但**不影响用户体验**。

### 工作原理

```
用户请求
    ↓
检索候选记忆
    ↓
┌─────────────────────────────┐
│  旧模型 (V2) 排序           │ ← 实际返回给用户
│  得分: [0.9, 0.8, 0.7...]   │
└─────────────────────────────┘
    ↓
返回给用户
    ↓ (同时)
┌─────────────────────────────┐
│  新模型 (V3) 排序           │ ← Shadow预测，仅记录
│  得分: [0.95, 0.85, 0.75...]│
└─────────────────────────────┘
    ↓
记录到 shadow_traces.db
    - 检索事件
    - 用户实际引用的记忆
    - 排序差异
```

**关键点**:
- V3模型运行，但**不影响用户看到的结果**
- 用户仍然使用V2的排序结果
- 收集V3的预测和真实用户行为的对比数据

---

## 当前MNEMOSYNE的Shadow策略

### 1. Shadow数据收集

**已实现功能**:

```python
# scripts/shadow_mode.py - ShadowModeCollector
class ShadowModeCollector:
    """收集生产追踪数据，用于模型验证和持续学习"""
    
    def log_retrieval(self, session_id, turn_id, query, candidates, metadata):
        """记录检索事件"""
        # 存储查询和候选记忆
        
    def log_citation(self, session_id, turn_id, retrieval_event_id, 
                     memory_id, citation, context_position, original_rank):
        """记录用户实际引用的记忆"""
        
    def log_ignored(self, session_id, turn_id, retrieval_event_id, 
                    memory_id, rank, injected):
        """记录未被引用的记忆"""
        
    def compute_metrics(self, session_id):
        """计算会话级别的检索质量指标"""
```

**存储位置**: SQLite数据库
- 默认路径: `shadow_traces.db`
- 当前配置: `D:/project/MNEMOSYNE-MNS/data/shadow.db`

**数据表结构**:
```sql
retrieval_events    # 检索事件
  - event_id, timestamp
  - session_id, turn_id
  - query_text, candidates_json
  
citation_events     # 引用事件（正样本）
  - event_id, timestamp
  - memory_id, original_rank
  - context_position
  
ignored_events      # 未引用事件（负样本）
  - memory_id, rank
  - injected (是否强制注入)
  
correction_events   # 用户纠正
  - correction_type
  - feedback_json
```

### 2. 当前环境配置

从`.env`文件看到：

```bash
MNEMOSYNE_SHADOW_MODE=1              # Shadow模式已启用 ✓
MNEMOSYNE_SHADOW_DB=D:/project/MNEMOSYNE-MNS/data/shadow.db
MNEMOSYNE_API_URL=http://127.0.0.1:8000
MNEMOSYNE_DB=D:/project/MNEMOSYNE-MNS/data/memory.db
```

**当前状态**: ✅ **Shadow模式已启用**

### 3. Shadow模式的用途

#### A. 贝叶斯统计收集

**当前问题**: V3模型的贝叶斯特征在推理时默认为零
```python
# scripts/rank_v3_features.py
# 13-18: Bayesian features default to zeros without real statistics
features[i, 13] = 0.0  # bayes_mean
features[i, 14] = 0.0  # bayes_std
features[i, 15] = 0.0  # bayes_ci_width
features[i, 16] = 0.0  # bayes_coverage
features[i, 17] = 0.0  # bayes_consistency
features[i, 18] = 0.0  # bayes_confidence
```

**Shadow模式解决方案**:
```python
# 从shadow_traces.db收集真实统计
def collect_bayesian_stats(shadow_db):
    """
    从Shadow数据计算每个记忆的贝叶斯统计
    """
    stats = {}
    for memory_id in all_memories:
        # 检索次数
        n_retrievals = count_retrieval_events(memory_id)
        
        # 引用次数
        n_citations = count_citation_events(memory_id)
        
        # Beta分布参数
        alpha = 1.0 + n_citations
        beta = 1.0 + (n_retrievals - n_citations)
        
        stats[memory_id] = {
            'n_retrievals': n_retrievals,
            'n_citations': n_citations,
            'bayes_mean': alpha / (alpha + beta),
            'bayes_std': compute_std(alpha, beta),
            ...
        }
    
    return stats
```

#### B. 模型性能验证

**可以计算的指标**:

```python
# 从shadow_traces.db分析
metrics = {
    # 排序质量
    'ndcg@10': compute_ndcg(predicted_rank, actual_citations),
    'mrr': compute_mrr(predicted_rank, actual_citations),
    'precision@5': compute_precision(predicted_rank, actual_citations),
    
    # 覆盖率
    'citation_rate': n_citations / n_retrievals,
    'top5_hit_rate': hit_rate_in_top_k(k=5),
    
    # 少数类性能
    'minority_recall': recall_on_rare_memories(),
}
```

#### C. A/B测试数据

对比V2和V3的真实表现：
```
V2 (生产)        V3 (Shadow)
NDCG: 0.805      NDCG: 0.87?
Citation率: 89%  Citation率: 92%?
```

---

## V3部署的Shadow策略

### Phase 1: 纯Shadow模式 (Week 1-2)

**配置**:
```bash
# .env 或环境变量
MNEMOSYNE_RANKER_VERSION=v2          # 生产仍用V2
MNEMOSYNE_SHADOW_MODE=1              # 启用Shadow收集
MNEMOSYNE_V3_SHADOW=1                # 同时运行V3（新增）
```

**行为**:
- 用户请求 → V2排序 → 返回结果
- 后台同时 → V3排序 → 记录预测
- 收集贝叶斯统计到 `memory.db`

**数据收集**:
```python
# 每次检索事件
for memory_id in retrieved_memories:
    # 更新统计
    memory_stats[memory_id]['n_retrievals'] += 1
    
    # 如果被引用
    if memory_id in cited_memories:
        memory_stats[memory_id]['n_citations'] += 1
```

### Phase 2: A/B测试 (Week 3-4)

**配置**:
```bash
MNEMOSYNE_RANKER_VERSION=v3          # 切换到V3
MNEMOSYNE_AB_RATIO=0.1               # 10%流量
MNEMOSYNE_SHADOW_MODE=1              # 继续收集
```

**行为**:
- 90%用户 → V2排序
- 10%用户 → V3排序
- 对比两组的指标

### Phase 3: 渐进上线 (Week 5+)

**配置**:
```bash
MNEMOSYNE_RANKER_VERSION=v3
MNEMOSYNE_AB_RATIO=0.2 → 0.5 → 1.0
```

---

## 如何使用Shadow数据

### 1. 查看收集的数据

```bash
# 连接Shadow数据库
sqlite3 data/shadow.db

# 查看检索统计
SELECT COUNT(*) FROM retrieval_events;

# 查看引用统计
SELECT COUNT(*) FROM citation_events;

# 每个记忆的统计
SELECT 
    memory_id,
    COUNT(*) as n_retrievals,
    SUM(CASE WHEN cited THEN 1 ELSE 0 END) as n_citations
FROM (
    SELECT memory_id, 1 as cited 
    FROM citation_events
    UNION ALL
    SELECT memory_id, 0 as cited 
    FROM ignored_events
)
GROUP BY memory_id;
```

### 2. 填充贝叶斯特征

创建脚本：
```python
# scripts/populate_bayesian_stats.py
def populate_from_shadow():
    """从Shadow数据库填充贝叶斯统计到memory.db"""
    
    shadow_conn = sqlite3.connect('data/shadow.db')
    memory_conn = sqlite3.connect('data/memory.db')
    
    # 计算每个记忆的统计
    stats = compute_stats_from_shadow(shadow_conn)
    
    # 写入memory.db
    for memory_id, stat in stats.items():
        memory_conn.execute("""
            INSERT OR REPLACE INTO bayesian_stats
            (memory_id, n_retrievals, n_citations, last_updated)
            VALUES (?, ?, ?, ?)
        """, (memory_id, stat['n_ret'], stat['n_cit'], time.time()))
    
    memory_conn.commit()
```

### 3. 用真实统计重训练

```python
# scripts/train_v3_with_real_stats.py
# 使用真实贝叶斯统计重新训练V3
real_stats = load_bayesian_stats_from_db('data/memory.db')

# 提取特征时使用真实统计
features = extract_v3_features_with_bayesian(
    queries, memories, memory_ids, 
    e5_cosine, real_stats
)
```

---

## 当前状态总结

### ✅ 已就绪
- Shadow模式代码已实现
- `.env`中已启用Shadow (`MNEMOSYNE_SHADOW_MODE=1`)
- 数据库路径已配置
- V3模型已部署并可推理

### 🔄 待实施
1. **启动服务** (使用V2，开启Shadow收集)
2. **添加V3 Shadow预测** (代码修改)
3. **收集数据** (1-2周)
4. **分析统计** (脚本)
5. **填充贝叶斯特征** (更新memory.db)
6. **切换到V3** (环境变量)

---

## 推荐行动

### Immediate (今天)

```bash
# 1. 确认Shadow数据库存在
ls -la data/shadow.db

# 2. 启动服务（当前使用V2+Shadow）
python -m uvicorn docker.server:app --reload

# 3. 检查Shadow记录
sqlite3 data/shadow.db "SELECT COUNT(*) FROM retrieval_events;"
```

### Short-term (本周)

1. 修改代码，添加V3 Shadow预测
2. 让V3在后台运行，记录预测但不返回
3. 收集1-2周真实数据

### Medium-term (下周)

1. 分析Shadow数据
2. 计算V3性能指标
3. 填充贝叶斯统计
4. 决定是否切换到V3

---

**当前结论**: Shadow模式基础设施已完备，可以开始收集生产数据验证V3性能。
