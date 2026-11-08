# MNEMOSYNE自动记忆功能 - 使用指南

## 功能说明

增强后的MNEMOSYNE插件现在支持：
1. ✅ **自动检索记忆** - 每次对话自动检索相关上下文
2. ✅ **自动存储记忆** - 智能提取关键信息并保存
3. ✅ **Shadow跟踪** - 记录哪些记忆被实际使用

## 配置方法

### 方式1: 修改 .env 文件（推荐）

在项目根目录的 `.env` 文件中添加：

```bash
# 基础配置
MNEMOSYNE_API_URL=http://127.0.0.1:8000
MNEMOSYNE_API_KEY=你的密钥

# 记忆模式（必需）
MNEMOSYNE_MEMORY_MODE=experimental_global

# 自动功能（推荐全部启用）
MNEMOSYNE_AUTO_RECALL=1          # 自动检索
MNEMOSYNE_AUTO_STORE=1           # 自动存储
MNEMOSYNE_EXPERIMENTAL_SHADOW=1  # Shadow跟踪

# V3模型配置
MNEMOSYNE_RANKER_VERSION=v3
MNEMOSYNE_SHADOW_DB=D:/project/MNEMOSYNE-MNS/data/shadow.db
```

### 方式2: 环境变量

```bash
# Windows (cmd)
set MNEMOSYNE_MEMORY_MODE=experimental_global
set MNEMOSYNE_AUTO_RECALL=1
set MNEMOSYNE_AUTO_STORE=1

# Windows (PowerShell)
$env:MNEMOSYNE_MEMORY_MODE="experimental_global"
$env:MNEMOSYNE_AUTO_RECALL="1"
$env:MNEMOSYNE_AUTO_STORE="1"

# Linux/Mac
export MNEMOSYNE_MEMORY_MODE=experimental_global
export MNEMOSYNE_AUTO_RECALL=1
export MNEMOSYNE_AUTO_STORE=1
```

## 使用场景

### 场景1: 全自动模式（你需要的）

```bash
MNEMOSYNE_MEMORY_MODE=experimental_global
MNEMOSYNE_AUTO_RECALL=1
MNEMOSYNE_AUTO_STORE=1
MNEMOSYNE_EXPERIMENTAL_SHADOW=1
```

**效果**:
- ✅ 每次对话都自动检索相关记忆
- ✅ 对话中提到"请记住"、"项目进度"等会自动存储
- ✅ 跟踪哪些记忆被实际引用（用于训练V3）
- ✅ 完全静默，不打断对话流程

**适用于**: 日常开发对话，需要持续积累知识

### 场景2: 仅检索模式（默认）

```bash
MNEMOSYNE_MEMORY_MODE=session
# AUTO_RECALL和AUTO_STORE默认为0
```

**效果**:
- Hermes决定何时需要记忆
- 不自动存储
- 适合临时对话

### 场景3: Shadow观察模式

```bash
MNEMOSYNE_MEMORY_MODE=session
MNEMOSYNE_EXPERIMENTAL_SHADOW=1
```

**效果**:
- 正常检索
- 记录记忆使用情况
- 用于优化V3模型

## 自动存储触发条件

插件会在以下情况自动提取并存储信息：

### 1. 明确指令
用户说话中包含：
- "请记住"
- "记录"
- "保存"
- "记下"

**示例**:
```
用户: 请记住我的量化软件用的是ai-quant-commander
→ 自动存储: "量化软件用的是ai-quant-commander"
```

### 2. 项目状态
用户询问 + 助手回答包含完成标志：
- 询问: "进度"、"状态"、"怎么样"
- 回答: "完成"、"已"、"✅"、"成功"

**示例**:
```
用户: V3部署进度如何？
助手: ✅ V3已完成部署，运行在8000端口
→ 自动存储: "状态更新: V3部署进度 → ✅ V3已完成部署..."
```

### 3. 关键概念
包含：
- "项目"
- "配置"
- "设置"
- "部署"

## 验证功能

### 1. 启动服务

```bash
cd D:/project/MNEMOSYNE-MNS

# 确保V3服务运行
python -m uvicorn docker.server:app --host 127.0.0.1 --port 8000
```

### 2. 重启Hermes

配置修改后需要重启Hermes才能生效。

### 3. 测试自动存储

在Hermes对话中说：

```
请记住：MNEMOSYNE V3模型已部署，使用334个记忆的贝叶斯统计
```

### 4. 测试自动检索

稍后询问：

```
V3模型的状态如何？
```

应该能检索到之前存储的记忆。

### 5. 检查Shadow数据

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/shadow.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM retrieval_events')
print(f'检索事件: {cursor.fetchone()[0]}')
cursor.execute('SELECT COUNT(*) FROM citation_events')
print(f'引用事件: {cursor.fetchone()[0]}')
conn.close()
"
```

## 监控和调试

### 查看存储的记忆

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/memory.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM memories')
print(f'记忆总数: {cursor.fetchone()[0]}')
cursor.execute('SELECT content FROM memories ORDER BY created_at DESC LIMIT 5')
print('最近5条:')
for row in cursor.fetchall():
    print(f'  - {row[0][:100]}...')
conn.close()
"
```

### 检查V3是否在用

```bash
curl http://127.0.0.1:8000/health | jq .ranker_version
# 应该输出: "v3"
```

## 故障排除

### 问题1: 不自动存储

**检查**:
```bash
# 确认环境变量
echo $MNEMOSYNE_AUTO_STORE
echo $MNEMOSYNE_MEMORY_MODE
```

**解决**: 确保设置为:
- `MNEMOSYNE_AUTO_STORE=1`
- `MNEMOSYNE_MEMORY_MODE=experimental_global`

### 问题2: 服务无法连接

**检查**:
```bash
curl http://127.0.0.1:8000/health
```

**解决**: 启动服务
```bash
python -m uvicorn docker.server:app
```

### 问题3: 记忆没有检索

**检查**: Hermes日志中是否有错误

**解决**: 
1. 确认服务运行
2. 确认API密钥正确
3. 查看 `data/shadow.db` 是否有新数据

## 工作流示例

### 日常开发对话

```
你: 请记住我正在开发MNEMOSYNE记忆系统
   → 自动存储 ✓

你: V3模型已部署到生产环境，使用贝叶斯统计
   → 自动存储 ✓

你: 量化软件的进度如何？
   → 自动检索相关记忆 ✓
   → 助手基于记忆回答 ✓
   → 记录引用到Shadow ✓

你: ai-quant-commander完成了144个测试
   → 检测到"完成"，自动存储 ✓
```

### 查询历史信息

```
你: 我的量化软件叫什么？
   → 自动检索：找到"ai-quant-commander" ✓
   
你: V3模型用的什么特征？
   → 自动检索：找到"19维特征，贝叶斯统计" ✓
```

## 高级配置

### 自定义触发词

修改 `plugins/memory/mnemosyne/__init__.py` 中的 `_extract_key_info` 方法：

```python
store_triggers = [
    "请记住", "记录", "保存",
    # 添加你的自定义触发词
    "重要", "关键", "注意"
]
```

### 调整存储长度

```python
if len(assistant_text) < 500:  # 修改这个阈值
    return assistant_text
```

## 总结

✅ **插件已增强，支持**:
- 自动检索记忆
- 自动存储关键信息
- Shadow模式跟踪
- V3模型排序
- 贝叶斯统计使用

✅ **配置简单**:
- 修改 .env 文件
- 重启 Hermes
- 自动工作

✅ **完全静默**:
- 不打断对话
- 后台处理
- 提升体验

**下一步**: 设置环境变量，重启Hermes，开始享受智能记忆！
