# MNEMOSYNE插件配置说明

## 环境变量

### 基础配置
- `MNEMOSYNE_API_URL`: API地址（默认: http://127.0.0.1:8000）
- `MNEMOSYNE_API_KEY`: API密钥（可选）

### 记忆模式
- `MNEMOSYNE_MEMORY_MODE`: 记忆模式
  - `session` (默认): 仅当前会话
  - `experimental_global`: 全局记忆（自动存储）
  - `experimental_shadow`: Shadow模式（引用跟踪）

### 自动记忆功能
- `MNEMOSYNE_AUTO_RECALL`: 每次对话自动检索记忆（默认: 0）
  - `0`: 按需检索（Hermes决定）
  - `1`: 强制每次都检索
  
- `MNEMOSYNE_AUTO_STORE`: 对话完成后自动存储记忆（默认: 0）
  - `0`: 不自动存储
  - `1`: 自动提取并存储重要信息

### Shadow模式
- `MNEMOSYNE_EXPERIMENTAL_SHADOW`: 启用引用跟踪（默认: 0）
  - `1`: 记录用户引用的记忆

## 使用场景

### 场景1: 仅检索（默认）
```bash
MNEMOSYNE_MEMORY_MODE=session
```
- Hermes决定何时需要记忆
- 只读取，不自动写入

### 场景2: 全自动模式（你需要的）
```bash
MNEMOSYNE_MEMORY_MODE=experimental_global
MNEMOSYNE_AUTO_RECALL=1
MNEMOSYNE_AUTO_STORE=1
MNEMOSYNE_EXPERIMENTAL_SHADOW=1
```
- 每次对话都检索
- 对话结束自动存储
- 跟踪记忆使用情况

### 场景3: Shadow观察模式
```bash
MNEMOSYNE_MEMORY_MODE=session
MNEMOSYNE_EXPERIMENTAL_SHADOW=1
```
- 正常检索
- 记录哪些记忆被实际使用
- 用于训练V3的贝叶斯统计
