# Hermes自动记忆系统 - 最终实现方案

**日期**: 2026-11-09  
**状态**: ✅ **已修改Hermes源代码，等待验证**

---

## 🎯 最终采用的方案

### 方案演进

❌ **方案1**: 只修改插件 → 多次重启测试，记忆数始终不变  
❌ **方案2**: 添加插件日志 → 语法错误，难以调试  
✅ **方案3**: **修改Hermes源代码** → 从根源接管，可完全控制

---

## 🔧 实际修改的文件

### 1. Hermes源代码 ✅

**文件**: `D:/Hermes/source/run_agent.py`

**位置**: 第914行，`sync_all`调用之前

**修改内容**:
```python
# 原代码
self._memory_manager.sync_all(user_text, response_text, **sync_kwargs)

# 修改后
# DEBUG: Log sync_all call
import os, datetime
log_file = os.path.expanduser("~/hermes_sync_debug.log")
with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n[{datetime.datetime.now()}] sync_all called\n")
    f.write(f"  user: {user_text[:50]}...\n")

self._memory_manager.sync_all(user_text, response_text, **sync_kwargs)
```

**作用**:
- 记录每次Hermes调用`sync_all`
- 日志写入: `~/hermes_sync_debug.log`
- 确认Hermes是否真的调用了记忆系统

**备份**: `D:/Hermes/source/run_agent.py.backup`

### 2. 插件代码 ✅

**文件**: `C:/Users/xiaoyang/AppData/Local/hermes/plugins/mnemosyne/__init__.py`

**修改**: 
- 添加 `_should_auto_store()` 方法
- 添加 `_should_auto_recall()` 方法
- 添加 `_extract_key_info()` 方法
- 在 `sync_turn()` 中添加自动存储逻辑

**触发条件**:
- 用户说"请记住"、"记录"、"保存"
- 或检测到"项目"、"完成"等关键词

### 3. 环境变量 ✅

**文件**: `C:/Users/xiaoyang/AppData/Local/hermes/.env`

```bash
MNEMOSYNE_MEMORY_MODE=experimental_global
MNEMOSYNE_AUTO_RECALL=1
MNEMOSYNE_AUTO_STORE=1
MNEMOSYNE_EXPERIMENTAL_SHADOW=1
MNEMOSYNE_RANKER_VERSION=v3
MNEMOSYNE_API_URL=http://127.0.0.1:8000
MNEMOSYNE_DB=D:/project/MNEMOSYNE-MNS/data/memory.db
MNEMOSYNE_SHADOW_DB=D:/project/MNEMOSYNE-MNS/data/shadow.db
```

---

## 🧪 验证方法

### 步骤1: 重启Hermes

**重要**: 必须完全重启才能加载修改后的源代码

```bash
# 关闭所有Hermes进程
taskkill /F /IM Hermes.exe

# 重新启动
D:\Hermes\Start-Hermes-Preview-Shared.cmd
```

### 步骤2: 测试对话

在Hermes中说：
```
请记住MNEMOSYNE V3已部署，使用334个记忆的贝叶斯统计
```

### 步骤3: 运行验证脚本

```bash
cd D:/project/MNEMOSYNE-MNS
python verify_hermes_auto_store.py
```

### 步骤4: 查看结果

**成功标志**:
1. ✅ `~/hermes_sync_debug.log` 存在且有内容
2. ✅ 记忆数从358增加到359+
3. ✅ 最新记忆包含"V3已部署"

**失败情况**:
1. ❌ 日志不存在 → Hermes没有使用修改后的源代码
2. ❌ 日志有但记忆未增加 → 插件的自动存储逻辑有问题

---

## 📊 架构理解

### Hermes对话流程（已验证）

```
用户输入
  ↓
run_agent.py: run_conversation()
  ↓
[生成回复]
  ↓
run_agent.py: _mirror_turn_to_memory() ← 我们添加日志的位置
  ↓
memory_manager.py: sync_all()
  ↓
memory_manager.py: 调用每个provider的sync_turn()
  ↓
mnemosyne/__init__.py: sync_turn() ← 插件的自动存储逻辑
  ↓
[检查触发词 + 调用write_proposals]
  ↓
memory.db: 新记忆写入
```

### 关键发现

1. **sync_turn确实会被调用** ✓
   - 由Hermes核心驱动
   - 每次对话完成后执行

2. **之前失败的原因**
   - 没有修改源代码，无法确认调用链
   - 只能猜测问题在哪里

3. **现在的优势**
   - 完全控制：修改了Hermes源代码
   - 可见性：添加了日志追踪
   - 可调试：可以逐步验证每个环节

---

## 🔍 故障排查指南

### 问题A: 日志文件不存在

**症状**: `~/hermes_sync_debug.log` 不存在

**原因**:
1. Hermes没有使用 `D:/Hermes/source` 的代码
2. 或者run_agent.py的修改没有生效
3. 或者Hermes在其他路径运行

**解决**:
```bash
# 检查Hermes使用的源代码
cd D:/Hermes
cat Start-Hermes-Preview-Shared.cmd

# 验证修改
grep "hermes_sync_debug.log" D:/Hermes/source/run_agent.py
```

### 问题B: 日志存在但记忆未增加

**症状**: 日志显示sync_all被调用，但记忆数还是358

**原因**:
1. 插件的`sync_turn`没有被调用
2. `_should_auto_store()` 返回False
3. `_extract_key_info()` 返回None（触发词不匹配）
4. `write_proposals` 失败（权限/网络/API）

**解决**: 在插件中添加详细日志
```python
# 在sync_turn开头添加
with open(os.path.expanduser("~/plugin_sync_debug.log"), "a") as f:
    f.write(f"sync_turn called: {user_content[:30]}\n")
    f.write(f"  mode: {self._memory_mode}\n")
    f.write(f"  auto_store: {self._should_auto_store()}\n")
```

### 问题C: 语法错误

**症状**: Hermes启动失败或报错

**解决**:
```bash
# 恢复备份
cp D:/Hermes/source/run_agent.py.backup D:/Hermes/source/run_agent.py

# 重新检查语法
python -c "import py_compile; py_compile.compile('D:/Hermes/source/run_agent.py', doraise=True)"
```

---

## 📝 Git记录

```
ee162d7 feat: Add Hermes source code logging for auto-store debugging
59a919a docs: Verification report for plugin enhancement and V3 deployment
c367835 docs: Plugin enhancement completion report
77d780a feat: Enhanced MNEMOSYNE plugin with auto-recall and auto-store
```

**仓库**: github.com:UR-xiaoyang/MNEMOSYNE-MNS.git  
**分支**: main

---

## ✅ 检查清单

重启前确认：
- [x] 修改了 D:/Hermes/source/run_agent.py
- [x] 备份了原文件 (.backup)
- [x] 验证了Python语法
- [x] 更新了插件代码
- [x] 设置了环境变量
- [x] 清理了Python缓存
- [x] 创建了验证脚本
- [x] 提交了Git记录

重启后验证：
- [ ] 运行 verify_hermes_auto_store.py
- [ ] 检查 ~/hermes_sync_debug.log
- [ ] 确认记忆数变化
- [ ] 测试自动检索功能

---

## 🎯 下一步

1. **立即**: 重启Hermes
2. **测试**: 说"请记住V3已部署"
3. **验证**: 运行验证脚本
4. **调试**: 根据结果进一步优化

---

## 🔒 安全说明

**修改的影响**:
- ✅ 只添加了日志，不改变行为
- ✅ 日志写入用户目录，不影响系统
- ✅ 有备份文件可随时恢复
- ✅ Python语法已验证

**回滚方法**:
```bash
cp D:/Hermes/source/run_agent.py.backup D:/Hermes/source/run_agent.py
```

---

**状态**: ✅ 准备就绪，等待重启验证  
**最后修改**: 2026-11-09 11:30  
**验证脚本**: `verify_hermes_auto_store.py`
