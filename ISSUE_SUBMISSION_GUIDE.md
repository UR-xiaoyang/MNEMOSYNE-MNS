# Issue提交指南

## GitHub Issue链接

Hermes Agent仓库: https://github.com/NousResearch/hermes-agent/issues

## Issue标题

```
Feature Request: Memory Provider Auto-Store Support
```

## Issue标签

建议添加:
- `enhancement`
- `memory-provider`
- `feature-request`

## Issue正文

直接复制 `HERMES_FEATURE_REQUEST.md` 的内容即可。

## 补充说明

如果Hermes团队询问更多细节，可以提供：

### 1. 当前实现的临时方案

我们目前通过修改 `run_agent.py` 实现，但这不是可持续的方案：

**修改位置**: `run_agent.py` 第918行
```python
# 在 sync_all 调用前添加日志
import os, datetime
log_file = os.path.expanduser("~/hermes_sync_debug.log")
with open(log_file, "a") as f:
    f.write("[" + str(datetime.datetime.now()) + "] sync_all called\n")
```

**插件实现**: `plugins/memory/mnemosyne/__init__.py`
```python
def sync_turn(self, user_content, assistant_content, **kwargs):
    # 尝试在这里做自动存储
    if self._should_auto_store() and self._memory_mode == "experimental_global":
        key_info = self._extract_key_info(user_content, assistant_content)
        if key_info:
            self._client.write_proposals(key_info)
```

### 2. 遇到的问题

1. **不清楚sync_turn是否允许写操作**
   - 文档说"只记录对话历史"
   - 但没说能不能调用write API
   - 导致不确定这样做是否符合设计

2. **修改源代码不可持续**
   - 每次升级Hermes都需要重新修改
   - 无法与其他用户分享方案
   - 维护成本高

3. **缺少官方指导**
   - Memory Provider文档只讲检索
   - 没有自动存储的最佳实践
   - 不知道正确的实现方式

### 3. 期望的API设计

最理想的情况是：

```python
# Hermes核心
class MemoryManager:
    def on_turn_complete_all(self, user_text, response_text, **kwargs):
        """新增方法：允许provider做写操作"""
        for provider in self._providers:
            provider.on_turn_complete(user_text, response_text, **kwargs)

# 插件实现
class MnemosyneMemoryProvider(MemoryProvider):
    def on_turn_complete(self, user_content, assistant_content, **kwargs):
        """明确允许写操作的钩子"""
        if self._detect_trigger(user_content):
            self._store_memory(user_content, assistant_content)
```

### 4. 社区价值

这个功能不仅对MNEMOSYNE有用，其他memory provider也会受益：
- 任何想做"记住这个"功能的provider
- 需要追踪项目状态的provider
- 想要自动知识积累的provider

### 5. 愿意贡献PR

如果Hermes团队同意这个方向，我们可以：
1. 实现 `on_turn_complete` hook
2. 更新Memory Provider文档
3. 提供MNEMOSYNE作为参考实现
4. 写测试用例

---

## 发Issue前的准备

1. ✅ 检查现有issues，确保没有重复
2. ✅ 准备好代码示例
3. ✅ 说明use case
4. ✅ 提出具体的API建议
5. ✅ 表达愿意贡献的意愿

## 预期回应

可能的回应：

**情况1: 接受**
- "Good idea, we'll add this in v0.23"
- "Please submit a PR"
→ 你可以开始实现

**情况2: 需要讨论**
- "Interesting, but have you considered..."
- "What about edge case X?"
→ 继续讨论设计

**情况3: 已支持**
- "You can already do this with..."
- "Check the docs for..."
→ 学习正确用法

**情况4: 不接受**
- "This is out of scope"
- "We prefer plugins to handle this"
→ 继续用临时方案

## 如果Issue被接受

需要实现的文件：

1. **`agent/memory_provider.py`**
   - 添加 `on_turn_complete` 抽象方法

2. **`agent/memory_manager.py`**
   - 添加 `on_turn_complete_all` 方法
   - 在后台worker中调用

3. **`run_agent.py`**
   - 在 `_mirror_turn_to_memory` 中调用
   - 添加触发词检测（可选）

4. **文档**
   - 更新Memory Provider integration guide
   - 添加auto-store示例

5. **测试**
   - `tests/agent/test_memory_auto_store.py`

---

**准备好了吗？可以去发Issue了！**

GitHub链接: https://github.com/NousResearch/hermes-agent/issues/new
