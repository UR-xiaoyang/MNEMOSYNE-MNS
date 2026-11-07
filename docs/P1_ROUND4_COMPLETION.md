# P1 Round 4: 精确注入过滤与无路径 CLI 输出

**任务ID:** t_4bd23714  
**完成日期:** 2026-11-03  
**状态:** ✓ 已完成并验证

---

## 执行摘要

P1 Round 4 修复了两个独立复核发现的精确问题：

1. **过滤精度问题**: 裸 `r'<'` pattern 导致数学表达式 `x < 5` 被错误过滤
2. **CLI 隐私泄露**: stdout 打印绝对 output path

修复后：
- 数学/自然语言中的 `<` 正常保留
- 明确注入标记（DOCTYPE、System note、Skill、IMPORTANT）仍被过滤
- CLI stdout 完全不含路径
- 173 pytest + 166 unittest 全部通过，无回归

---

## 问题详情

### 1. 注入过滤过于宽泛

**实测问题:**
```python
>>> from scripts.normalize_conversations import is_framework_injection
>>> is_framework_injection("x < 5 is a valid comparison")
True  # ❌ False positive：数学表达式被拒绝
```

**根本原因:**
`scripts/normalize_conversations.py` line 62 有裸 `r'<'` pattern，任何包含 `<` 的内容都被认为是注入。

**影响范围:**
- 数学比较: `x < threshold`, `5 < 10`, `if temp < 100`
- 自然语言: `Results show A < B`, `I need value < limit`
- 所有合法内容中的角括号

**测试缺陷:**
`tests/test_p1_round3_injection_filter.py::test_should_include_allows_ordinary_angle_brackets` (line 47-56) 没有任何断言，属于空通过测试。

### 2. CLI stdout 泄露路径

**问题代码:**
```python
# scripts/normalize_conversations.py line 404
print(f"✓ Output: {args.output}")
```

**实际输出:**
```
✓ Output: C:\Users\xiaoyang\AppData\Local\Temp\tmpq4tdzkks\output.jsonl
```

**违反要求:** 
P1 Round 4 task body 明确要求 "CLI stdout 不得输出任何路径，包括 output path"。

---

## 修复实现

### 修复 1: 精确注入模式

**变更文件:** `scripts/normalize_conversations.py` lines 47-73

**修改前:**
```python
injection_patterns = [
    r'\[System note:',
    r'# Skill:',
    r'<!DOCTYPE html>',
    r'<',                          # ❌ 裸角括号，误判数学表达式
    r'\[IMPORTANT:',
    r'You are a .* assistant',
]
```

**修改后:**
```python
injection_patterns = [
    r'\[System note:',           # [System note: ...]
    r'# Skill:',                 # # Skill: skill-name
    r'<!DOCTYPE',                # <!DOCTYPE html> or <!DOCTYPE HTML PUBLIC
    r'\[IMPORTANT:',             # [IMPORTANT: ...]
    r'You are a .* assistant',   # You are a helpful assistant
    r'You are an .* assistant',  # You are an AI assistant
]
# 删除裸 r'<'，新增 'You are an' 变体
```

**设计原则:**
- 保留明确注入标记（系统提示、技能注入、HTML、重要通知）
- 删除可能误判自然语言的通用符号
- 每个 pattern 附注释说明匹配意图

### 修复 2: CLI stdout 路径脱敏

**变更文件:** `scripts/normalize_conversations.py` line 405

**修改前:**
```python
print(f"✓ Output: {args.output}")  # 泄露绝对路径
```

**修改后:**
```python
print("✓ Output written")  # 固定消息，无路径
```

### 修复 3: 补全空测试断言

**变更文件:** `tests/test_p1_round3_injection_filter.py` lines 47-56

**修改前:**
```python
def test_should_include_allows_ordinary_angle_brackets(self):
    result = should_include_message("user", "x < 5 in the equation")
    # 无断言 - 空通过
```

**修改后:**
```python
def test_should_include_allows_ordinary_angle_brackets(self):
    result = should_include_message("user", "x < 5 in the equation")
    self.assertTrue(result, "Ordinary angle brackets in math should be allowed")
```

---

## 测试验证

### 新增测试文件

**`tests/test_p1_round4_precise_filter.py` (175 lines, 9 tests)**

**TestPreciseInjectionFilter (7 tests):**
1. `test_should_include_allows_mathematical_comparisons` - 数学比较 `x < 5`, `temp < 100`
2. `test_should_include_allows_natural_language_angle_brackets` - 自然语言 `value < threshold`
3. `test_should_include_still_filters_explicit_html_doctype` - 仍过滤 `<!DOCTYPE html>`
4. `test_should_include_still_filters_system_notes` - 仍过滤 `[System note:]`
5. `test_should_include_still_filters_skill_injections` - 仍过滤 `# Skill:`
6. `test_should_include_still_filters_important_markers` - 仍过滤 `[IMPORTANT:]`
7. `test_should_include_still_filters_assistant_injections` - 仍过滤 `You are a/an ... assistant`

**TestCLINoPathOutput (2 tests):**
1. `test_normalize_cli_stdout_contains_no_output_path` - JSONL 源场景
2. `test_normalize_cli_stdout_contains_no_inventory_path` - SQLite 源场景

每个 CLI 测试验证：
- ✓ returncode == 0
- ✓ stdout 不含 output_file 路径
- ✓ stdout 不含 source_file 路径
- ✓ stdout 不含 inventory_file 路径
- ✓ stdout 不含 tmpdir 路径
- ✓ stdout 包含成功指示符和消息计数

### 测试结果

```bash
# 聚焦测试（Round 4）
$ python -m pytest tests/test_p1_round4_precise_filter.py -v
============================
9 passed in 0.48s
============================

# 修复后的 Round 3 测试
$ python -m pytest tests/test_p1_round3_injection_filter.py::TestInjectionFiltering::test_should_include_allows_ordinary_angle_brackets -v
============================
1 passed in 0.06s
============================

# 完整项目测试套件
$ python -m pytest tests/ -q
173 passed, 2 warnings in 43.01s

$ python -m unittest discover -s tests -q
Ran 166 tests in 13.657s
OK

$ python -m py_compile scripts/*.py docker/server.py
✓ 无语法错误
```

**结论:** 无回归，所有 P0+P1 Round 1-4 测试保持通过。

### 行为验证

**精确过滤行为:**
```python
# 数学表达式 - 正常保留
>>> is_framework_injection('x < 5 is valid')
False  # ✓

>>> is_framework_injection('if temperature < 100 then')
False  # ✓

# 明确注入 - 仍被过滤
>>> is_framework_injection('<!DOCTYPE html>')
True   # ✓

>>> is_framework_injection('[System note: task assignment]')
True   # ✓

>>> is_framework_injection('# Skill: test-driven-development')
True   # ✓

>>> is_framework_injection('[IMPORTANT: The user launched]')
True   # ✓

>>> is_framework_injection('You are a helpful assistant')
True   # ✓
```

**CLI 输出脱敏:**
```bash
# 之前（Round 3）
✓ Output: C:\Users\xiaoyang\AppData\Local\Temp\tmp123\output.jsonl

# 之后（Round 4）
✓ Output written
```

---

## 测试计数更新

### Round 4 新增

- **pytest:** 9 个新测试 (test_p1_round4_precise_filter.py)
- **unittest:** 9 个新测试（同上，unittest 也会发现）
- **修复空测试:** 1 个 (test_p1_round3_injection_filter.py)

### 累计总数（P0 + P1 Round 1-4）

- **pytest:** 173 passed
- **unittest:** 166 passed
- **py_compile:** 0 errors

---

## 向后兼容性

**无 Breaking Changes:**

修复类型为 **缩小过滤范围**（减少 False positive），不影响合法用例：
- 之前被错误拒绝的数学表达式现在正常保留
- 之前被正确拒绝的注入内容依然被拒绝
- CLI 行为对脚本调用者透明（returncode 和功能不变）

**CLI stdout 变更:**
- 从 `✓ Output: <path>` 改为 `✓ Output written`
- 脚本调用者若解析 stdout 需要适配，但这是隐私修复的必要变更
- 实际 output 文件路径仍通过 `--output` 参数已知，只是不回显到 stdout

---

## 文件变更清单

### 修改文件 (2)
1. `scripts/normalize_conversations.py`
   - Line 61-67: 精确注入 patterns，删除裸 `<`
   - Line 405: CLI stdout 路径脱敏
   
2. `tests/test_p1_round3_injection_filter.py`
   - Line 56: 补全空测试断言

### 新增文件 (1)
1. `tests/test_p1_round4_precise_filter.py` (175 lines)
   - 7 个精确过滤测试
   - 2 个 CLI 隐私测试

### 更新文档 (2)
1. `P1_COMPLETION_REPORT.md` - 新增 Round 4 章节
2. `docs/P1_ROUND4_COMPLETION.md` - 本文档

---

## 遵守约束确认

- [x] 仅使用 toy fixtures，未访问真实私有来源
- [x] 未运行真实 inventory/normalize（测试中的 CLI 调用使用 tempdir fixtures）
- [x] 未训练/下载/提交/push
- [x] 保持 token 强制与 SQLite mode=ro
- [x] 更新 P1 报告和测试计数
- [x] 确认 memory_dataset 无派生 artifacts
- [x] 运行 focused tests、pytest -q、unittest discover -q、py_compile
- [x] 173 pytest + 166 unittest 全部通过

---

## 总结

P1 Round 4 最终隐私/过滤修复已完成：

1. ✓ **精确注入过滤**: 删除裸 `<` pattern，数学/自然语言正常保留
2. ✓ **CLI 隐私强化**: stdout 完全不含路径
3. ✓ **空测试修复**: Round 3 遗留的无断言测试补全
4. ✓ **完整验证**: 9 个新测试 + 173 pytest + 166 unittest 全通过
5. ✓ **无回归**: P0 + P1 Round 1-3 所有测试保持通过

**最终状态:** P1 隐私边界与归一化契约已完全修复，可进入 Phase 2。
