"""
P2 Round 5: Export Pipeline 主导出示例 code fence 完整性验证

验证 shadow-mode-trace-schema.md Export Pipeline 章节中的主导出示例：
1. 使用标准 ```bash 开头（不是 ***）
2. 有匹配的闭合 ```
3. 导出命令和所有 flags 都在 fenced block 内部
4. 使用占位符而非真实路径

该测试必须能够识别旧的 *** fixture 作为 violation。
"""
import re
from pathlib import Path

DOCS_ROOT = Path(__file__).parent.parent.parent / "docs"
SHADOW_DOC = DOCS_ROOT / "shadow-mode-trace-schema.md"


def extract_export_pipeline_section(content: str) -> str:
    """提取 Export Pipeline (P2) 到 Trace Event Schema 之间的内容"""
    lines = content.splitlines()  # Handle both \n and \r\n properly
    section_lines = []
    in_section = False
    
    for line in lines:
        if "## Export Pipeline" in line:  # Matches "## Export Pipeline (P2)"
            in_section = True
        elif in_section and line.startswith("## Trace Event Schema"):
            break
        
        if in_section:
            section_lines.append(line)
    
    return "\n".join(section_lines)


def validate_code_fence_structure(section: str) -> dict:
    """
    验证 code fence 结构并返回详细诊断信息
    
    Returns dict with:
        - has_triple_star: bool
        - has_bash_fence: bool  
        - fence_pairs_balanced: bool
        - export_command_in_fence: bool
        - flags_in_fence: bool
        - uses_placeholders: bool
        - violations: list[str]
    """
    violations = []
    
    # 1. 检查是否有 *** 作为 fence（独立行或行尾）
    # 可能是独立的 *** 行，或者附在前一行末尾如 "text:***"
    has_triple_star = "***" in section
    if has_triple_star:
        violations.append("发现 *** 标记（应为 ```bash）")
    
    # 2. 检查是否有标准 ```bash fence
    has_bash_fence = "```bash" in section
    if not has_bash_fence:
        violations.append("缺少标准 ```bash fence")
    
    # 3. 检查 fence 是否成对（偶数个 ```）
    fence_count = section.count("```")
    fence_pairs_balanced = (fence_count > 0 and fence_count % 2 == 0)
    if not fence_pairs_balanced:
        violations.append(f"code fence 不成对: 找到 {fence_count} 个 ```")
    
    # 4. 提取所有 bash code blocks
    bash_blocks = re.findall(r'```bash\n(.*?)```', section, re.DOTALL)
    
    # 5. 检查主导出命令是否在 fenced block 内
    export_command_in_fence = any(
        "build_shadow_training_data.py" in block 
        for block in bash_blocks
    )
    if not export_command_in_fence:
        violations.append("主导出命令 build_shadow_training_data.py 不在 fenced block 内")
    
    # 6. 检查关键 flags 是否都在 fenced block 内
    required_flags = ["--source-db", "--output-dir", "--include-text"]
    flags_in_fence = all(
        any(flag in block for block in bash_blocks)
        for flag in required_flags
    )
    if not flags_in_fence:
        missing_flags = [
            flag for flag in required_flags
            if not any(flag in block for block in bash_blocks)
        ]
        violations.append(f"关键 flags 不在 fenced block 内: {missing_flags}")
    
    # 7. 检查是否使用占位符（不是真实路径）
    uses_placeholders = True
    for block in bash_blocks:
        if "build_shadow_training_data.py" in block:
            # 应该使用 <EXPLICIT_SOURCE_DB> 和 <OUTPUT_DIR>
            if "/path/to/" in block and "<" not in block:
                uses_placeholders = False
                violations.append("使用真实路径 /path/to/ 而非占位符")
            if "shadow_traces.db" in block and "<" not in block and "EXPLICIT" not in block:
                uses_placeholders = False
                violations.append("使用具体文件名 shadow_traces.db 而非占位符")
    
    return {
        "has_triple_star": has_triple_star,
        "has_bash_fence": has_bash_fence,
        "fence_pairs_balanced": fence_pairs_balanced,
        "export_command_in_fence": export_command_in_fence,
        "flags_in_fence": flags_in_fence,
        "uses_placeholders": uses_placeholders,
        "violations": violations
    }


def test_validate_code_fence_structure_detects_triple_star():
    """验证 validate_code_fence_structure 能识别 *** fixture"""
    # 模拟旧的损坏格式
    broken_fixture = """
## Export Pipeline (P2)

### Building Training Data

Export shadow traces:
***
```bash
python scripts/build_shadow_training_data.py \\
  --source-db <EXPLICIT_SOURCE_DB> \\
  --output-dir <OUTPUT_DIR> \\
  --include-text
```
"""
    
    result = validate_code_fence_structure(broken_fixture)
    
    # 必须检测到 *** violation
    assert result["has_triple_star"], "未能检测到 *** fence"
    assert any("***" in v for v in result["violations"]), \
        "violations 列表未包含 *** 相关错误"


def test_export_pipeline_code_fence_integrity():
    """主测试：Export Pipeline 主导出示例必须使用标准 code fence"""
    assert SHADOW_DOC.exists(), f"文档不存在: {SHADOW_DOC}"
    
    content = SHADOW_DOC.read_text(encoding="utf-8")
    section = extract_export_pipeline_section(content)
    
    assert section, "未能提取 Export Pipeline 章节"
    
    result = validate_code_fence_structure(section)
    
    # 断言所有检查项
    assert not result["has_triple_star"], \
        "Export Pipeline 章节包含损坏的 *** fence（应为 ```bash）"
    
    assert result["has_bash_fence"], \
        "Export Pipeline 章节缺少标准 ```bash fence"
    
    assert result["fence_pairs_balanced"], \
        f"code fence 不成对闭合"
    
    assert result["export_command_in_fence"], \
        "主导出命令 build_shadow_training_data.py 不在 fenced block 内"
    
    assert result["flags_in_fence"], \
        "关键 flags (--source-db, --output-dir, --include-text) 不在 fenced block 内"
    
    assert result["uses_placeholders"], \
        "导出示例使用真实路径而非占位符"
    
    # 如果有任何 violations，全部报告
    if result["violations"]:
        raise AssertionError(
            f"Export Pipeline code fence 完整性验证失败:\n" +
            "\n".join(f"  - {v}" for v in result["violations"])
        )


def test_evaluator_example_uses_standard_fence():
    """Evaluating Quality 小节的 evaluator 示例必须使用标准 ```bash fence"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    
    # 提取 Evaluating Quality 小节
    lines = content.split("\n")
    section_lines = []
    in_section = False
    for line in lines:
        if "### Evaluating Quality" in line:
            in_section = True
        elif in_section and (line.startswith("### ") or line.startswith("## ")):
            break
        if in_section:
            section_lines.append(line)
    
    section_text = "\n".join(section_lines)
    
    # 不能有 ***
    assert "***" not in section_text, \
        "Evaluating Quality 小节包含损坏的 *** fence"
    
    # 必须有 ```bash
    assert "```bash" in section_text, \
        "Evaluating Quality 小节缺少 ```bash fence"
    
    # fence 必须成对
    fence_count = section_text.count("```")
    assert fence_count % 2 == 0, \
        f"Evaluating Quality code fence 不成对: 找到 {fence_count} 个 ```"
    
    # 必须使用占位符
    bash_blocks = re.findall(r'```bash\n(.*?)```', section_text, re.DOTALL)
    for block in bash_blocks:
        if "evaluate_shadow_traces.py" in block:
            assert "<OUTPUT_DIR>" in block or "./training_data" in block, \
                "evaluator 示例应使用占位符 <OUTPUT_DIR>"
