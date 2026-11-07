"""
P2 Round 4: Shadow文档 code fence 格式验证

确保 shadow-mode-trace-schema.md 中的 CLI 示例使用标准 Markdown code fence，
无损坏的 *** fence 或其他 malformed blocks。
"""
import re
from pathlib import Path

DOCS_ROOT = Path(__file__).parent.parent.parent / "docs"
SHADOW_DOC = DOCS_ROOT / "shadow-mode-trace-schema.md"


def test_export_pipeline_uses_standard_bash_fence():
    """主导出示例必须使用标准 ```bash fence，不能是 ***"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    # 直接检查：在 Building Training Data 小节不能有独立的 *** 行
    in_building_section = False
    section_lines = []
    
    for i, line in enumerate(lines):
        if "### Building Training Data" in line:
            in_building_section = True
        elif in_building_section and (line.startswith("### ") or line.startswith("## ")):
            break
        
        if in_building_section:
            section_lines.append((i + 1, line))  # 保存行号和内容
    
    # 检查是否有损坏的 *** fence（可能独立成行或附在其他内容后）
    for line_num, line in section_lines:
        if "***" in line and "```" not in line:
            raise AssertionError(
                f"第 {line_num} 行: 发现损坏的 *** fence（应为 ```bash）\n行内容: {line}"
            )
    
    # 必须包含标准 ```bash fence
    section_text = "\n".join([l for _, l in section_lines])
    assert "```bash" in section_text, "Building Training Data 小节缺少 ```bash fence"


def test_export_cli_command_structure():
    """主导出 CLI 示例必须包含正确的命令和 flags"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    
    # 提取 Building Training Data 小节
    lines = content.split("\n")
    section_lines = []
    in_section = False
    for line in lines:
        if "### Building Training Data" in line:
            in_section = True
        elif in_section and (line.startswith("### ") or line.startswith("## ")):
            break
        if in_section:
            section_lines.append(line)
    
    section_text = "\n".join(section_lines)
    
    # 必须包含主导出脚本名称
    assert "build_shadow_training_data.py" in section_text
    
    # 必须包含关键 flags
    assert "--source-db" in section_text, "缺少 --source-db flag"
    assert "--output-dir" in section_text, "缺少 --output-dir flag"
    assert "--include-text" in section_text, "缺少 --include-text flag"


def test_export_cli_uses_placeholders_not_real_paths():
    """主导出 CLI 示例必须使用占位符，不能包含真实路径"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    
    # 提取 bash code blocks
    bash_blocks = re.findall(r'```bash\n(.*?)```', content, re.DOTALL)
    
    for block in bash_blocks:
        if "build_shadow_training_data.py" in block:
            # 不能包含真实路径模式
            assert "/path/to/" not in block or "<" in block, \
                "导出示例包含真实路径而非占位符"
            assert "shadow_traces.db" not in block or "<" in block or "EXPLICIT" in block, \
                "导出示例包含具体文件名而非占位符"


def test_evaluator_uses_standard_bash_fence():
    """evaluator 示例必须使用标准 ```bash fence"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    
    # 定位 Evaluating Quality 小节
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
    
    # 必须包含标准 bash code fence
    assert "```bash" in section_text, "Evaluating Quality 小节缺少 ```bash fence"
    
    # 不能包含损坏的 *** fence
    assert "***" not in section_text, "Evaluating Quality 小节包含损坏的 *** fence"


def test_evaluator_cli_uses_placeholders():
    """evaluator CLI 示例必须使用占位符"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    
    # 提取 Evaluating Quality 小节的 bash blocks
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
    bash_blocks = re.findall(r'```bash\n(.*?)```', section_text, re.DOTALL)
    
    for block in bash_blocks:
        if "evaluate_shadow_traces.py" in block:
            # 应该使用占位符
            assert "./training_data" in block or "<" in block, \
                "evaluator 示例应使用占位符或相对路径"


def test_no_malformed_fences_in_export_pipeline():
    """Export Pipeline 章节不能包含任何 malformed code fence"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    
    # 提取整个 Export Pipeline 章节
    lines = content.split("\n")
    section_lines = []
    in_section = False
    for i, line in enumerate(lines):
        if "## Export Pipeline" in line:
            in_section = True
        elif in_section and i > 0 and line.startswith("## ") and "Export Pipeline" not in line:
            break
        if in_section:
            section_lines.append(line)
    
    section_text = "\n".join(section_lines)
    
    # 不能包含 *** 作为 fence 开始
    assert not re.search(r'^\*\*\*\s*$', section_text, re.MULTILINE), \
        "Export Pipeline 章节包含 *** fence"
    
    # 所有 code blocks 必须成对
    opening_fences = section_text.count("```")
    assert opening_fences % 2 == 0, f"code fence 不成对: 找到 {opening_fences} 个 ```"


def test_document_contains_redaction_explanation():
    """文档必须说明 --include-text 是可选的且会执行 redaction"""
    content = SHADOW_DOC.read_text(encoding="utf-8")
    
    # 在 Building Training Data 小节查找说明
    lines = content.split("\n")
    section_lines = []
    in_section = False
    for line in lines:
        if "### Building Training Data" in line:
            in_section = True
        elif in_section and (line.startswith("### ") or line.startswith("## ")):
            break
        if in_section:
            section_lines.append(line)
    
    section_text = "\n".join(section_lines).lower()
    
    # 必须提到 redaction
    assert "redact" in section_text, "Building Training Data 小节未说明 redaction"
    
    # 必须说明 --include-text 是可选的
    assert "optional" in section_text, "未说明 --include-text 是可选的"
