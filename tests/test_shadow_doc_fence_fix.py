"""Test that shadow-mode-trace-schema.md Export Pipeline uses proper markdown code fence."""
from pathlib import Path

def test_export_pipeline_has_proper_bash_fence():
    """Verify Export Pipeline section uses ```bash fence, not *** delimiter."""
    doc_path = Path("docs/shadow-mode-trace-schema.md")
    
    # Read raw bytes to avoid any caching
    with open(doc_path, 'rb') as f:
        raw_bytes = f.read()
    content = raw_bytes.decode('utf-8')
    
    # Find the export command section
    export_idx = content.find('Export shadow traces to JSONL training samples with explicit authorization:')
    assert export_idx != -1, "Could not find Export shadow traces line"
    
    # Get the next 500 characters after this line
    section = content[export_idx:export_idx + 500]
    
    # 1. No real asterisks (chr(0x2a) = *) should be used as fence delimiter
    real_asterisk_triple = chr(0x2a) * 3  # Real *** characters
    assert real_asterisk_triple not in section, \
        f"Found literal '***' (asterisks) in Export Pipeline section"
    
    # 2. Must contain opening ```bash
    assert '```bash' in section, "Missing ```bash code fence opener"
    
    # 3. Opening ```bash must come before the build command
    bash_pos = section.find('```bash')
    build_cmd_pos = section.find('python scripts/build_shadow_training_data.py')
    assert bash_pos != -1 and build_cmd_pos != -1, "Missing bash fence or build command"
    assert bash_pos < build_cmd_pos, "```bash must come before build command"
    
    # 4. Must have closing ``` after the command
    closing_fence_pos = section.find('```', bash_pos + 7)  # +7 to skip opening ```bash
    assert closing_fence_pos != -1, "Missing closing ``` fence"
    assert closing_fence_pos > build_cmd_pos, "Closing ``` must come after build command"
    
    # 5. Required flags must be within the fenced body
    fenced_body = section[bash_pos:closing_fence_pos]
    assert '--source-db' in fenced_body, "Missing --source-db flag in fenced block"
    assert '--output-dir' in fenced_body, "Missing --output-dir flag in fenced block"
    assert '<EXPLICIT_SOURCE_DB>' in fenced_body, "Missing <EXPLICIT_SOURCE_DB> placeholder"
    assert '<OUTPUT_DIR>' in fenced_body, "Missing <OUTPUT_DIR> placeholder"


def test_p2_report_backup_status():
    """Verify P2_COMPLETION_REPORT.md shows backups as deleted, not retained."""
    report_path = Path("docs/P2_COMPLETION_REPORT.md")
    content = report_path.read_text(encoding='utf-8')
    
    # Find the Backup section
    backup_idx = content.find('### Backup')
    assert backup_idx != -1, "Could not find Backup section in P2 report"
    
    # Get the backup section (next 200 chars)
    backup_section = content[backup_idx:backup_idx + 200]
    
    # Should say "deleted", not "retained"
    assert 'deleted' in backup_section.lower(), "Backup section should mention 'deleted'"
    assert 'retained' not in backup_section.lower(), "Backup section should not say 'retained'"
