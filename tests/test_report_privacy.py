"""
Test that P1 reports do not contain sensitive private source information.

This test enforces the privacy boundary by ensuring that completion reports
do not disclose real private source counts, absolute paths, or machine-local
directory structures that could reveal the user's conversation history scale.
"""

import re
from pathlib import Path


def test_p1_completion_report_contains_no_sensitive_counts():
    """P1_COMPLETION_REPORT.md must not contain real private source counts or scale details."""
    report_path = Path("P1_COMPLETION_REPORT.md")
    
    if not report_path.exists():
        # Report doesn't exist yet, test passes
        return
    
    content = report_path.read_text(encoding="utf-8")
    
    # Detect patterns where numbers appear near inventory/scan terminology
    # This catches phrases like "245 sessions", "55,896 messages", "627 个 Claude JSONL 文件"
    # in the context of unauthorized scans or actual detection results.
    
    # Pattern: number + session/message/file terminology (case-insensitive, with optional formatting)
    sensitive_patterns = [
        (r'\b\d+\s+sessions?\b', 'session count'),
        (r'\b\d[\d,]+\s+messages?\b', 'message count'),  # Matches "55,896 messages"
        (r'\d+\s+个\s+(?:Claude|Hermes|Codex|OpenCode)\s+\w+\s+文件', 'Chinese file count with agent name'),
        (r'\d+\s+个\s+\w+\s+数据库', 'Chinese database count'),
        (r'\*\*\d+\s+个\s+.*数据库\*\*', 'emphasized Chinese database count'),
        (r'\*\*\d+\s+个\s+.*文件\*\*', 'emphasized Chinese file count'),
    ]
    
    violations = []
    for pattern, description in sensitive_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Filter out acceptable test-related numbers
            # Allow small numbers in test contexts (e.g., "3 sessions")
            # but flag large-scale inventory numbers
            real_violations = []
            for match in matches:
                # Extract the number from the match
                num_str = re.search(r'\d[\d,]*', match)
                if num_str:
                    num = int(num_str.group().replace(',', ''))
                    # Flag if number suggests real inventory scale (>50 for sessions/files, >1000 for messages)
                    if (('session' in match.lower() or '文件' in match or '数据库' in match) and num > 50) or \
                       ('message' in match.lower() and num > 1000):
                        real_violations.append(match)
            
            if real_violations:
                violations.append(f"{description}: {real_violations}")
    
    assert not violations, (
        f"P1_COMPLETION_REPORT.md contains real inventory-scale counts: {'; '.join(violations)}. "
        "These numbers reveal the actual private source scale from unauthorized scans. "
        "Replace the entire section with generic language like "
        "'Prior unauthorized default discovery occurred; related artifacts have been removed.'"
    )


def test_p1_completion_report_contains_no_absolute_paths():
    """P1_COMPLETION_REPORT.md must not contain absolute or user directory paths."""
    report_path = Path("P1_COMPLETION_REPORT.md")
    
    if not report_path.exists():
        return
    
    content = report_path.read_text(encoding="utf-8")
    
    # Patterns that indicate absolute or user-specific paths
    forbidden_patterns = [
        r"C:\\Users\\",
        r"C:/Users/",
        r"/home/\w+",
        r"D:\\",
        r"E:\\",
    ]
    
    for pattern in forbidden_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        assert not matches, (
            f"P1_COMPLETION_REPORT.md contains absolute path pattern '{pattern}': {matches}. "
            "Use generic placeholders like $SOURCE_PATH or relative paths instead."
        )


def test_p1_completion_report_contains_no_private_directory_references():
    """P1_COMPLETION_REPORT.md must not reference specific private directories."""
    report_path = Path("P1_COMPLETION_REPORT.md")
    
    if not report_path.exists():
        return
    
    content = report_path.read_text(encoding="utf-8")
    
    # Directory names that indicate private local sources
    # Note: LOCALAPPDATA and USERPROFILE as environment variable NAMES in 
    # technical documentation (like dataset-card-v1.md) are OK - those are
    # generic placeholders. But in a completion report describing what was
    # actually scanned, they reveal the real local structure.
    forbidden_in_reports = [
        "LOCALAPPDATA/hermes",
        "USERPROFILE/.claude",
        "USERPROFILE/.codex",
        "LOCALAPPDATA/opencode",
        "AppData/Local/hermes",
        "AppData/Roaming/Claude",
    ]
    
    for forbidden in forbidden_in_reports:
        # Case-insensitive check
        assert forbidden.lower() not in content.lower(), (
            f"P1_COMPLETION_REPORT.md contains private directory reference '{forbidden}'. "
            "These references reveal the actual local directory structure that was scanned. "
            "Replace with generic descriptions like 'local agent history directories'."
        )


def test_p1_related_reports_contain_no_sensitive_counts():
    """Other P1 round reports must not contain the real private source count."""
    report_paths = [
        Path("docs/P1_ROUND3_COMPLETION.md"),
        Path("docs/P1_ROUND4_COMPLETION.md"),
    ]
    
    for report_path in report_paths:
        if not report_path.exists():
            continue
        
        content = report_path.read_text(encoding="utf-8")
        
        assert "628" not in content, (
            f"{report_path} contains '628', the real unauthorized source count. "
            "Replace with generic language."
        )


def test_dataset_card_env_vars_allowed():
    """
    Verify that dataset-card-v1.md can still contain environment variable
    documentation - these are generic technical placeholders, not disclosures.
    """
    card_path = Path("docs/dataset-card-v1.md")
    
    if not card_path.exists():
        return
    
    content = card_path.read_text(encoding="utf-8")
    
    # These are OK in technical documentation as they're generic placeholders
    # showing the *type* of path, not actual discovered sources
    allowed_patterns = [
        "$LOCALAPPDATA/hermes/state.db",
        "$USERPROFILE/.claude",
        "$LOCALAPPDATA/opencode",
    ]
    
    # Just verify the file can be read - we're documenting that these
    # generic placeholders in technical docs are acceptable
    assert content  # File is readable


def test_sensitive_count_detector_works():
    """
    Verify that the sensitive count detector would catch violations.
    This test validates the test itself by checking a known-bad fixture.
    """
    # Fixture with sensitive counts that should be detected
    sensitive_report = """
    ## Unauthorized Scan Results
    
    The system detected:
    - **1 个 Hermes 数据库**: 245 sessions, 55,896 messages
    - **627 个 Claude JSONL 文件**: containing project history
    
    This reveals actual inventory scale.
    """
    
    # Apply the same detection logic
    sensitive_patterns = [
        (r'\b\d+\s+sessions?\b', 'session count'),
        (r'\b\d[\d,]+\s+messages?\b', 'message count'),
        (r'\d+\s+个\s+(?:Claude|Hermes|Codex|OpenCode)\s+\w+\s+文件', 'Chinese file count with agent name'),
        (r'\d+\s+个\s+\w+\s+数据库', 'Chinese database count'),
    ]
    
    violations = []
    for pattern, description in sensitive_patterns:
        matches = re.findall(pattern, sensitive_report, re.IGNORECASE)
        if matches:
            real_violations = []
            for match in matches:
                num_str = re.search(r'\d[\d,]*', match)
                if num_str:
                    num = int(num_str.group().replace(',', ''))
                    if (('session' in match.lower() or '文件' in match or '数据库' in match) and num > 50) or \
                       ('message' in match.lower() and num > 1000):
                        real_violations.append(match)
            
            if real_violations:
                violations.append(f"{description}: {real_violations}")
    
    # This test should find violations in the fixture
    assert violations, (
        "Sensitive count detector failed to catch known violations in test fixture. "
        "The detector may be broken or too permissive."
    )
    
    # Verify specific violations are caught
    assert any('245 sessions' in str(v) for v in violations), "Should detect '245 sessions'"
    assert any('55,896 messages' in str(v) or '55896 messages' in str(v) for v in violations), "Should detect '55,896 messages'"
    assert any('627' in str(v) and 'Claude' in str(v) for v in violations), "Should detect '627 个 Claude ... 文件'"


def test_benign_counts_not_flagged():
    """
    Verify that legitimate small counts in test/protocol contexts are not flagged.
    """
    benign_report = """
    ## Test Results
    
    Protocol validation:
    - Tested with 3 sessions
    - Generated 200 test messages
    - Character limit: 500 tokens
    - Code coverage: 2,450 lines
    
    These are test parameters, not inventory disclosures.
    """
    
    # Apply the same detection logic
    sensitive_patterns = [
        (r'\b\d+\s+sessions?\b', 'session count'),
        (r'\b\d[\d,]+\s+messages?\b', 'message count'),
    ]
    
    violations = []
    for pattern, description in sensitive_patterns:
        matches = re.findall(pattern, benign_report, re.IGNORECASE)
        if matches:
            real_violations = []
            for match in matches:
                num_str = re.search(r'\d[\d,]*', match)
                if num_str:
                    num = int(num_str.group().replace(',', ''))
                    if (('session' in match.lower() and num > 50)) or \
                       ('message' in match.lower() and num > 1000):
                        real_violations.append(match)
            
            if real_violations:
                violations.append(f"{description}: {real_violations}")
    
    # Benign counts should NOT trigger violations
    assert not violations, (
        f"Detector incorrectly flagged benign test counts: {violations}. "
        "The threshold may be too low or the filter too aggressive."
    )
