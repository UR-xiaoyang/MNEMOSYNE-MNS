"""
Test dataset-card documentation contract enforcement.

Validates that docs/dataset-card-v1.md accurately reflects the explicit-authorization
privacy contract: no implicit home-directory fallback, mandatory token→path mapping,
and privacy-safe output.
"""

import re
from pathlib import Path


def test_dataset_card_no_default_fallback():
    """Dataset card must NOT describe default home-directory fallback locations."""
    card_path = Path("docs/dataset-card-v1.md")
    assert card_path.exists(), f"Dataset card not found: {card_path}"
    
    content = card_path.read_text(encoding="utf-8")
    
    # Must NOT contain "default: $LOCALAPPDATA" or "default: $USERPROFILE" patterns
    forbidden_patterns = [
        r"default:\s*\$LOCALAPPDATA",
        r"default:\s*\$USERPROFILE",
        r"default:\s*\$HOME",
        r"defaults?\s+to\s+\$LOCALAPPDATA",
        r"defaults?\s+to\s+\$USERPROFILE",
        r"defaults?\s+to\s+\$HOME",
    ]
    
    violations = []
    for pattern in forbidden_patterns:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        if matches:
            for match in matches:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                violations.append(f"Line {line_num}: {match.group()}")
    
    assert not violations, (
        f"Dataset card contains forbidden default home-directory fallback:\n"
        + "\n".join(violations)
    )


def test_dataset_card_explicit_authorization_only():
    """Dataset card must describe explicit authorization via environment variables."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Must mention explicit authorization environment variables
    required_env_vars = [
        "HERMES_STATE_DB",
        "CLAUDE_HISTORY_DIR",
        "CODEX_SESSIONS_DIR",
        "OPENCODE_DATA_DIR",
    ]
    
    for env_var in required_env_vars:
        assert env_var in content, (
            f"Dataset card must mention explicit authorization variable: {env_var}"
        )


def test_dataset_card_no_fallback_statement():
    """Dataset card must state that unset variables result in empty inventory."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Must contain statement about no fallback behavior
    no_fallback_indicators = [
        "no variables are set",
        "empty inventory",
        "no default",
        "no fallback",
        "not set",
    ]
    
    found = any(indicator in content.lower() for indicator in no_fallback_indicators)
    assert found, (
        "Dataset card must state that unset variables result in empty inventory "
        "with no fallback to home directories"
    )


def test_dataset_card_token_mapping_in_normalize():
    """Dataset card normalize examples must include --source <token>=<path> argument."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Find normalize_conversations.py command examples
    normalize_pattern = r"python\s+scripts/normalize_conversations\.py.*?```"
    matches = list(re.finditer(normalize_pattern, content, re.DOTALL))
    
    assert matches, "Dataset card must contain normalize_conversations.py examples"
    
    # Each example must include --source <token>=<path>
    for match in matches:
        example = match.group()
        assert "--source" in example, (
            f"normalize_conversations.py example must include --source argument:\n{example}"
        )


def test_dataset_card_safety_guarantees_no_paths():
    """Safety Guarantees section must describe token/safe identifiers, not paths."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Find Safety Guarantees section
    safety_section_match = re.search(
        r"##\s*Safety Guarantees.*?(?=##|\Z)",
        content,
        re.DOTALL | re.IGNORECASE
    )
    
    if safety_section_match:
        safety_section = safety_section_match.group()
        
        # Should mention tokens or safe identifiers in inventory context
        has_token_mention = "token" in safety_section.lower() or "privacy-safe" in safety_section.lower()
        
        # Should NOT say "inventory metadata (paths" without negation
        # But "never...paths" or "not...paths" is acceptable
        problematic_phrase = re.search(
            r"inventory.*metadata.*\([^)]*\bpaths\b[^)]*\)(?!.*never)(?!.*not)",
            safety_section,
            re.IGNORECASE
        )
        
        # Check if the mention is negative (e.g., "never message content or absolute paths")
        if "paths" in safety_section.lower():
            # It's OK if paths are mentioned in a negative context
            negative_context = re.search(
                r"(never|not|no).*paths",
                safety_section,
                re.IGNORECASE
            )
            assert negative_context or has_token_mention, (
                "If 'paths' appears in Safety Guarantees, it must be in negative context "
                "(never/not/no) or paired with privacy-safe tokens"
            )
        
        assert has_token_mention, (
            "Safety Guarantees should mention tokens or privacy-safe identifiers"
        )


def test_dataset_card_allows_legitimate_env_var_names():
    """Test must not falsely reject legitimate environment variable names."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # These are legitimate and must be present
    legitimate_vars = ["HERMES_STATE_DB", "CLAUDE_HISTORY_DIR"]
    
    for var in legitimate_vars:
        assert var in content, f"Legitimate environment variable must be present: {var}"


def test_dataset_card_missing_mapping_rejection():
    """Dataset card must document that normalize rejects missing token mappings."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Should mention rejection or requirement of token mapping
    rejection_indicators = [
        "reject",
        "require",
        "mandatory",
        "must provide",
        "missing mapping",
    ]
    
    # And should be in context of source or token
    found_token_context = "token" in content.lower() or "mapping" in content.lower()
    found_rejection = any(indicator in content.lower() for indicator in rejection_indicators)
    
    assert found_token_context and found_rejection, (
        "Dataset card must document that normalize requires/rejects missing token mappings"
    )


def test_normalized_schema_uses_source_file_not_source_token():
    """Normalized JSONL schema must use 'source_file' field, not 'source_token'."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Find the schema/traceability field list in Step 2
    # Line 66 context: "Preserves traceability: `id, source, source_token, session_id...`"
    schema_pattern = r"Preserves traceability:.*?`[^`]+`"
    matches = list(re.finditer(schema_pattern, content, re.DOTALL))
    
    assert matches, "Dataset card Step 2 must document traceability schema"
    
    for match in matches:
        schema_text = match.group()
        # The actual output field is 'source_file', not 'source_token'
        assert "source_file" in schema_text, (
            f"Traceability schema must use 'source_file' (the actual JSONL field name), "
            f"not 'source_token':\n{schema_text}"
        )
        
        # Should NOT list 'source_token' as a JSONL output field
        # (it's used in inventory and as a CLI argument, but not in normalized output)
        if "source_token" in schema_text:
            # Check if it's clearly marked as NOT being in the output
            # For now, we'll be strict: the field list should use source_file
            assert False, (
                f"Traceability schema incorrectly lists 'source_token' as output field. "
                f"The JSONL output uses 'source_file' to hold the privacy-safe token:\n{schema_text}"
            )


def test_normalize_example_is_valid_posix_command():
    """Normalize command examples must be valid POSIX/bash with proper line continuation."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Find normalize_conversations.py command examples
    # Looking for bash blocks that contain the normalize command
    normalize_pattern = r"```bash\n(.*?python\s+scripts/normalize_conversations\.py.*?)```"
    matches = list(re.finditer(normalize_pattern, content, re.DOTALL))
    
    assert matches, "Dataset card must contain normalize bash examples"
    
    for match in matches:
        example = match.group(1).strip()
        
        # Extract just the python command (skip comment lines)
        python_lines = []
        for line in example.split('\n'):
            if line.strip().startswith('#') or not line.strip():
                continue
            python_lines.append(line)
        
        python_command = '\n'.join(python_lines)
        
        # Check for excessive escaping like \\\\\\ (more than 2 backslashes in a row before newline)
        excessive_escape = re.search(r"\\{3,}\s*$", python_command, re.MULTILINE)
        assert not excessive_escape, (
            f"Normalize example contains excessive backslash escaping (should be single \\ for line continuation):\n"
            f"{python_command}\n"
            f"Found at: {excessive_escape.group() if excessive_escape else 'N/A'}"
        )
        
        # Valid line continuation: single backslash at end of line (may have trailing whitespace)
        # Invalid: multiple backslashes or backslashes not at line end
        lines = python_command.split('\n')
        for i, line in enumerate(lines[:-1], 1):  # All but last line
            if '\\' in line:
                # Should end with exactly one backslash (possibly followed by whitespace)
                if not re.search(r'\\\s*$', line):
                    assert False, (
                        f"Line {i} has backslash not at end (invalid continuation):\n{line}"
                    )
                # Should NOT have multiple consecutive backslashes at end
                if re.search(r'\\\\+\s*$', line):
                    assert False, (
                        f"Line {i} has multiple backslashes at end (should be single \\):\n{line}"
                    )


def test_normalize_example_matches_argparse_contract():
    """Normalize example arguments must match actual argparse schema."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Extract first normalize example
    normalize_pattern = r"```bash\n(.*?python\s+scripts/normalize_conversations\.py.*?)```"
    matches = list(re.finditer(normalize_pattern, content, re.DOTALL))
    
    assert matches, "Dataset card must contain normalize bash examples"
    
    example = matches[0].group(1).strip()
    
    # Must contain --inventory
    assert "--inventory" in example, "Normalize example must include --inventory argument"
    
    # Must contain --output
    assert "--output" in example, "Normalize example must include --output argument"
    
    # Must contain at least one --source (repeatable)
    source_count = len(re.findall(r"--source\s+\S+", example))
    assert source_count >= 1, (
        f"Normalize example must include at least one --source argument (found {source_count})"
    )
    
    # --source format should be <token>=<path>
    source_args = re.findall(r"--source\s+(\S+)", example)
    for source_arg in source_args:
        assert "=" in source_arg, (
            f"--source argument must use <token>=<path> format, got: {source_arg}"
        )


def test_normalize_example_demonstrates_multiple_sources():
    """Normalize example should show that --source is repeatable for multiple sources."""
    card_path = Path("docs/dataset-card-v1.md")
    content = card_path.read_text(encoding="utf-8")
    
    # Extract normalize examples
    normalize_pattern = r"```bash\n(.*?python\s+scripts/normalize_conversations\.py.*?)```"
    matches = list(re.finditer(normalize_pattern, content, re.DOTALL))
    
    # At least one example should show multiple --source arguments
    found_multi_source = False
    for match in matches:
        example = match.group(1).strip()
        source_count = len(re.findall(r"--source\s+", example))
        if source_count >= 2:
            found_multi_source = True
            break
    
    assert found_multi_source, (
        "At least one normalize example should demonstrate multiple --source arguments "
        "to show repeatability"
    )
