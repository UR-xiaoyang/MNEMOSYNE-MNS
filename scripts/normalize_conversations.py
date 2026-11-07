#!/usr/bin/env python3
"""
Normalize conversations from various sources to standardized JSONL format.

Extracts only user-visible messages, excludes system/developer/hidden content,
and creates stable IDs for traceability without copying sensitive data.
"""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional


def should_include_message(role: str, content: str) -> bool:
    """
    Determine if a message should be included in normalized output.
    
    Excludes:
    - system/developer messages
    - hidden reasoning
    - framework injections
    - skill text
    - agent/IDE events
    
    Args:
        role: Message role (user, assistant, system, developer, tool)
        content: Message content
    
    Returns:
        True if message should be included
    """
    # Exclude system and developer messages
    if role in ("system", "developer"):
        return False
    
    # Exclude framework/skill/agent injections
    if is_framework_injection(content):
        return False
    
    # Include user, assistant, and tool messages
    return True


def is_framework_injection(content: str) -> bool:
    """
    Detect framework/skill/agent injected content.
    
    Uses precise patterns to avoid false positives on ordinary angle brackets
    in mathematical expressions or natural language.
    
    Args:
        content: Message content to check
    
    Returns:
        True if content appears to be framework injection
    """
    # Precise injection patterns - avoid bare '<' to prevent math false positives
    injection_patterns = [
        r'\[System note:',           # [System note: ...]
        r'# Skill:',                 # # Skill: skill-name
        r'<!DOCTYPE',                # <!DOCTYPE html> or <!DOCTYPE HTML PUBLIC
        r'\[IMPORTANT:',             # [IMPORTANT: ...]
        r'You are a .* assistant',   # You are a helpful assistant
        r'You are an .* assistant',  # You are an AI assistant
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    
    return False


def generate_message_id(
    source: str,
    source_file: str,
    session_id: str,
    ordinal: int,
    role: str,
    content: str,
    timestamp: int
) -> str:
    """
    Generate stable, deterministic message ID.
    
    Args:
        source: Source name (hermes, claude, etc.)
        source_file: Source file path
        session_id: Session identifier
        ordinal: Message order in session
        role: Message role
        content: Message content
        timestamp: Unix timestamp
    
    Returns:
        Stable hash-based ID
    """
    # Create deterministic ID from key fields
    id_input = f"{source}::{source_file}::{session_id}::{ordinal}::{role}::{timestamp}::{content[:100]}"
    hash_obj = hashlib.sha256(id_input.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def truncate_if_needed(content: str, max_chars: int = 10000) -> Dict[str, Any]:
    """
    Truncate oversized content with auditable marker.
    
    Args:
        content: Message content
        max_chars: Maximum characters to keep
    
    Returns:
        Dict with content, truncated flag, and original_length
    """
    if len(content) <= max_chars:
        return {
            "content": content,
            "truncated": False
        }
    
    # Keep head and tail with truncation marker
    head_size = max_chars // 2
    tail_size = max_chars - head_size
    
    truncated_content = (
        content[:head_size] +
        f"\n\n[TRUNCATED: original length {len(content)} chars, showing head+tail]\n\n" +
        content[-tail_size:]
    )
    
    return {
        "content": truncated_content,
        "truncated": True,
        "original_length": len(content)
    }


def normalize_hermes_db(db_path: str, output_path: Path, *, source_token: str) -> int:
    """
    Normalize Hermes SQLite database to standard JSONL.
    
    Args:
        db_path: Path to Hermes state.db
        output_path: Output JSONL path
        source_token: Required opaque token to use instead of path in output (keyword-only)
    
    Returns:
        Number of messages normalized
    """
    if not source_token:
        raise ValueError("source_token is required and must be non-empty")
    # Open database in read-only mode
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        # Query messages with session info
        cursor.execute("""
            SELECT 
                m.id,
                m.session_id,
                m.role,
                m.content,
                m.created_at,
                ROW_NUMBER() OVER (PARTITION BY m.session_id ORDER BY m.created_at, m.id) as ordinal
            FROM messages m
            WHERE m.role NOT IN ('system', 'developer')
            ORDER BY m.session_id, m.created_at, m.id
        """)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in cursor:
                role = row['role']
                content = row['content'] or ""
                
                # Skip if not includable
                if not should_include_message(role, content):
                    continue
                
                # Truncate if needed
                truncation_result = truncate_if_needed(content)
                
                # Generate stable ID (use token for ID generation, not absolute path)
                msg_id = generate_message_id(
                    source="hermes",
                    source_file=source_token,
                    session_id=row['session_id'],
                    ordinal=row['ordinal'],
                    role=role,
                    content=content,
                    timestamp=row['created_at']
                )
                
                # Create normalized message (use token in output, not absolute path)
                normalized = {
                    "id": msg_id,
                    "source": "hermes",
                    "source_file": source_token,
                    "session_id": row['session_id'],
                    "ordinal": row['ordinal'],
                    "timestamp": row['created_at'],
                    "role": role,
                    "content": truncation_result["content"],
                    "metadata": {},
                    "redacted": False,
                    "truncated": truncation_result["truncated"]
                }
                
                if truncation_result["truncated"]:
                    normalized["metadata"]["original_length"] = truncation_result["original_length"]
                
                f.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                count += 1
        
        return count
    
    finally:
        conn.close()


def normalize_jsonl_file(
    source: str,
    source_file: Path,
    output_path: Path,
    role_field: str = "role",
    content_field: str = "content",
    *,
    source_token: str
) -> int:
    """
    Normalize JSONL file (Claude/Codex format).
    
    Args:
        source: Source name (claude, codex)
        source_file: Input JSONL path
        output_path: Output JSONL path
        role_field: Field name for role
        content_field: Field name for content
        source_token: Required opaque token to use instead of path in output (keyword-only)
    
    Returns:
        Number of messages normalized
    """
    if not source_token:
        raise ValueError("source_token is required and must be non-empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(source_file, 'r', encoding='utf-8') as inf:
        with open(output_path, 'w', encoding='utf-8') as outf:
            ordinal = 0
            current_session = None
            
            for line_num, line in enumerate(inf, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                role = record.get(role_field, "unknown")
                content = record.get(content_field, "")
                
                # Skip if not includable
                if not should_include_message(role, content):
                    continue
                
                # Try to extract session ID
                session_id = record.get("session_id") or record.get("conversation_id") or f"file_{source_file.stem}"
                
                # Track ordinal per session
                if session_id != current_session:
                    current_session = session_id
                    ordinal = 0
                ordinal += 1
                
                # Truncate if needed
                truncation_result = truncate_if_needed(content)
                
                # Generate stable ID (use token for ID generation, not absolute path)
                timestamp = record.get("timestamp") or record.get("created_at") or line_num
                msg_id = generate_message_id(
                    source=source,
                    source_file=source_token,
                    session_id=session_id,
                    ordinal=ordinal,
                    role=role,
                    content=content,
                    timestamp=timestamp
                )
                
                # Create normalized message (use token in output, not absolute path)
                normalized = {
                    "id": msg_id,
                    "source": source,
                    "source_file": source_token,
                    "session_id": session_id,
                    "ordinal": ordinal,
                    "timestamp": timestamp,
                    "role": role,
                    "content": truncation_result["content"],
                    "metadata": {},
                    "redacted": False,
                    "truncated": truncation_result["truncated"]
                }
                
                if truncation_result["truncated"]:
                    normalized["metadata"]["original_length"] = truncation_result["original_length"]
                
                outf.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Normalize conversations to standard JSONL schema"
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="Inventory JSON from dataset_inventory.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path"
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="source_mappings",
        help="Explicit source token to path mapping: <token>=<path> (repeatable)"
    )
    
    args = parser.parse_args()
    
    # Parse source mappings
    source_map = {}
    if args.source_mappings:
        for mapping in args.source_mappings:
            if "=" not in mapping:
                raise ValueError(f"Invalid source mapping format: {mapping}. Expected <token>=<path>")
            token, path = mapping.split("=", 1)
            source_map[token] = path
    
    # Load inventory
    with open(args.inventory) as f:
        inventory = json.load(f)
    
    total_count = 0
    
    # Process each source
    for source in inventory["sources"]:
        source_type = source["source"]
        source_token = source.get("source_token")
        
        # Resolve actual path from token mapping
        if source_token:
            if source_token not in source_map:
                raise ValueError(
                    f"Source token '{source_token}' has no path mapping. "
                    f"Provide explicit mapping with: --source {source_token}=<path>"
                )
            actual_path = source_map[source_token]
        else:
            # Fallback for legacy inventories that still have source_file
            actual_path = source.get("source_file")
            if not actual_path:
                raise ValueError(f"Source has neither 'source_token' nor 'source_file': {source}")
        
        # Print only token/type, not absolute path
        if source_token:
            print(f"Processing {source_type}: {source_token}")
        else:
            print(f"Processing {source_type}")
        
        if source_type == "hermes" and source["format"] == "sqlite":
            count = normalize_hermes_db(actual_path, args.output, source_token=source_token)
            total_count += count
            print(f"  ✓ Normalized {count} messages")
        
        elif source["format"] == "jsonl":
            count = normalize_jsonl_file(
                source=source_type,
                source_file=Path(actual_path),
                output_path=args.output,
                source_token=source_token
            )
            total_count += count
            print(f"  ✓ Normalized {count} messages")
    
    print(f"\n✓ Total: {total_count} messages normalized")
    print("✓ Output written")


if __name__ == "__main__":
    main()
