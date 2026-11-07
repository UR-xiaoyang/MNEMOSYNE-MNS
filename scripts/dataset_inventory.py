#!/usr/bin/env python3
"""
Dataset source inventory - read-only cataloging without copying private data.

Discovers available conversation sources from environment variables and creates
a manifest with metadata only (no content copying).

Supported sources:
- HERMES_STATE_DB or LOCALAPPDATA/hermes/state.db
- USERPROFILE/.claude (if CLAUDE_HISTORY_DIR set)
- USERPROFILE/.codex/sessions (if CODEX_SESSIONS_DIR set)
- LOCALAPPDATA/opencode (if OPENCODE_DATA_DIR set)
"""
import argparse
import json
import os
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


def discover_sources(env_overrides: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Discover available conversation sources from EXPLICIT environment variables only.
    
    Args:
        env_overrides: Optional dict to override environment variables (for testing)
    
    Returns:
        List of source metadata dicts (no content, only counts and opaque identifiers)
        
    Privacy boundary: NO implicit fallback to LOCALAPPDATA, USERPROFILE, or HOME.
    Only processes explicitly authorized sources via environment variables.
    """
    env = env_overrides if env_overrides is not None else os.environ
    sources = []
    
    # Hermes source - ONLY if explicitly set
    hermes_db = env.get("HERMES_STATE_DB")
    if hermes_db and Path(hermes_db).exists():
        sources.append(_inspect_hermes_db(hermes_db))
    
    # Claude source - ONLY if explicitly set
    claude_dir = env.get("CLAUDE_HISTORY_DIR")
    if claude_dir and Path(claude_dir).exists():
        sources.extend(_inspect_claude_dir(claude_dir))
    
    # Codex source - ONLY if explicitly set
    codex_dir = env.get("CODEX_SESSIONS_DIR")
    if codex_dir and Path(codex_dir).exists():
        sources.extend(_inspect_codex_dir(codex_dir))
    
    # OpenCode source - ONLY if explicitly set
    opencode_dir = env.get("OPENCODE_DATA_DIR")
    if opencode_dir and Path(opencode_dir).exists():
        sources.extend(_inspect_opencode_dir(opencode_dir))
    
    return sources


def _make_source_token(path: str) -> str:
    """
    Create a stable, opaque identifier for a source path.
    
    Returns a non-reversible hash token instead of exposing absolute paths.
    """
    return hashlib.sha256(path.encode('utf-8')).hexdigest()[:16]


def _inspect_hermes_db(db_path: str) -> Dict[str, Any]:
    """Inspect Hermes SQLite database (read-only)."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        # Get schema summary
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        return {
            "source": "hermes",
            "source_token": _make_source_token(db_path),
            "format": "sqlite",
            "size_bytes": Path(db_path).stat().st_size,
            "session_count": session_count,
            "message_count": message_count,
            "schema": {"tables": tables}
        }
    finally:
        conn.close()


def _inspect_claude_dir(claude_dir: str) -> List[Dict[str, Any]]:
    """Inspect Claude JSONL files."""
    sources = []
    claude_path = Path(claude_dir)
    
    # Look for history JSONL files
    for jsonl_file in claude_path.glob("**/*.jsonl"):
        if jsonl_file.stat().st_size == 0:
            continue
        
        # Count lines without loading content
        with open(jsonl_file, 'rb') as f:
            line_count = sum(1 for _ in f)
        
        sources.append({
            "source": "claude",
            "source_token": _make_source_token(str(jsonl_file)),
            "format": "jsonl",
            "size_bytes": jsonl_file.stat().st_size,
            "record_count": line_count,
            "schema": {"format": "claude_conversation"}
        })
    
    return sources


def _inspect_codex_dir(codex_dir: str) -> List[Dict[str, Any]]:
    """Inspect Codex session files."""
    sources = []
    codex_path = Path(codex_dir)
    
    # Look for session JSONL or SQLite files
    for session_file in codex_path.glob("**/*.jsonl"):
        if session_file.stat().st_size == 0:
            continue
        
        with open(session_file, 'rb') as f:
            line_count = sum(1 for _ in f)
        
        sources.append({
            "source": "codex",
            "source_token": _make_source_token(str(session_file)),
            "format": "jsonl",
            "size_bytes": session_file.stat().st_size,
            "record_count": line_count,
            "schema": {"format": "codex_session"}
        })
    
    return sources


def _inspect_opencode_dir(opencode_dir: str) -> List[Dict[str, Any]]:
    """Inspect OpenCode SQLite databases."""
    sources = []
    opencode_path = Path(opencode_dir)
    
    for db_file in opencode_path.glob("**/*.db"):
        if db_file.stat().st_size == 0:
            continue
        
        uri = f"file:{db_file.as_posix()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            cursor = conn.cursor()
            
            # Check if it's an OpenCode database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if "messages" in tables:
                cursor.execute("SELECT COUNT(*) FROM messages")
                message_count = cursor.fetchone()[0]
                
                sources.append({
                    "source": "opencode",
                    "source_token": _make_source_token(str(db_file)),
                    "format": "sqlite",
                    "size_bytes": db_file.stat().st_size,
                    "message_count": message_count,
                    "schema": {"tables": tables}
                })
            
            conn.close()
        except sqlite3.Error:
            # Skip files that aren't valid SQLite databases
            continue
    
    return sources


def run_inventory(output_path: Path, env_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Run inventory and write results.
    
    Args:
        output_path: Where to write inventory JSON
        env_overrides: Optional environment overrides for testing
    
    Returns:
        Inventory summary dict
    """
    sources = discover_sources(env_overrides)
    
    inventory = {
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_count": len(sources),
        "sources": sources
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    
    return inventory


def main():
    parser = argparse.ArgumentParser(
        description="Inventory conversation sources without copying private data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for inventory JSON"
    )
    
    args = parser.parse_args()
    
    inventory = run_inventory(args.output)
    
    print(f"✓ Inventory complete: {inventory['source_count']} sources found")
    for source in inventory['sources']:
        # Only show source category and record counts - NO paths
        if source['source'] == 'hermes':
            print(f"  - {source['source']}: {source['session_count']} sessions, {source['message_count']} messages")
        elif 'message_count' in source:
            print(f"  - {source['source']}: {source['message_count']} messages")
        elif 'record_count' in source:
            print(f"  - {source['source']}: {source['record_count']} records")
        else:
            print(f"  - {source['source']}: {source.get('size_bytes', 0)} bytes")
    
    if inventory['source_count'] == 0:
        print("\n⚠ No sources found. Explicit authorization required via environment variables:")
        print("  - HERMES_STATE_DB=<path>")
        print("  - CLAUDE_HISTORY_DIR=<path>")
        print("  - CODEX_SESSIONS_DIR=<path>")
        print("  - OPENCODE_DATA_DIR=<path>")
        print("\nDo NOT use default home directory locations without explicit authorization.")


if __name__ == "__main__":
    main()
