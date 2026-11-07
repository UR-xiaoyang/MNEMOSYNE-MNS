#!/usr/bin/env python3
"""
Build session-isolated train/validation/test splits.

Ensures no session appears in multiple splits, preserves temporal ordering,
and creates reproducible split assignments with manifest hash.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple


def collect_sessions(input_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect all messages grouped by session.
    
    Args:
        input_path: Input JSONL with normalized messages
    
    Returns:
        Dict mapping session_id to list of messages
    """
    sessions = defaultdict(list)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            message = json.loads(line)
            session_id = message["session_id"]
            sessions[session_id].append(message)
    
    return dict(sessions)


def sort_sessions_by_time(sessions: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, int]]:
    """
    Sort sessions by their earliest message timestamp.
    
    Args:
        sessions: Dict of session_id -> messages
    
    Returns:
        List of (session_id, earliest_timestamp) sorted by timestamp
    """
    session_times = []
    
    for session_id, messages in sessions.items():
        # Find earliest timestamp in session
        earliest = min(msg["timestamp"] for msg in messages)
        session_times.append((session_id, earliest))
    
    # Sort by timestamp
    session_times.sort(key=lambda x: x[1])
    
    return session_times


def assign_splits(
    session_times: List[Tuple[str, int]],
    split_ratio: Tuple[float, float, float],
    seed: int
) -> Dict[str, str]:
    """
    Assign sessions to train/validation/test splits.
    
    Args:
        session_times: List of (session_id, timestamp) sorted chronologically
        split_ratio: (train, val, test) ratios that sum to 1.0
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping session_id to split name
    """
    train_ratio, val_ratio, test_ratio = split_ratio
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    
    total_sessions = len(session_times)
    
    # Calculate split sizes
    train_size = int(total_sessions * train_ratio)
    val_size = int(total_sessions * val_ratio)
    test_size = total_sessions - train_size - val_size  # Remainder goes to test
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    session_ids = [sess_id for sess_id, _ in session_times]
    random.shuffle(session_ids)
    
    # Assign to splits
    session_mapping = {}
    
    for i, session_id in enumerate(session_ids):
        if i < train_size:
            session_mapping[session_id] = "train"
        elif i < train_size + val_size:
            session_mapping[session_id] = "validation"
        else:
            session_mapping[session_id] = "test"
    
    return session_mapping


def compute_split_hash(session_mapping: Dict[str, str], seed: int) -> str:
    """
    Compute deterministic hash of split assignment.
    
    Args:
        session_mapping: Session to split assignments
        seed: Random seed used
    
    Returns:
        SHA-256 hash hex string
    """
    # Sort for deterministic ordering
    sorted_items = sorted(session_mapping.items())
    hash_input = f"{seed}::{json.dumps(sorted_items)}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]


def write_splits(
    sessions: Dict[str, List[Dict[str, Any]]],
    session_mapping: Dict[str, str],
    output_dir: Path
) -> Dict[str, int]:
    """
    Write split files.
    
    Args:
        sessions: Dict of session_id -> messages
        session_mapping: Session to split assignments
        output_dir: Output directory
    
    Returns:
        Dict with message counts per split
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open all split files
    split_files = {
        "train": open(output_dir / "train.jsonl", 'w', encoding='utf-8'),
        "validation": open(output_dir / "validation.jsonl", 'w', encoding='utf-8'),
        "test": open(output_dir / "test.jsonl", 'w', encoding='utf-8')
    }
    
    split_counts = {"train": 0, "validation": 0, "test": 0}
    
    try:
        # Write each session to its assigned split
        for session_id, messages in sessions.items():
            split_name = session_mapping[session_id]
            
            # Sort messages by ordinal within session
            sorted_messages = sorted(messages, key=lambda m: m["ordinal"])
            
            for message in sorted_messages:
                split_files[split_name].write(
                    json.dumps(message, ensure_ascii=False) + '\n'
                )
                split_counts[split_name] += 1
    
    finally:
        # Close all files
        for f in split_files.values():
            f.close()
    
    return split_counts


def build_splits(
    input_path: Path,
    output_dir: Path,
    seed: int = 42,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict[str, Any]:
    """
    Build session-isolated splits with manifest.
    
    Args:
        input_path: Input JSONL with normalized messages
        output_dir: Output directory for splits
        seed: Random seed for reproducibility
        split_ratio: (train, val, test) ratios
    
    Returns:
        Summary dict with counts and manifest info
    """
    # Collect sessions
    sessions = collect_sessions(input_path)
    
    # Sort by time
    session_times = sort_sessions_by_time(sessions)
    
    # Assign to splits
    session_mapping = assign_splits(session_times, split_ratio, seed)
    
    # Compute hash
    split_hash = compute_split_hash(session_mapping, seed)
    
    # Write splits
    split_counts = write_splits(sessions, session_mapping, output_dir)
    
    # Create manifest
    manifest = {
        "version": "1.0.0",
        "seed": seed,
        "split_ratio": {
            "train": split_ratio[0],
            "validation": split_ratio[1],
            "test": split_ratio[2]
        },
        "total_sessions": len(sessions),
        "total_messages": sum(len(msgs) for msgs in sessions.values()),
        "split_hash": split_hash,
        "session_mapping": session_mapping,
        "split_counts": {
            "train": {
                "sessions": sum(1 for s in session_mapping.values() if s == "train"),
                "messages": split_counts["train"]
            },
            "validation": {
                "sessions": sum(1 for s in session_mapping.values() if s == "validation"),
                "messages": split_counts["validation"]
            },
            "test": {
                "sessions": sum(1 for s in session_mapping.values() if s == "test"),
                "messages": split_counts["test"]
            }
        }
    }
    
    # Write manifest
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest


def verify_split_isolation(output_dir: Path) -> Dict[str, Any]:
    """
    Verify no session appears in multiple splits.
    
    Args:
        output_dir: Directory with split files
    
    Returns:
        Verification report
    """
    session_splits = {}
    violations = []
    
    for split_name in ["train", "validation", "test"]:
        split_file = output_dir / f"{split_name}.jsonl"
        if not split_file.exists():
            continue
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                message = json.loads(line)
                session_id = message["session_id"]
                
                if session_id in session_splits:
                    if session_splits[session_id] != split_name:
                        violations.append({
                            "session_id": session_id,
                            "splits": [session_splits[session_id], split_name],
                            "line": line_num
                        })
                else:
                    session_splits[session_id] = split_name
    
    return {
        "passed": len(violations) == 0,
        "total_sessions": len(session_splits),
        "violations": violations
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build session-isolated train/validation/test splits"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input cleaned JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing splits"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        print("Verifying split isolation...")
        report = verify_split_isolation(args.output_dir)
        
        print(f"\n✓ Verification Results:")
        print(f"  Total sessions: {report['total_sessions']}")
        print(f"  Violations: {len(report['violations'])}")
        
        if not report['passed']:
            print("\n✗ ISOLATION VIOLATED:")
            for v in report['violations'][:5]:
                print(f"  - Session {v['session_id']} in {v['splits']}")
            exit(1)
        else:
            print("\n✓ Split isolation verified")
    
    else:
        split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)
        
        print(f"Building splits from {args.input}...")
        print(f"  Ratio: {split_ratio}")
        print(f"  Seed: {args.seed}")
        
        manifest = build_splits(
            args.input,
            args.output_dir,
            seed=args.seed,
            split_ratio=split_ratio
        )
        
        print(f"\n✓ Splits created:")
        print(f"  Total sessions: {manifest['total_sessions']}")
        print(f"  Total messages: {manifest['total_messages']}")
        print(f"  Split hash: {manifest['split_hash']}")
        
        for split_name in ["train", "validation", "test"]:
            counts = manifest['split_counts'][split_name]
            print(f"  {split_name}: {counts['sessions']} sessions, {counts['messages']} messages")
        
        print(f"\n✓ Manifest: {args.output_dir / 'split_manifest.json'}")
        
        # Verify
        print("\nVerifying split isolation...")
        report = verify_split_isolation(args.output_dir)
        
        if not report['passed']:
            print("✗ WARNING: Split isolation check failed!")
            exit(1)
        else:
            print("✓ Split isolation verified")


if __name__ == "__main__":
    main()
