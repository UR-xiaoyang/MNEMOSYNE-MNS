#!/usr/bin/env python3
"""
Redact sensitive data and validate dataset.

Removes API keys, tokens, emails, IPs, connection strings, and home paths
while maintaining dataset integrity and audit trail.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List


# Sensitive data patterns
REDACTION_PATTERNS = {
    "api_key": [
        r'sk-[a-zA-Z0-9]{32,}',  # OpenAI-style keys
        r'api[_-]?key["\s:=]+[a-zA-Z0-9_-]{20,}',
        r'AKIA[0-9A-Z]{16}',  # AWS keys
    ],
    "bearer_token": [
        r'Bearer\s+[a-zA-Z0-9_\-\.=]+',
        r'token["\s:=]+[a-zA-Z0-9_\-\.]{20,}',
    ],
    "private_key": [
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'private[_-]?key["\s:=]+[a-zA-Z0-9+/=]{40,}',
    ],
    "connection_string": [
        r'(?:postgresql|mysql|mongodb|redis)://[^:\s]+:[^@\s]+@[^\s]+',
        r'(?:Server|Host)=[^;]+;(?:User|UID)=[^;]+;(?:Password|PWD)=[^;]+',
    ],
    "email": [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    ],
    "ip_address": [
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',  # IPv6
    ],
    "home_path": [
        r'C:\\Users\\[^\\]+',
        r'/home/[^/\s]+',
        r'/Users/[^/\s]+',
    ],
}


def redact_sensitive_data(content: str) -> Dict[str, Any]:
    """
    Redact sensitive information from content.
    
    Args:
        content: Text to redact
    
    Returns:
        Dict with redacted content, redaction flag, and types
    """
    redacted_content = content
    redacted = False
    redaction_types = []
    
    for redaction_type, patterns in REDACTION_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, redacted_content, re.IGNORECASE)
            if matches:
                redacted = True
                if redaction_type not in redaction_types:
                    redaction_types.append(redaction_type)
                
                # Replace with type-specific placeholder
                placeholder = f"[REDACTED_{redaction_type.upper()}]"
                redacted_content = re.sub(pattern, placeholder, redacted_content, flags=re.IGNORECASE)
    
    return {
        "content": redacted_content,
        "redacted": redacted,
        "redaction_types": redaction_types
    }


def scan_for_secrets(content: str) -> Dict[str, Any]:
    """
    Scan for remaining secrets after redaction.
    
    Args:
        content: Content to scan
    
    Returns:
        Dict with secret count and types found
    """
    secret_count = 0
    secret_types = []
    
    for secret_type, patterns in REDACTION_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                secret_count += len(matches)
                if secret_type not in secret_types:
                    secret_types.append(secret_type)
    
    return {
        "secret_count": secret_count,
        "secret_types": secret_types
    }


def validate_message_schema(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate message has required schema fields.
    
    Args:
        message: Message dict to validate
    
    Returns:
        Dict with valid flag and errors list
    """
    required_fields = [
        "id", "source", "source_file", "session_id", "ordinal",
        "timestamp", "role", "content", "metadata", "redacted", "truncated"
    ]
    
    errors = []
    for field in required_fields:
        if field not in message:
            errors.append(f"Missing required field: {field}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def process_dataset(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process dataset with redaction and validation.
    
    Args:
        input_path: Input JSONL path
        output_dir: Output directory for cleaned data and rejection log
    
    Returns:
        Processing summary dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_path = output_dir / "messages.jsonl"
    rejection_path = output_dir / "rejections.jsonl"
    
    stats = {
        "processed": 0,
        "accepted": 0,
        "rejected": 0,
        "redacted_count": 0
    }
    
    with open(input_path, 'r', encoding='utf-8') as inf:
        with open(cleaned_path, 'w', encoding='utf-8') as outf:
            with open(rejection_path, 'w', encoding='utf-8') as rejf:
                
                for line_num, line in enumerate(inf, 1):
                    stats["processed"] += 1
                    
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Reject malformed JSON
                        rejection = {
                            "line_num": line_num,
                            "reason": "json_decode_error",
                            "error": str(e),
                            "reference_id": f"line_{line_num}"
                        }
                        rejf.write(json.dumps(rejection, ensure_ascii=False) + '\n')
                        stats["rejected"] += 1
                        continue
                    
                    # Validate schema
                    validation = validate_message_schema(message)
                    if not validation["valid"]:
                        rejection = {
                            "line_num": line_num,
                            "reason": "schema_validation_failed",
                            "errors": validation["errors"],
                            "reference_id": message.get("id", f"line_{line_num}")
                        }
                        rejf.write(json.dumps(rejection, ensure_ascii=False) + '\n')
                        stats["rejected"] += 1
                        continue
                    
                    # Redact sensitive content
                    redaction_result = redact_sensitive_data(message["content"])
                    message["content"] = redaction_result["content"]
                    
                    if redaction_result["redacted"]:
                        message["redacted"] = True
                        message["metadata"]["redaction_types"] = redaction_result["redaction_types"]
                        stats["redacted_count"] += 1
                    
                    # Write cleaned message
                    outf.write(json.dumps(message, ensure_ascii=False) + '\n')
                    stats["accepted"] += 1
    
    return stats


def audit_dataset(cleaned_dir: Path) -> Dict[str, Any]:
    """
    Run comprehensive audit on cleaned dataset.
    
    Args:
        cleaned_dir: Directory containing cleaned messages
    
    Returns:
        Audit report dict
    """
    messages_path = cleaned_dir / "messages.jsonl"
    
    audit = {
        "total_messages": 0,
        "secret_scan": {"passed": True, "secret_count": 0, "secret_types": []},
        "schema_check": {"passed": True, "errors": []},
    }
    
    if not messages_path.exists():
        audit["error"] = "Cleaned messages file not found"
        return audit
    
    with open(messages_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            audit["total_messages"] += 1
            
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                audit["schema_check"]["passed"] = False
                audit["schema_check"]["errors"].append(f"Line {line_num}: Invalid JSON")
                continue
            
            # Validate schema
            validation = validate_message_schema(message)
            if not validation["valid"]:
                audit["schema_check"]["passed"] = False
                audit["schema_check"]["errors"].extend(
                    [f"Line {line_num}: {err}" for err in validation["errors"]]
                )
            
            # Scan for remaining secrets
            secret_scan = scan_for_secrets(message["content"])
            if secret_scan["secret_count"] > 0:
                audit["secret_scan"]["passed"] = False
                audit["secret_scan"]["secret_count"] += secret_scan["secret_count"]
                for secret_type in secret_scan["secret_types"]:
                    if secret_type not in audit["secret_scan"]["secret_types"]:
                        audit["secret_scan"]["secret_types"].append(secret_type)
    
    return audit


def main():
    parser = argparse.ArgumentParser(
        description="Redact sensitive data and validate dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input normalized JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for cleaned data"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run audit on existing cleaned data"
    )
    
    args = parser.parse_args()
    
    if args.audit_only:
        print("Running audit...")
        audit = audit_dataset(args.output_dir)
        
        print(f"\n✓ Audit Results:")
        print(f"  Total messages: {audit['total_messages']}")
        print(f"  Schema check: {'✓ PASSED' if audit['schema_check']['passed'] else '✗ FAILED'}")
        print(f"  Secret scan: {'✓ PASSED' if audit['secret_scan']['passed'] else '✗ FAILED'}")
        
        if not audit['secret_scan']['passed']:
            print(f"    WARNING: {audit['secret_scan']['secret_count']} secrets found!")
            print(f"    Types: {', '.join(audit['secret_scan']['secret_types'])}")
        
        if not audit['schema_check']['passed']:
            print(f"    Schema errors: {len(audit['schema_check']['errors'])}")
            for error in audit['schema_check']['errors'][:5]:
                print(f"      - {error}")
        
        # Save audit report
        audit_path = args.output_dir / "audit_report.json"
        with open(audit_path, 'w') as f:
            json.dump(audit, f, indent=2)
        print(f"\n✓ Audit report: {audit_path}")
        
        if not (audit['schema_check']['passed'] and audit['secret_scan']['passed']):
            exit(1)
    
    else:
        print(f"Processing {args.input}...")
        stats = process_dataset(args.input, args.output_dir)
        
        print(f"\n✓ Processing complete:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Accepted: {stats['accepted']}")
        print(f"  Rejected: {stats['rejected']}")
        print(f"  Redacted: {stats['redacted_count']}")
        
        print(f"\n✓ Output:")
        print(f"  Cleaned: {args.output_dir / 'messages.jsonl'}")
        print(f"  Rejections: {args.output_dir / 'rejections.jsonl'}")
        
        # Run audit
        print(f"\nRunning audit...")
        audit = audit_dataset(args.output_dir)
        
        audit_path = args.output_dir / "audit_report.json"
        with open(audit_path, 'w') as f:
            json.dump(audit, f, indent=2)
        
        if not (audit['schema_check']['passed'] and audit['secret_scan']['passed']):
            print("\n✗ AUDIT FAILED - dataset not ready for training")
            exit(1)
        else:
            print("\n✓ Audit passed - dataset ready")


if __name__ == "__main__":
    main()
