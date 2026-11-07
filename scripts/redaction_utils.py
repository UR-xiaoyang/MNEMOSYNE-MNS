"""Text redaction utilities for sensitive data removal."""

import re
from typing import Dict, List, Tuple


def detect_and_redact_secrets(text: str) -> Tuple[str, List[str]]:
    """
    Detect and redact sensitive information from text.
    
    Args:
        text: Input text to scan and redact
        
    Returns:
        Tuple of (redacted_text, list_of_detected_secret_types)
    """
    detected_types = []
    redacted = text
    
    # API keys (various patterns)
    api_key_patterns = [
        (r'sk-[a-zA-Z0-9]{20,}', 'openai_api_key'),
        (r'Bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'bearer_token'),
        (r'api[_-]?key[\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?', 'api_key'),
    ]
    
    for pattern, secret_type in api_key_patterns:
        if re.search(pattern, redacted, re.IGNORECASE):
            detected_types.append(secret_type)
            redacted = re.sub(pattern, '[REDACTED_API_KEY]', redacted, flags=re.IGNORECASE)
    
    # Private keys
    if re.search(r'-----BEGIN.*PRIVATE KEY-----', redacted):
        detected_types.append('private_key')
        redacted = re.sub(
            r'-----BEGIN.*PRIVATE KEY-----.*?-----END.*PRIVATE KEY-----',
            '[REDACTED_PRIVATE_KEY]',
            redacted,
            flags=re.DOTALL
        )
    
    # Connection strings
    conn_patterns = [
        (r'(postgres|mysql|mongodb)://[^\s]+', 'connection_string'),
        (r'Server=.*?;.*?Password=[^;]+', 'connection_string'),
    ]
    
    for pattern, secret_type in conn_patterns:
        if re.search(pattern, redacted, re.IGNORECASE):
            if secret_type not in detected_types:
                detected_types.append(secret_type)
            redacted = re.sub(pattern, '[REDACTED_CONNECTION_STRING]', redacted, flags=re.IGNORECASE)
    
    # Email addresses
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', redacted):
        detected_types.append('email')
        redacted = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[REDACTED_EMAIL]',
            redacted
        )
    
    # IP addresses
    if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', redacted):
        detected_types.append('ip_address')
        redacted = re.sub(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            '[REDACTED_IP]',
            redacted
        )
    
    # Windows/Unix home paths
    home_patterns = [
        (r'C:\\Users\\[^\\]+', 'windows_home_path'),
        (r'/home/[^/\s]+', 'unix_home_path'),
        (r'/Users/[^/\s]+', 'macos_home_path'),
    ]
    
    for pattern, secret_type in home_patterns:
        if re.search(pattern, redacted):
            if 'home_path' not in detected_types:
                detected_types.append('home_path')
            redacted = re.sub(pattern, '[REDACTED_HOME_PATH]', redacted)
    
    return redacted, detected_types


def scan_for_secrets(text: str) -> Dict[str, int]:
    """
    Scan text for secrets without redacting (for reporting).
    
    Returns:
        Dict mapping secret types to counts
    """
    _, detected = detect_and_redact_secrets(text)
    
    counts = {}
    for secret_type in detected:
        counts[secret_type] = counts.get(secret_type, 0) + 1
    
    return counts
