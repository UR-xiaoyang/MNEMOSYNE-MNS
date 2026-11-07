"""Mode-aware memory configuration for MNEMOSYNE.

Supports two modes:
- session (default): SQLite conversation persistence, same-session context only
- experimental_global: Optional cross-conversation memory (does NOT touch Hermes global memory)
"""

from __future__ import annotations

import os
from typing import Literal


MemoryMode = Literal["session", "experimental_global"]


def get_memory_mode() -> MemoryMode:
    """Get the configured memory mode from environment.
    
    Returns:
        'session' (default) or 'experimental_global'
        
    Raises:
        ValueError: If MNEMOSYNE_MEMORY_MODE is set to an unknown value
    """
    mode_str = os.environ.get("MNEMOSYNE_MEMORY_MODE", "session").strip().lower()
    
    if mode_str == "session":
        return "session"
    elif mode_str == "experimental_global":
        return "experimental_global"
    else:
        raise ValueError(
            f"Unknown MNEMOSYNE_MEMORY_MODE: '{mode_str}'. "
            f"Valid values: 'session', 'experimental_global'"
        )


def is_experimental_global_enabled() -> bool:
    """Check if experimental global memory mode is enabled.
    
    Returns:
        True if mode is experimental_global, False otherwise
    """
    try:
        return get_memory_mode() == "experimental_global"
    except ValueError:
        # Unknown mode: fail safely by defaulting to session
        return False


def should_expose_proposals_tool() -> bool:
    """Check if mnemosyne_write_proposals tool should be exposed.
    
    Returns:
        True only in experimental_global mode
    """
    return is_experimental_global_enabled()
