"""Server-side payload filtering for turn recording."""

from __future__ import annotations

import re


def is_visible_user_or_assistant_content(text: str, role: str) -> bool:
    """Accept only visible user/assistant text; reject system/developer/skill/agent content.
    
    Args:
        text: The text content to validate
        role: Either 'user' or 'assistant'
        
    Returns:
        True if content appears to be normal visible conversation text
    """
    if not text or not text.strip():
        return False
    
    # Reject system/developer role markers
    system_markers = [
        r"<\s*system\s*>",
        r"<\s*/\s*system\s*>",
        r"<\s*developer\s*>",
        r"<\s*/\s*developer\s*>",
        r"#\s*System\s*Prompt",
        r"#\s*SYSTEM\s*:",
        r"You\s+are\s+an?\s+(AI|assistant|agent)",
        r"^System:",
        r"^Developer:",
    ]
    
    for pattern in system_markers:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False
    
    # Reject skill injection patterns
    skill_markers = [
        r"<\s*skill\s+name=",
        r"<\s*skills?\s*>",
        r"^\s*#\s*Skills?\s*$",
        r"skill_view\s*\(",
        r"skill_manage\s*\(",
    ]
    
    for pattern in skill_markers:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False
    
    # Reject agent/framework injection patterns
    agent_markers = [
        r"<\s*thinking\s*>",
        r"<\s*reflection\s*>",
        r"<\s*antml:function_calls\s*>",
        r"<\s*memory-context\s*>",
        r"\[System\s+note:",
        r"\[MNEMOSYNE\s+Context\]",
        r"\[Previous\s+conversation\s+context\]",
        r"^\s*```\s*thinking",
    ]
    
    for pattern in agent_markers:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False
    
    # All checks passed
    return True


def filter_turn_payload(user_text: str, assistant_text: str) -> tuple[bool, str]:
    """Validate turn payload contains only visible conversation text.
    
    Args:
        user_text: User message content
        assistant_text: Assistant response content
        
    Returns:
        (is_valid, error_message) tuple
    """
    if not is_visible_user_or_assistant_content(user_text, "user"):
        return False, "user_text contains system/developer/skill/agent content"
    
    if not is_visible_user_or_assistant_content(assistant_text, "assistant"):
        return False, "assistant_text contains system/developer/skill/agent content"
    
    return True, ""
