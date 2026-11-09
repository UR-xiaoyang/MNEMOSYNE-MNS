# Hermes Feature Request: Memory Provider Auto-Store Support

## Summary

Request to enhance the Memory Provider interface to support **automatic memory storage** triggered by user conversations, enabling seamless knowledge accumulation without manual intervention.

## Current Limitation

Currently, the `MemoryProvider.sync_turn()` method is called after each conversation turn, but:

1. **Plugin providers cannot reliably trigger write operations from sync_turn**
   - The method is designed for recording conversation history
   - Cross-dialogue write operations (like `write_proposals`) may be restricted
   - No clear guidance on what's allowed in `sync_turn`

2. **No built-in mechanism for auto-store**
   - Users must manually call memory tools to save important information
   - No way to automatically detect and store key information from conversations
   - Breaks the flow of natural conversation

## Proposed Enhancement

### Option 1: Add `on_turn_complete` Hook (Recommended)

Add a new lifecycle hook that explicitly supports write operations:

```python
class MemoryProvider(ABC):
    def on_turn_complete(
        self, 
        user_content: str, 
        assistant_content: str,
        *,
        session_id: str = "",
        should_store: bool = False  # Hint from Hermes
    ) -> None:
        """
        Called after turn completion with permission to write.
        
        Unlike sync_turn (history recording), this hook explicitly 
        supports cross-dialogue writes like memory storage.
        
        Args:
            should_store: Hermes's suggestion based on conversation 
                         context (e.g., user said "remember this")
        """
        pass
```

**Benefits**:
- Clear separation: `sync_turn` = history, `on_turn_complete` = storage
- Backward compatible (optional hook)
- Hermes can provide hints about storage-worthy content

### Option 2: Clarify `sync_turn` Permissions

Update documentation to explicitly allow:
- Write operations (not just read/record)
- Cross-dialogue memory storage
- Calling provider's own write APIs

And provide examples in the Memory Provider integration guide.

### Option 3: Add Auto-Store Config

Add framework-level support:

```yaml
# config.yaml
memory:
  provider: mnemosyne
  auto_store:
    enabled: true
    triggers:
      - "remember"
      - "save this"
      - "important"
    mode: experimental_global  # Required for writes
```

Hermes would then:
1. Detect trigger phrases
2. Call provider's storage method
3. Handle errors gracefully

## Use Case

**MNEMOSYNE Memory System** needs to automatically:
1. Detect when user says "请记住..." (remember...) or mentions project status
2. Extract key information from the conversation
3. Store it for future retrieval using V3 ranking model
4. Track citations for Bayesian statistics

**Current workaround**:
- Modified Hermes source code directly (`run_agent.py`)
- Added custom logic in plugin's `sync_turn`
- No official support or documentation

## Expected Behavior

```python
# In Hermes conversation:
User: "Please remember: MNEMOSYNE V3 is deployed with 334 memories"
Assistant: "I'll remember that."

# Behind the scenes:
# 1. Hermes calls memory_provider.on_turn_complete()
# 2. Provider detects trigger phrase "Please remember"
# 3. Provider extracts: "MNEMOSYNE V3 is deployed with 334 memories"
# 4. Provider stores to database
# 5. User continues conversation naturally
```

Later:
```python
User: "What's the status of MNEMOSYNE V3?"
# Provider automatically retrieves stored memory
Assistant: "MNEMOSYNE V3 is deployed with 334 memories."
```

## Related Files

In current Hermes codebase:
- `agent/memory_provider.py` - Base class definition
- `agent/memory_manager.py` - Calls `sync_turn()`
- `run_agent.py` - Turn completion logic

## Implementation Notes

### For Plugin Authors

If Option 1 is implemented, plugin authors can:

```python
class MyMemoryProvider(MemoryProvider):
    def on_turn_complete(self, user_content, assistant_content, **kwargs):
        # Safe to write here
        if self._should_auto_store():
            key_info = self._extract_key_info(user_content, assistant_content)
            if key_info:
                self._write_to_store(key_info)
```

### For Hermes Core

```python
# In run_agent.py or memory_manager.py
def _mirror_turn_to_memory(...):
    # Existing sync_turn call
    self._memory_manager.sync_all(user_text, response_text, **kwargs)
    
    # New hook
    should_store = self._detect_storage_intent(user_text)
    self._memory_manager.on_turn_complete_all(
        user_text, 
        response_text, 
        should_store=should_store,
        **kwargs
    )
```

## Benefits

1. **Better UX**: Users don't need to explicitly call memory tools
2. **Natural Conversations**: "Remember this" works like talking to a human
3. **Plugin Flexibility**: Each provider can implement their own auto-store logic
4. **Clear API**: Separation of concerns (recording vs storing)
5. **Backward Compatible**: Optional feature, doesn't break existing providers

## Priority

**High** - This is a fundamental feature for memory systems that want to:
- Accumulate knowledge automatically
- Track project status over time
- Build long-term context
- Reduce cognitive load on users

## Alternative Workarounds

Currently, users must either:
1. Modify Hermes source code (fragile, not upgradeable)
2. Manually call memory tools every time (poor UX)
3. Use external scripts to parse conversation history (unreliable)

None of these are sustainable solutions.

---

**Would you consider adding this feature to Hermes?**

We're happy to contribute a PR if the approach is acceptable. The MNEMOSYNE project already has a working prototype that could be adapted.

## Additional Context

- **Project**: MNEMOSYNE Memory System (github.com:UR-xiaoyang/MNEMOSYNE-MNS)
- **Current Hermes Version**: v0.22.5
- **Use Case**: Automatic memory accumulation with V3 ranking and Bayesian statistics
- **Community Interest**: Would benefit any memory provider that wants auto-store functionality
