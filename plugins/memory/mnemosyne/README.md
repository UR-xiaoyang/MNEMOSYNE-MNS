# MNEMOSYNE Hermes memory provider

This standalone provider implements Hermes' `MemoryProvider` lifecycle without changing Hermes core.

## Install for a local Hermes checkout

Copy this directory to the Hermes user-plugin memory path:

```text
~/.hermes/plugins/memory/mnemosyne/
```

The provider expects a running MNEMOSYNE service:

```bash
MNEMOSYNE_API_URL=http://127.0.0.1:8000
hermes memory setup
```

`MNEMOSYNE_API_KEY` is optional and is sent as a bearer token when configured.

## Memory Modes

MNEMOSYNE supports two distinct memory modes controlled by `MNEMOSYNE_MEMORY_MODE`:

### Session Mode (Default)

**Default behavior when `MNEMOSYNE_MEMORY_MODE` is unset or set to `session`.**

- **SQLite conversation persistence**: All conversation turns are stored in local SQLite
- **Same-session context only**: Context retrieval is scoped strictly to the current session_id
- **No cross-conversation memory**: Does NOT provide global/project memory across sessions
- **No automatic promotion**: Conversation turns remain session-scoped; never auto-promoted to global
- **Provider prefix**: `[Previous conversation context]`
- **No proposals tool**: `mnemosyne_write_proposals` is NOT exposed to the agent

This is the recommended mode for most users. It provides reliable conversation history within each session without cross-session state.

### Experimental Global Mode (Opt-in)

**Enabled by setting `MNEMOSYNE_MEMORY_MODE=experimental_global`.**

- **Cross-conversation memory**: Provides experimental global memory that persists across sessions
- **Separate namespace**: Uses `experimental_global:` prefix, completely isolated from Hermes built-in memory
- **Explicit approval required**: All writes require `approved=true`; proposals never auto-commit
- **Does NOT touch Hermes memory**: Will never read, modify, or migrate Hermes' built-in MEMORY.md, USER.md, skills, or state.db
- **Proposals tool exposed**: `mnemosyne_write_proposals` tool is available in this mode
- **Provider segments**: Context includes both session recent + experimental global memories in distinct sections

**This mode is experimental and opt-in only.** To revert: set `MNEMOSYNE_MEMORY_MODE=session` and restart the service.

## Lifecycle behavior

- `queue_prefetch()` calls `/context/recent` (session mode) or `/context/recent` + `/experimental/global-memories/retrieve` (experimental_global mode)
- `prefetch()` injects context with mode-appropriate header:
  - **Session mode**: `[Previous conversation context]` with chronological session turns only
  - **Experimental_global mode**: Two distinct sections with headers `[Previous conversation context]` and `[Experimental cross-conversation memory]`
- `sync_turn()` persists the completed user/assistant exchange as session-scoped working context via `/context/turn` with server-side payload filtering
- Shadow/citation tracing requires explicit `MNEMOSYNE_EXPERIMENTAL_SHADOW=1` (not enabled by default)
- Session context is visible only when the Hermes session ID matches
- The provider is local-first and does not upload data except to the configured MNEMOSYNE API URL

## Configuration Variables

- `MNEMOSYNE_API_URL`: Service URL (default: `http://127.0.0.1:8000`)
- `MNEMOSYNE_API_KEY`: Optional Bearer token for authentication
- `MNEMOSYNE_MEMORY_MODE`: `session` (default) or `experimental_global`
- `MNEMOSYNE_DB`: SQLite database path (default: `./memory.db`)
- `MNEMOSYNE_EXPERIMENTAL_SHADOW`: Enable Shadow Mode tracing (`0` or `1`, default `0`; requires explicit opt-in)
