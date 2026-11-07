# Hermes Agent Integration

`HermesMemoryAdapter` is the host-agent bridge for the local MNEMOSYNE-MNS pipeline. It is intentionally an explicit Python API: it does not monkey-patch Hermes, alter resident memory, or write a proposal automatically.

## Memory Architecture

MNEMOSYNE operates in one of two modes controlled by `MNEMOSYNE_MEMORY_MODE`:

### Session Mode (Default)

**Default when `MNEMOSYNE_MEMORY_MODE` is unset or set to `session`.**

- **SQLite conversation persistence**: Stores all conversation turns in local SQLite
- **Same-session context recovery**: Retrieves context only from the current session_id
- **No cross-conversation state**: Does NOT provide global/project memory across sessions
- **Provider behavior**: Calls `/context/recent` and injects with header `[Previous conversation context]`
- **Payload filtering**: Server-side filtering rejects system/developer/skill/agent injection attempts
- **No proposals tool**: `mnemosyne_write_proposals` is not exposed to the agent
- **No Shadow/citation**: Shadow tracing disabled unless `MNEMOSYNE_EXPERIMENTAL_SHADOW=1` is explicitly set

This is the **recommended default mode** for most deployments.

### Experimental Global Mode (Opt-in)

**Enabled by setting `MNEMOSYNE_MEMORY_MODE=experimental_global`.**

- **Cross-conversation memory**: Provides experimental global memory across sessions
- **Isolated namespace**: Uses `experimental_global:` prefix; completely separate from Hermes built-in memory
- **Explicit approval required**: All writes to `/experimental/global-memories` must have `approved=true`; stored in dedicated SQLite table
- **Non-destructive**: Will NEVER read, modify, or migrate Hermes' built-in MEMORY.md, USER.md, skills, or state.db
- **Proposals tool exposed**: `mnemosyne_write_proposals` is available in this mode
- **Segmented context**: Provider calls `/context/recent` + `/experimental/global-memories/retrieve` and injects with distinct headers `[Previous conversation context]` and `[Experimental cross-conversation memory]`

**Important**: This mode is experimental. It does not touch Hermes global memory. To disable: set `MNEMOSYNE_MEMORY_MODE=session` and restart.

## Lifecycle

### Session Mode (Default)

```text
before_turn()
  → /context/recent (chronological session context)
  → context_text injected with header [Previous conversation context]

host model generates a response

after_turn() / sync_turn()
  → /context/turn (persist turn in session history, idempotent per turn)
```

### Experimental Global Mode (Legacy)

```text
before_turn()
  ROUTE → MemoryStore retrieve (session/project/global) → RANK → ASSEMBLE
  → context_text + optional retrieval trace

host model generates a response

after_turn() / sync_turn()
  → persist a session-scoped working-context entity (idempotent per turn)
  → optional citation and ignored traces

record_correction()
  explicit correction trace for a cited memory
```

**Note**: The experimental global mode lifecycle with ROUTE/RANK/ASSEMBLE is a legacy experimental path. Most deployments should use session mode.

## Minimal setup

```python
from hermes_memory_adapter import HermesMemoryAdapter, MemoryAdapterConfig
from memory_service import MemoryService
from memory_store import MemoryStore
from shadow_mode import ShadowModeCollector

store = MemoryStore("~/.hermes/mnemosyne/memory.db")
service = MemoryService(pipeline, store)
shadow = ShadowModeCollector("~/.hermes/mnemosyne/shadow.db")
adapter = HermesMemoryAdapter(
    service,
    shadow,
    MemoryAdapterConfig(
        shadow_enabled=False,       # opt in explicitly
        top_k=20,
        max_context_tokens=4000,
        tokens_per_memory=200,
    ),
)
```

## Before and after a turn

```python
context = adapter.before_turn(
    user_text,
    session_id=session_id,
    turn_id=turn_id,
    scope_level="session",
    destination="local",
)

# Pass context.context_text to the host model's prompt builder.
response = await generate_response(user_text, context.context_text)
trace = adapter.after_turn(context, response)

if user_corrected_memory_id:
    adapter.record_correction(
        trace,
        memory_id=user_corrected_memory_id,
        correction_type="outdated",
        feedback={"action": "supersede"},
    )
```

`TurnContext.assembled` contains the budget-admitted memory records. `TurnContext.retrieval_event_id` is populated only when Shadow Mode is enabled and a collector is supplied.

## Session working context

Every completed Hermes turn is stored through `POST /context/turn` as a `session` entity. Server-side payload filtering validates that only visible user/assistant text is stored; system prompts, developer instructions, skill injections, and agent framework markers are rejected with HTTP 422.

The ID is deterministic (`session:<session_id>:<turn_id>`), so retries do not duplicate context. On the next turn, `/context/recent` retrieves entities in chronological order whose session ID matches, then applies token-budget truncation. These entities are never promoted to project/global memory automatically.

Before Hermes compacts its transcript, the provider creates a checkpoint through `POST /context/checkpoint`. It is bounded (at most 12,000 characters) and keyed by session, turn and transcript hash. A newer checkpoint in the same session supersedes the previous checkpoint. MNEMOSYNE advertises checkpoint API v2: a required checkpoint failure stops compaction rather than silently discarding history.

`project_id` is taken from Hermes' workspace (`cwd`) and carried on retrieval requests, creating an identity boundary in addition to the `session` / `project` / `global` visibility hierarchy.

## Safety contract

- `before_turn()` never writes a memory.
- `sync_turn()` writes only a session-scoped working-context record via `/context/turn`; server-side filtering prevents injection of system/developer/skill/agent content.
- Session mode: zero cross-conversation retrieval; `/context/recent` returns empty for unknown session_id.
- Experimental global mode: `/experimental/global-memories` requires `approved=true`; unapproved or duplicate writes are rejected with HTTP 422.
- Neither mode reads, modifies, or migrates Hermes' built-in global MEMORY.md, USER.md, skills, or state.db.
- Explicit project/session writes must include matching identity; missing identity is not a wildcard.
- Citation detection and trace recording are opt-in through Shadow Mode.
- `record_correction()` records a correction only when the memory was cited in this turn.
- Long-term memory changes require explicit `MemoryService.write()` or `MemoryService.supersede()` calls.
- Shadow Mode is disabled by default and stores data locally in SQLite.

## Hermes plugin provider

For native Hermes lifecycle integration, copy `plugins/memory/mnemosyne/` into Hermes' user plugin memory directory. It implements the `MemoryProvider` contract and calls the HTTP API:

```text
~/.hermes/plugins/memory/mnemosyne/
```

Configure:
- `MNEMOSYNE_API_URL` (default `http://127.0.0.1:8000`)
- `MNEMOSYNE_API_KEY` (optional Bearer token)
- `MNEMOSYNE_MEMORY_MODE` (`session` or `experimental_global`, default `session`)
- `MNEMOSYNE_DB` (SQLite path, default `./memory.db`)
- `MNEMOSYNE_EXPERIMENTAL_SHADOW` (`0` or `1`, default `0`; explicit opt-in required for Shadow/citation tracing)

When `MNEMOSYNE_API_KEY` is set on the service, all endpoints except `/health` require the matching `Authorization: Bearer ***` header; the provider sends this automatically.

## Rollback and Recovery

If experimental global mode causes issues:
1. Set `MNEMOSYNE_MEMORY_MODE=session`
2. Restart the MNEMOSYNE service
3. Hermes built-in memory is unaffected; experimental entries remain in SQLite with `experimental_global:` prefix and can be deleted safely

Unknown `MNEMOSYNE_MEMORY_MODE` values: the service logs an error, falls back to `session` mode, and continues.
