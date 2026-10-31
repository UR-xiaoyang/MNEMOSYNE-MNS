# Hermes Agent Integration

`HermesMemoryAdapter` is the host-agent bridge for the local MNEMOSYNE-MNS pipeline. It is intentionally an explicit Python API: it does not monkey-patch Hermes, alter resident memory, or write a proposal automatically.

## Lifecycle

```text
before_turn()
  ROUTE → MemoryStore retrieve → RANK → ASSEMBLE
  → context_text + optional retrieval trace

host model generates a response

after_turn()
  citation detection
  → optional citation and ignored traces

record_correction()
  explicit correction trace for a cited memory
```

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

## Safety contract

- `before_turn()` never writes a memory.
- `after_turn()` only detects citations and, when enabled, records trace events.
- `record_correction()` records a correction only when the memory was cited in this turn.
- Long-term memory changes still require an explicit `MemoryService.write()` or `MemoryService.supersede()` call.
- Shadow Mode is disabled by default and stores data locally in SQLite.

## Hermes plugin provider

For native Hermes lifecycle integration, copy `plugins/memory/mnemosyne/` into Hermes' user plugin memory directory. It implements the `MemoryProvider` contract and calls the HTTP API:

```text
~/.hermes/plugins/memory/mnemosyne/
```

Configure `MNEMOSYNE_API_URL` (default `http://127.0.0.1:8000`) and optionally `MNEMOSYNE_API_KEY`. The plugin keeps Shadow citation logging opt-in through `MNEMOSYNE_SHADOW_MODE=1`.
