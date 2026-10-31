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

## Lifecycle behavior

- `queue_prefetch()` calls `/retrieve_store` for the next turn.
- `prefetch()` injects only returned `injected=true` memory content.
- `retrieve_event_id` is accepted from the API for Shadow Mode correlation.
- `sync_turn()` is intentionally a no-op; model output never commits resident memory.
- `mnemosyne_write_proposals` calls `/write_proposals` and returns suggestions only.
- An explicit host action must call `/memories` or `/memories/supersede` to mutate storage.

The provider is local-first and does not upload data except to the MNEMOSYNE API URL configured by the user.
