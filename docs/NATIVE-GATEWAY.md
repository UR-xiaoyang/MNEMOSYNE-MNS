# Native C++ Session Gateway

`cpp/gateway.cpp` is the high-throughput local gateway for MNEMOSYNE's default **session** mode. It replaces the Python/FastAPI process on the latency-sensitive path while retaining the existing HTTP contract used by the Hermes MemoryProvider.

## Scope

The native gateway owns:

- loopback-only HTTP ingress (`127.0.0.1`);
- bearer-token validation with constant-time comparison;
- SQLite/WAL session-turn persistence and FTS updates;
- chronological, token-budgeted `/context/recent` reads;
- payload boundary checks and mode-aware `409` responses.

It does **not** claim to replace the Hybrid ranker yet. E5/ONNX tokenization, ranking, Shadow data export, and experimental-global policy are deliberately outside this first native slice. Those paths remain served by the Python runtime until the ONNX tokenizer and ranking parity are independently tested. This prevents silently changing retrieval quality merely to remove Python.

The native process and the existing Python service use the same `memories` and `memories_fts` SQLite schema, so they can be cut over without a data conversion. Do not run both writers against the same database in production; use one gateway at a time.

## Build (Windows / MSYS2 UCRT64)

CMake is supported through `cpp/CMakeLists.txt`, but is not required for the direct MSYS2 build:

```bash
cd /d/project/MNEMOSYNE-MNS
mkdir -p cpp/build
ROOT_NATIVE="$(cygpath -w "$PWD")"
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -Wno-cast-function-type "$ROOT_NATIVE/cpp/gateway.cpp" \
  -o "$ROOT_NATIVE/cpp/build/mnemosyne_gateway.exe" -lws2_32
```

The executable dynamically loads SQLite to keep the build dependency-free. Supply a vetted SQLite DLL with FTS5 enabled through `MNEMOSYNE_SQLITE_DLL`; do not point it at Python's `_sqlite3.pyd` or assume an arbitrary `sqlite3.dll` is ABI-compatible.

```bash
export MNEMOSYNE_SQLITE_DLL='C:/path/to/sqlite3.dll'
export MNEMOSYNE_DB='D:/path/to/data/memory.db'
export MNEMOSYNE_API_KEY='<secret>'
./cpp/build/mnemosyne_gateway.exe --port 8000
```

The gateway refuses non-loopback bindings. Keep it on `127.0.0.1`; a local Hermes provider does not need LAN exposure.

## HTTP compatibility

| Endpoint | Native session behavior |
|---|---|
| `GET /health` | Public health response, reports `native-session-context` |
| `GET /` | Authenticated service discovery, reports `memory_mode: session` |
| `POST /context/turn` | Idempotently writes a visible user/assistant turn |
| `POST /context/checkpoint` | Writes an idempotent checkpoint; checkpoints are excluded from recent-turn retrieval |
| `POST /context/recent` | Returns chronological turns under a token budget |
| Legacy retrieval / ranking paths | Explicit `409`; use the Python service for `experimental_global` or Hybrid ranking |

Bodies must use direct UTF-8. The intentionally narrow parser rejects `\uXXXX` escape sequences rather than risk corrupting memory content.

## Verification and benchmark

Run the black-box smoke test against a disposable database and the same SQLite DLL that will be deployed:

```bash
python cpp/native_gateway_smoke.py \
  --gateway cpp/build/mnemosyne_gateway.exe \
  --sqlite-dll 'C:/path/to/sqlite3.dll' \
  --database cpp/build/native_gateway_smoke.db \
  --samples 1000
```

The output records native `/context/recent` round-trip p50/p95 latency. It is **not** a comparison against the Python service. Claim a speedup only after running the same request mix, database snapshot, authentication setting, and sample count against both gateways.

## Cutover and rollback

1. Back up the SQLite database while the current gateway is stopped.
2. Run the native smoke test on a disposable DB.
3. Start the native gateway on an unused loopback port and point `MNEMOSYNE_API_URL` at it.
4. Verify a real Hermes session writes and recalls context.
5. Stop the Python gateway and move the native service to the production port.

Rollback is immediate: stop the native process and restore the existing Python service with the same database. No schema migration is required.
