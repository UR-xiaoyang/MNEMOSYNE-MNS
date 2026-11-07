#!/usr/bin/env python3
"""Black-box smoke test and latency sampler for the native MNEMOSYNE gateway.

This script uses only the Python standard library. It does not benchmark the
Python gateway: use it after obtaining an equivalent authenticated baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", type=Path, required=True, help="Path to mnemosyne_gateway.exe")
    parser.add_argument("--sqlite-dll", required=True, help="Vetted SQLite DLL with FTS5 support")
    parser.add_argument("--database", type=Path, required=True, help="Disposable SQLite path for this smoke test")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--samples", type=int, default=200)
    return parser.parse_args()


def request(base_url: str, method: str, path: str, token: str, payload: dict | None = None) -> tuple[int, dict]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    http_request = Request(f"{base_url}{path}", data=encoded, method=method, headers=headers)
    try:
        with urlopen(http_request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def wait_until_ready(base_url: str, token: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"gateway exited before readiness (exit={process.returncode})")
        try:
            status, payload = request(base_url, "GET", "/health", token)
            if status == 200 and payload.get("model") == "native-session-context":
                return
        except URLError:
            pass
        time.sleep(0.05)
    raise TimeoutError("gateway did not become ready within 5 seconds")


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    args = parse_args()
    if not args.gateway.is_file():
        raise FileNotFoundError(args.gateway)
    if not Path(args.sqlite_dll).is_file():
        raise FileNotFoundError(args.sqlite_dll)
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    args.database.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (args.database, Path(f"{args.database}-wal"), Path(f"{args.database}-shm")):
        candidate.unlink(missing_ok=True)

    token = "native-gateway-smoke-token"
    environment = os.environ.copy()
    environment.update(
        {
            "MNEMOSYNE_SQLITE_DLL": str(Path(args.sqlite_dll).resolve()),
            "MNEMOSYNE_API_KEY": token,
            "MNEMOSYNE_DB": str(args.database.resolve()),
        }
    )
    process = subprocess.Popen(
        [str(args.gateway.resolve()), "--port", str(args.port)],
        cwd=args.gateway.resolve().parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{args.port}"

    try:
        wait_until_ready(base_url, token, process)
        turn = {
            "session_id": "native-smoke",
            "turn_id": "turn-1",
            "project_id": "mnemosyne-mns",
            "user_text": "Use SQLite for the local store.",
            "assistant_text": "The native gateway stores session context in SQLite.",
        }
        status, created = request(base_url, "POST", "/context/turn", token, turn)
        assert status == 200 and created["memory_id"] == "session:native-smoke:turn-1"
        status, duplicate = request(base_url, "POST", "/context/turn", token, turn)
        assert status == 200 and duplicate == created
        status, recent = request(
            base_url,
            "POST",
            "/context/recent",
            token,
            {"session_id": "native-smoke", "project_id": "mnemosyne-mns", "max_tokens": 4000},
        )
        assert status == 200 and len(recent["results"]) == 1
        assert request(base_url, "POST", "/retrieve", token, {})[0] == 409
        assert request(base_url, "GET", "/", "wrong-token")[0] == 401
        checkpoint_payload = {
            "session_id": "native-smoke",
            "turn_id": "checkpoint-1",
            "project_id": "mnemosyne-mns",
            "transcript": "first checkpoint",
        }
        status, first_checkpoint = request(base_url, "POST", "/context/checkpoint", token, checkpoint_payload)
        assert status == 200
        status, duplicate_checkpoint = request(base_url, "POST", "/context/checkpoint", token, checkpoint_payload)
        assert status == 200 and duplicate_checkpoint == first_checkpoint
        status, second_checkpoint = request(
            base_url,
            "POST",
            "/context/checkpoint",
            token,
            {**checkpoint_payload, "turn_id": "checkpoint-2", "transcript": "second checkpoint"},
        )
        assert status == 200 and second_checkpoint != first_checkpoint

        timings = []
        write_timings = []
        append_and_snapshot_timings = []
        payload = {"session_id": "native-smoke", "project_id": "mnemosyne-mns", "max_tokens": 4000}
        for _ in range(args.samples):
            started = time.perf_counter()
            status, response = request(base_url, "POST", "/context/recent", token, payload)
            timings.append((time.perf_counter() - started) * 1000)
            assert status == 200 and response["results"][0]["memory_id"] == created["memory_id"]
        for index in range(args.samples):
            session_id = f"native-benchmark-{index}"
            benchmark_turn = {
                "session_id": session_id,
                "turn_id": "turn-1",
                "project_id": "mnemosyne-mns",
                "user_text": "native write benchmark",
                "assistant_text": "session data persisted",
            }
            started = time.perf_counter()
            status, response = request(base_url, "POST", "/context/turn", token, benchmark_turn)
            write_timings.append((time.perf_counter() - started) * 1000)
            assert status == 200 and response["memory_id"] == f"session:{session_id}:turn-1"
            started = time.perf_counter()
            status, response = request(
                base_url,
                "POST",
                "/context/recent",
                token,
                {"session_id": session_id, "project_id": "mnemosyne-mns", "max_tokens": 4000},
            )
            append_and_snapshot_timings.append(write_timings[-1] + (time.perf_counter() - started) * 1000)
            assert status == 200 and response["results"][0]["memory_id"] == f"session:{session_id}:turn-1"
        summary = lambda values: {
            "min": round(min(values), 3),
            "p50": round(statistics.median(values), 3),
            "p95": round(percentile(values, 0.95), 3),
            "max": round(max(values), 3),
        }
        print(
            json.dumps(
                {
                    "status": "pass",
                    "samples": args.samples,
                    "recent_round_trip_ms": summary(timings),
                    "turn_write_round_trip_ms": summary(write_timings),
                    "turn_write_plus_recent_ms": summary(append_and_snapshot_timings),
                },
                indent=2,
            )
        )
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


if __name__ == "__main__":
    sys.exit(main())
