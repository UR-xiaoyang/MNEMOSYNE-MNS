"""Hermes MemoryProvider plugin for a local MNEMOSYNE-MNS service."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable

from agent.memory_provider import MemoryProvider, RecallStatus, spawn_context_thread

_DEFAULT_URL = "http://127.0.0.1:8000"


class MnemosyneHTTPClient:
    """Small requests-based client for the public MNEMOSYNE API."""

    def __init__(self, *, base_url: str = _DEFAULT_URL, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            json=payload,
            timeout=8,
        )
        response.raise_for_status()
        return response.json()

    def retrieve_store(
        self,
        query: str,
        *,
        session_id: str,
        turn_id: str,
        scope_level: str,
        destination: str,
        top_k: int,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/retrieve_store",
            {
                "query": query,
                "memories": [],
                "top_k": top_k,
                "scope_level": scope_level,
                "destination": destination,
                "session_id": session_id,
                "turn_id": turn_id,
            },
        )

    def write_proposals(self, text: str, *, scope_level: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/write_proposals",
            {"text": text, "scope_level": scope_level},
        )

    def detect_citations(self, response: str, injected_memories: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/detect_citations",
            {"response": response, "injected_memories": injected_memories},
        )

    def log_citation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/shadow/citations", payload)


class MnemosyneMemoryProvider(MemoryProvider):
    """Use MNEMOSYNE as Hermes' explicit retrieve/rank/assemble provider.

    The provider has no implicit remember tool and ``sync_turn`` is deliberately
    a no-op. Long-term writes remain proposals plus an explicit host/API commit.
    """

    def __init__(self, client_factory: Callable[..., Any] | None = None) -> None:
        self._client_factory = client_factory or MnemosyneHTTPClient
        self._client: Any | None = None
        self._session_id = ""
        self._turn_id = ""
        self._pending_query = ""
        self._scope_level = "global"
        self._destination = "local"
        self._top_k = 20
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_lock = threading.Lock()
        self._context = ""
        self._injected_memories: list[dict[str, Any]] = []
        self._retrieval_event_id: str | None = None
        self._last_count = 0

    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        return bool(os.environ.get("MNEMOSYNE_API_URL", _DEFAULT_URL))

    def unavailable_reason(self) -> str:
        return "Set MNEMOSYNE_API_URL or start the local MNEMOSYNE service on port 8000."

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "MNEMOSYNE API URL",
                "default": _DEFAULT_URL,
                "env_var": "MNEMOSYNE_API_URL",
            },
            {
                "key": "api_key",
                "description": "Optional API bearer token",
                "secret": True,
                "required": False,
                "env_var": "MNEMOSYNE_API_KEY",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._turn_id = "turn-0"
        self._client = self._client_factory(
            base_url=os.environ.get("MNEMOSYNE_API_URL", _DEFAULT_URL),
            api_key=os.environ.get("MNEMOSYNE_API_KEY", ""),
        )

    def system_prompt_block(self) -> str:
        return (
            "# MNEMOSYNE Memory\n"
            "Relevant memories are injected before each turn. "
            "Use mnemosyne_write_proposals for suggestions; never assume a proposal was committed."
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_id = f"turn-{turn_number}"
        self._pending_query = message
        self._scope_level = str(kwargs.get("scope_level", "global"))
        self._destination = str(kwargs.get("destination", "local"))
        self._top_k = int(kwargs.get("top_k", self._top_k))

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._client is None:
            return
        if session_id:
            self._session_id = session_id
        self._pending_query = query
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._prefetch_thread = spawn_context_thread(
            self._prefetch,
            name="mnemosyne-prefetch",
        )
        if self._prefetch_thread is None:
            self._prefetch()
        else:
            self._prefetch_thread.start()

    def _prefetch(self) -> None:
        query = self._pending_query
        if not query or self._client is None:
            return
        try:
            outcome = self._client.retrieve_store(
                query,
                session_id=self._session_id,
                turn_id=self._turn_id,
                scope_level=self._scope_level,
                destination=self._destination,
                top_k=self._top_k,
            )
            results = outcome.get("results", [])
            context = "\n".join(
                str(item.get("content", "")).strip()
                for item in results
                if item.get("injected", True) and str(item.get("content", "")).strip()
            )
            with self._prefetch_lock:
                self._context = "\n".join(filter(None, ("[MNEMOSYNE Context]", context)))
                self._injected_memories = [
                    {
                        "memory_id": item.get("memory_id"),
                        "content": item.get("content", ""),
                        "rank": item.get("rank", 0),
                    }
                    for item in results
                    if item.get("injected", True)
                ]
                self._retrieval_event_id = outcome.get("retrieval_event_id")
                self._last_count = len(self._injected_memories)
        except Exception:
            with self._prefetch_lock:
                self._context = ""
                self._injected_memories = []
                self._retrieval_event_id = None
                self._last_count = 0

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if session_id:
            self._session_id = session_id
        with self._prefetch_lock:
            context = self._context
            self._context = ""
        return context

    def recall_status(self) -> RecallStatus | None:
        return RecallStatus(self.name, self._last_count) if self._last_count else None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        turn_author: dict[str, Any] | None = None,
    ) -> None:
        """Record citations only when the host explicitly enables Shadow Mode."""
        if self._client is None or not os.environ.get("MNEMOSYNE_SHADOW_MODE") == "1":
            return None
        with self._prefetch_lock:
            event_id = self._retrieval_event_id
            injected = list(self._injected_memories)
        if not event_id or not injected:
            return None
        try:
            citations = self._client.detect_citations(assistant_content, injected).get("citations", [])
            for citation in citations:
                memory_id = citation.get("memory_id")
                match = next((item for item in injected if item.get("memory_id") == memory_id), None)
                if match is None:
                    continue
                self._client.log_citation(
                    {
                        "session_id": session_id or self._session_id,
                        "turn_id": self._turn_id,
                        "retrieval_event_id": event_id,
                        "memory_id": memory_id,
                        "citation": citation,
                        "original_rank": int(match.get("rank", 0)),
                    }
                )
        except Exception:
            return None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "mnemosyne_write_proposals",
                "description": "Generate conservative memory write proposals without committing them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Candidate memory text."},
                        "scope_level": {
                            "type": "string",
                            "enum": ["session", "project", "global"],
                            "default": "project",
                        },
                    },
                    "required": ["text"],
                },
            }
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if tool_name != "mnemosyne_write_proposals" or self._client is None:
            return json.dumps({"error": f"Unknown or unavailable tool: {tool_name}"})
        result = self._client.write_proposals(
            str(args.get("text", "")),
            scope_level=str(args.get("scope_level", "project")),
        )
        return json.dumps(result, ensure_ascii=False)

    def shutdown(self) -> None:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=3)
        self._prefetch_thread = None
        self._client = None


def register(ctx) -> None:
    ctx.register_memory_provider(MnemosyneMemoryProvider())
