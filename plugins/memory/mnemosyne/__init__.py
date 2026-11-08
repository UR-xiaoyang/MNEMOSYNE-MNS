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

    def retrieve_context_recent(
        self,
        *,
        session_id: str,
        project_id: str = "",
        max_tokens: int = 2000,
        tokens_per_memory: int = 200,
    ) -> dict[str, Any]:
        """Retrieve recent chronological conversation context for session mode."""
        return self._request(
            "POST",
            "/context/recent",
            {
                "session_id": session_id,
                "project_id": project_id,
                "max_tokens": max_tokens,
                "tokens_per_memory": tokens_per_memory,
            },
        )

    def retrieve_experimental_global(
        self,
        query: str,
        *,
        max_tokens: int = 1000,
        tokens_per_memory: int = 200,
    ) -> dict[str, Any]:
        """Retrieve approved experimental global memories."""
        return self._request(
            "POST",
            "/experimental/global-memories/retrieve",
            {
                "query": query,
                "max_tokens": max_tokens,
                "tokens_per_memory": tokens_per_memory,
            },
        )

    def retrieve_store(
        self,
        query: str,
        *,
        session_id: str,
        turn_id: str,
        scope_level: str,
        destination: str,
        top_k: int,
        project_id: str = "",
    ) -> dict[str, Any]:
        """Legacy retrieve_store endpoint (kept for compatibility)."""
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
                "project_id": project_id,
            },
        )

    def write_proposals(self, text: str, *, scope_level: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/write_proposals",
            {"text": text, "scope_level": scope_level},
        )

    def record_context_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str,
        turn_id: str,
        project_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/context/turn",
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "project_id": project_id,
                "user_text": user_text,
                "assistant_text": assistant_text,
            },
        )

    def record_context_checkpoint(
        self,
        *,
        session_id: str,
        turn_id: str,
        project_id: str,
        transcript: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/context/checkpoint",
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "project_id": project_id,
                "transcript": transcript,
            },
        )

    def detect_citations(self, response: str, injected_memories: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/detect_citations",
            {
                "response": response,
                "injected_memories": [
                    {"id": item.get("memory_id"), "content": item.get("content", "")}
                    for item in injected_memories
                ],
            },
        )

    def log_citation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/shadow/citations", payload)


class MnemosyneMemoryProvider(MemoryProvider):
    """Use MNEMOSYNE as Hermes' session context provider.

    Default mode (session): chronological conversation context persistence.
    Experimental mode (experimental_global): opt-in cross-conversation memory.
    """

    pre_compress_checkpoint_api_version = 2

    def __init__(self, client_factory: Callable[..., Any] | None = None) -> None:
        self._client_factory = client_factory or MnemosyneHTTPClient
        self._client: Any | None = None
        self._session_id = ""
        self._turn_id = ""
        self._pending_query = ""
        self._memory_mode = "session"
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_lock = threading.Lock()
        self._context = ""
        self._context_query = ""
        self._injected_memories: list[dict[str, Any]] = []
        self._retrieval_event_id: str | None = None
        self._last_count = 0
        self._project_id = ""

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

    def _get_memory_mode(self) -> str:
        """Get memory mode from environment with safe fallback."""
        mode = os.environ.get("MNEMOSYNE_MEMORY_MODE", "session").strip().lower()
        if mode in ("session", "experimental_global"):
            return mode
        # Unknown mode: safe fallback to session
        return "session"

    def _should_auto_recall(self) -> bool:
        """检查是否启用自动检索."""
        return os.environ.get("MNEMOSYNE_AUTO_RECALL", "0") == "1"

    def _should_auto_store(self) -> bool:
        """检查是否启用自动存储."""
        return os.environ.get("MNEMOSYNE_AUTO_STORE", "0") == "1"

    def _extract_key_info(self, user_text: str, assistant_text: str) -> str | None:
        """从对话中提取关键信息用于存储."""
        store_triggers = ["请记住", "记录", "保存", "记下", "项目", "进度", "状态", "完成", "配置", "设置", "部署"]
        for trigger in store_triggers:
            if trigger in user_text:
                if len(assistant_text) < 500:
                    return assistant_text
                else:
                    return f"{user_text[:100]}... → {assistant_text[:300]}..."
        if any(word in user_text for word in ["进度", "状态", "怎么样"]):
            if any(word in assistant_text for word in ["完成", "已", "✅", "成功"]):
                return f"状态更新: {user_text[:50]} → {assistant_text[:200]}"
        return None


    def _should_auto_recall(self) -> bool:
        """检查是否启用自动检索."""
        return os.environ.get("MNEMOSYNE_AUTO_RECALL", "0") == "1"
    
    def _should_auto_store(self) -> bool:
        """检查是否启用自动存储."""
        return os.environ.get("MNEMOSYNE_AUTO_STORE", "0") == "1"
    
    def _extract_key_info(self, user_text: str, assistant_text: str) -> str | None:
        """从对话中提取关键信息用于存储."""
        store_triggers = ["请记住", "记录", "保存", "记下", "项目", "进度", "状态", "完成"]
        for trigger in store_triggers:
            if trigger in user_text:
                if len(assistant_text) < 500:
                    return assistant_text
                else:
                    return f"{user_text[:100]}... → {assistant_text[:300]}..."
        if any(word in user_text for word in ["进度", "状态", "怎么样"]):
            if any(word in assistant_text for word in ["完成", "已", "✅", "成功"]):
                return f"状态更新: {user_text[:50]} → {assistant_text[:200]}"
        return None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._turn_id = "turn-0"
        self._project_id = str(kwargs.get("cwd") or "")
        self._memory_mode = self._get_memory_mode()
        self._client = self._client_factory(
            base_url=os.environ.get("MNEMOSYNE_API_URL", _DEFAULT_URL),
            api_key=os.environ.get("MNEMOSYNE_API_KEY", ""),
        )

    def system_prompt_block(self) -> str:
        if self._memory_mode == "session":
            return (
                "# Conversation Context\n"
                "Previous conversation history is restored at the start of each session."
            )
        else:
            return (
                "# MNEMOSYNE Memory\n"
                "Experimental cross-conversation memory is enabled. "
                "Session context and approved global memories are injected before each turn."
            )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_id = f"turn-{turn_number}"
        self._pending_query = message

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
            if self._memory_mode == "session":
                # Session mode: chronological context only
                outcome = self._client.retrieve_context_recent(
                    session_id=self._session_id,
                    project_id=self._project_id,
                    max_tokens=2000,
                    tokens_per_memory=200,
                )
                results = outcome.get("results", [])
                context_header = "[Previous conversation context]"
                context_body = "\n".join(
                    str(item.get("content", "")).strip()
                    for item in results
                    if str(item.get("content", "")).strip()
                )
                with self._prefetch_lock:
                    self._context = "\n".join(filter(None, (context_header, context_body)))
                    self._context_query = query
                    self._injected_memories = []
                    self._retrieval_event_id = None
                    self._last_count = len(results)
            
            elif self._memory_mode == "experimental_global":
                # Experimental global mode: session context + approved global memories
                session_outcome = self._client.retrieve_context_recent(
                    session_id=self._session_id,
                    project_id=self._project_id,
                    max_tokens=1500,
                    tokens_per_memory=200,
                )
                session_results = session_outcome.get("results", [])
                session_context = "\n".join(
                    str(item.get("content", "")).strip()
                    for item in session_results
                    if str(item.get("content", "")).strip()
                )
                
                global_outcome = self._client.retrieve_experimental_global(
                    query,
                    max_tokens=1000,
                    tokens_per_memory=200,
                )
                global_results = global_outcome.get("results", [])
                global_context = "\n".join(
                    str(item.get("content", "")).strip()
                    for item in global_results
                    if str(item.get("content", "")).strip()
                )
                
                parts = []
                if session_context:
                    parts.append("[Previous conversation context]\n" + session_context)
                if global_context:
                    parts.append("[Experimental cross-conversation memory]\n" + global_context)
                
                with self._prefetch_lock:
                    self._context = "\n\n".join(parts)
                    self._context_query = query
                    self._injected_memories = global_results
                    self._retrieval_event_id = global_outcome.get("retrieval_event_id")
                    self._last_count = len(session_results) + len(global_results)
        
        except Exception:
            with self._prefetch_lock:
                self._context = ""
                self._context_query = ""
                self._injected_memories = []
                self._retrieval_event_id = None
                self._last_count = 0

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if session_id:
            self._session_id = session_id
        requested_query = query.strip()
        with self._prefetch_lock:
            cached_query = self._context_query
        if requested_query and requested_query != cached_query:
            self._pending_query = requested_query
            self._prefetch()
        with self._prefetch_lock:
            context = self._context
            self._context = ""
            self._context_query = ""
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
        """Record this turn as session working context; Shadow requires explicit experimental switch."""
        if self._client is None or not user_content.strip():
            return None
        
        # Always record visible turn to context
        try:
            self._client.record_context_turn(
                user_content,
                assistant_content,
                session_id=session_id or self._session_id,
                turn_id=self._turn_id,
                project_id=self._project_id,
            )
        except Exception:
            pass

        # Shadow/citation only with explicit experimental switch
        if os.environ.get("MNEMOSYNE_EXPERIMENTAL_SHADOW") != "1":
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
        
        # Auto-store if enabled
        if self._should_auto_store() and self._memory_mode == "experimental_global":
            try:
                key_info = self._extract_key_info(user_content, assistant_content)
                if key_info:
                    self._client.write_proposals(key_info, scope_level="session")
            except Exception:
                pass

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Rebind session identity and discard any prefetched context from the old session."""
        if not new_session_id:
            return
        self._session_id = new_session_id
        self._turn_id = "turn-0"
        with self._prefetch_lock:
            self._context = ""
            self._context_query = ""
            self._injected_memories = []
            self._retrieval_event_id = None
            self._last_count = 0
        self._pending_query = ""

    def on_pre_compress(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Checkpoint the current session transcript before Hermes compacts it."""
        if self._client is None or not messages:
            return ""
        transcript = "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in messages
            if isinstance(item, dict) and str(item.get("content", "")).strip()
        )
        if len(transcript) > 12_000:
            transcript = transcript[:5_979] + "\n[...checkpoint transcript truncated...]\n" + transcript[-5_979:]
        try:
            result = self._client.record_context_checkpoint(
                session_id=self._session_id,
                turn_id=self._turn_id,
                project_id=self._project_id,
                transcript=transcript,
            )
        except Exception:
            if kwargs.get("require_checkpoint"):
                raise
            return ""
        return "[MNEMOSYNE Session Checkpoint]\ncheckpoint stored before compaction" if result else ""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        # Only expose proposals tool in experimental_global mode
        if self._memory_mode != "experimental_global":
            return []
        
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
        
        if self._memory_mode != "experimental_global":
            return json.dumps({"error": "mnemosyne_write_proposals is only available in experimental_global mode"})
        
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
