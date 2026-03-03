from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class TraceStore:
    def __init__(self, max_items: int = 200) -> None:
        self._max_items = max(20, max_items)
        self._lock = threading.Lock()
        self._items: deque[dict[str, Any]] = deque(maxlen=self._max_items)
        self._index: dict[str, dict[str, Any]] = {}

    def start_trace(self, trace_id: str, session_id: str, user_text: str) -> None:
        item = {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_text": user_text,
            "source": None,
            "created_at": time.time(),
            "finished_at": None,
            "events": [],
            "error": None,
        }
        with self._lock:
            if len(self._items) == self._items.maxlen and self._items:
                oldest = self._items[0]
                self._index.pop(str(oldest.get("trace_id", "")), None)
            self._items.append(item)
            self._index[trace_id] = item

    def add_event(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            trace = self._index.get(trace_id)
            if not trace:
                return
            trace["events"].append(
                {
                    "event": event,
                    "at": time.time(),
                    "payload": payload,
                }
            )

    def finish_trace(self, trace_id: str, source: str, error: str | None = None) -> None:
        with self._lock:
            trace = self._index.get(trace_id)
            if not trace:
                return
            trace["source"] = source
            trace["error"] = error
            trace["finished_at"] = time.time()

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._index.get(trace_id)
            if not trace:
                return None
            return {
                "trace_id": trace["trace_id"],
                "session_id": trace["session_id"],
                "user_text": trace["user_text"],
                "source": trace["source"],
                "created_at": trace["created_at"],
                "finished_at": trace["finished_at"],
                "events": list(trace["events"]),
                "error": trace["error"],
            }

    def latest(self, limit: int = 20) -> list[dict[str, Any]]:
        take = max(1, min(limit, self._max_items))
        with self._lock:
            values = list(self._items)[-take:]
        values.reverse()
        return [
            {
                "trace_id": row.get("trace_id"),
                "session_id": row.get("session_id"),
                "source": row.get("source"),
                "error": row.get("error"),
                "created_at": row.get("created_at"),
                "finished_at": row.get("finished_at"),
            }
            for row in values
        ]

    def size(self) -> int:
        with self._lock:
            return len(self._items)


class MetricsStore:
    def __init__(self) -> None:
        self._started_at = time.time()
        self._lock = threading.Lock()
        self._request_total = 0
        self._request_by_source: dict[str, int] = {}
        self._tool_calls_total = 0
        self._tool_success_total = 0
        self._tool_by_name: dict[str, int] = {}
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._error_total = 0
        self._memory_recall_requests = 0
        self._memory_recall_hits_total = 0
        self._memory_recall_latency_ms_total = 0.0
        self._memory_remember_failures = 0
        self._fast_requests_total = 0
        self._fast_parse_success_total = 0
        self._fast_partial_success_total = 0
        self._fast_parse_latency_ms_total = 0.0
        self._fast_tool_calls_total = 0

    def record_request(self, source: str) -> None:
        with self._lock:
            self._request_total += 1
            self._request_by_source[source] = self._request_by_source.get(source, 0) + 1

    def record_tool(self, tool_name: str, success: bool) -> None:
        with self._lock:
            self._tool_calls_total += 1
            if success:
                self._tool_success_total += 1
            self._tool_by_name[tool_name] = self._tool_by_name.get(tool_name, 0) + 1

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._prompt_tokens += max(0, prompt_tokens)
            self._completion_tokens += max(0, completion_tokens)

    def record_error(self) -> None:
        with self._lock:
            self._error_total += 1

    def record_memory_recall(self, hits: int, latency_ms: float) -> None:
        with self._lock:
            self._memory_recall_requests += 1
            self._memory_recall_hits_total += max(0, int(hits))
            self._memory_recall_latency_ms_total += max(0.0, float(latency_ms))

    def record_memory_remember_failure(self) -> None:
        with self._lock:
            self._memory_remember_failures += 1

    def record_fast_request(
        self,
        *,
        matched: bool,
        partial_success: bool,
        tool_calls_count: int,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._fast_requests_total += 1
            if matched:
                self._fast_parse_success_total += 1
            if partial_success:
                self._fast_partial_success_total += 1
            self._fast_tool_calls_total += max(0, int(tool_calls_count))
            self._fast_parse_latency_ms_total += max(0.0, float(latency_ms))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime = max(0.0, time.time() - self._started_at)
            avg_recall_latency = (
                self._memory_recall_latency_ms_total / self._memory_recall_requests
                if self._memory_recall_requests > 0
                else 0.0
            )
            avg_fast_latency = (
                self._fast_parse_latency_ms_total / self._fast_requests_total
                if self._fast_requests_total > 0
                else 0.0
            )
            return {
                "uptime_seconds": uptime,
                "requests_total": self._request_total,
                "requests_by_source": dict(self._request_by_source),
                "tool_calls_total": self._tool_calls_total,
                "tool_success_total": self._tool_success_total,
                "tool_calls_by_name": dict(self._tool_by_name),
                "token_usage": {
                    "prompt_tokens": self._prompt_tokens,
                    "completion_tokens": self._completion_tokens,
                },
                "memory": {
                    "recall_requests": self._memory_recall_requests,
                    "recall_hits_total": self._memory_recall_hits_total,
                    "recall_latency_ms_total": self._memory_recall_latency_ms_total,
                    "recall_latency_ms_avg": avg_recall_latency,
                    "remember_failures": self._memory_remember_failures,
                },
                "fast_mode": {
                    "fast_requests_total": self._fast_requests_total,
                    "fast_parse_success_total": self._fast_parse_success_total,
                    "fast_partial_success_total": self._fast_partial_success_total,
                    "fast_parse_latency_ms_total": self._fast_parse_latency_ms_total,
                    "fast_parse_latency_ms_avg": avg_fast_latency,
                    "fast_tool_calls_total": self._fast_tool_calls_total,
                },
                "errors_total": self._error_total,
            }
