from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.core.config import AppConfig


class HaContextService:
    def __init__(self, config: AppConfig) -> None:
        self._enabled = config.ha_context_enabled
        self._bridge_url = config.ha_bridge_url.rstrip("/")
        self._path = config.ha_context_path if config.ha_context_path.startswith("/") else f"/{config.ha_context_path}"
        self._ttl_seconds = config.ha_context_ttl_seconds
        self._timeout_seconds = config.ha_context_timeout_seconds
        self._max_domains = config.ha_context_max_service_domains
        self._max_chars = config.ha_context_max_chars
        self._cache: dict[str, Any] = {"data": None, "fetched_at": 0.0, "last_error": None}

    async def fetch(self, force_refresh: bool = False) -> tuple[dict[str, Any] | None, str | None, bool]:
        if not self._enabled:
            return None, "disabled", False

        now = time.time()
        cached_data = self._cache.get("data")
        cached_at = float(self._cache.get("fetched_at", 0.0) or 0.0)
        is_fresh = cached_data is not None and (now - cached_at) <= self._ttl_seconds
        if is_fresh and not force_refresh:
            return cached_data, None, True

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                resp = await client.get(f"{self._bridge_url}{self._path}")
                if resp.status_code >= 400:
                    error = f"context_status_{resp.status_code}"
                    self._cache["last_error"] = error
                    if cached_data is not None:
                        return cached_data, error, True
                    return None, error, False
                payload = resp.json()
        except Exception as ex:
            error = f"context_unreachable: {ex}"
            self._cache["last_error"] = error
            if cached_data is not None:
                return cached_data, error, True
            return None, error, False

        if not isinstance(payload, dict):
            error = "context_invalid_payload"
            self._cache["last_error"] = error
            if cached_data is not None:
                return cached_data, error, True
            return None, error, False

        self._cache["data"] = payload
        self._cache["fetched_at"] = now
        self._cache["last_error"] = None
        return payload, None, False

    def build_prompt(self, context: dict[str, Any] | None) -> str | None:
        if not isinstance(context, dict):
            return None

        tools = context.get("tool_catalog")
        tool_names: list[str] = []
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    name = str(tool.get("tool_name", "")).strip()
                    if name:
                        tool_names.append(name)

        domains: list[str] = []
        services = context.get("ha_services")
        if isinstance(services, list):
            for row in services[: max(0, self._max_domains)]:
                if isinstance(row, dict):
                    domain = str(row.get("domain", "")).strip()
                    if domain:
                        domains.append(domain)

        snapshot = {
            "ha_connected": bool(context.get("ha_connected", False)),
            "tool_names": sorted(set(tool_names)),
            "known_entities": context.get("known_entities", {}),
            "entity_states": context.get("entity_states", {}),
            "service_domains": sorted(set(domains)),
        }
        raw = json.dumps(snapshot, ensure_ascii=False)
        if len(raw) > self._max_chars:
            raw = raw[: self._max_chars] + "..."
        return "以下是当前 Home Assistant 上下文（可能不完整，优先遵循用户最新指令）：\n" + raw

    def health_meta(self) -> dict[str, Any]:
        fetched_at = float(self._cache.get("fetched_at", 0.0) or 0.0)
        age_seconds = max(0.0, time.time() - fetched_at) if fetched_at > 0 else None
        return {
            "enabled": self._enabled,
            "path": self._path,
            "ttl_seconds": self._ttl_seconds,
            "cached": self._cache.get("data") is not None,
            "cache_age_seconds": age_seconds,
            "last_error": self._cache.get("last_error"),
        }
