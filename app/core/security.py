from __future__ import annotations

import re
from typing import Any


class PromptInjectionGuard:
    def __init__(self, enabled: bool, patterns: list[str] | tuple[str, ...]) -> None:
        self._enabled = enabled
        self._patterns = [re.compile(re.escape(item), flags=re.IGNORECASE) for item in patterns if item]

    def inspect(self, text: str) -> tuple[bool, str | None]:
        if not self._enabled:
            return False, None
        normalized = text.strip()
        if not normalized:
            return False, None
        for pattern in self._patterns:
            if pattern.search(normalized):
                return True, f"blocked_by_pattern:{pattern.pattern}"
        return False, None


class PermissionManager:
    def __init__(self, whitelist: tuple[str, ...], default_role: str = "operator") -> None:
        self._whitelist = set(whitelist)
        self._default_role = default_role
        self._role_prefixes: dict[str, tuple[str, ...]] = {
            "viewer": tuple(),
            "operator": ("home.",),
            "admin": ("",),
        }

    def set_whitelist(self, tool_names: list[str] | tuple[str, ...]) -> None:
        self._whitelist = {name.strip() for name in tool_names if str(name).strip()}

    def resolve_role(self, metadata: dict[str, Any]) -> str:
        role = str(metadata.get("role", self._default_role)).strip().lower()
        if role in self._role_prefixes:
            return role
        return self._default_role

    def is_allowed(self, role: str, tool_name: str) -> bool:
        if tool_name not in self._whitelist:
            return False

        prefixes = self._role_prefixes.get(role, self._role_prefixes[self._default_role])
        if not prefixes:
            return False
        if "" in prefixes:
            return True
        return any(tool_name.startswith(prefix) for prefix in prefixes)

    def health_meta(self) -> dict[str, Any]:
        return {
            "default_role": self._default_role,
            "whitelist_size": len(self._whitelist),
            "roles": sorted(self._role_prefixes.keys()),
        }
