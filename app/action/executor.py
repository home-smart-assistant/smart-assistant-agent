from __future__ import annotations

from typing import Any

import httpx

from app.core.models import ToolCall
from app.core.security import PermissionManager
from app.tools.catalog import ToolCatalog


class ActionExecutor:
    def __init__(
        self,
        bridge_url: str,
        timeout_seconds: float,
        auto_execute: bool,
        rollback_on_failure: bool,
        catalog: ToolCatalog,
        permission_manager: PermissionManager,
    ) -> None:
        self._bridge_url = bridge_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._auto_execute = auto_execute
        self._rollback_on_failure = rollback_on_failure
        self._catalog = catalog
        self._permission_manager = permission_manager

    async def execute(
        self,
        tool_calls: list[ToolCall],
        trace_id: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not tool_calls:
            return []
        if not self._auto_execute:
            return [
                {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "success": True,
                    "message": "auto_execute_disabled",
                    "trace_id": trace_id,
                    "executed": False,
                }
                for call in tool_calls
            ]

        results: list[dict[str, Any]] = []
        rollback_stack: list[ToolCall] = []
        confirm_high_risk = self._is_high_risk_confirmed(metadata or {})

        for call in tool_calls:
            if not self._permission_manager.is_allowed(role, call.tool_name):
                result = {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "success": False,
                    "message": f"permission_denied:{role}",
                    "trace_id": trace_id,
                    "executed": False,
                }
                results.append(result)
                if self._rollback_on_failure and rollback_stack:
                    results.extend(await self._rollback(rollback_stack, trace_id))
                break

            if self._catalog.requires_confirmation(call.tool_name) and not confirm_high_risk:
                result = {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "success": False,
                    "message": "confirmation_required:high_risk_tool",
                    "trace_id": trace_id,
                    "executed": False,
                }
                results.append(result)
                if self._rollback_on_failure and rollback_stack:
                    results.extend(await self._rollback(rollback_stack, trace_id))
                break

            valid_args, error = self._catalog.validate(call)
            if error:
                result = {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "success": False,
                    "message": error,
                    "trace_id": trace_id,
                    "executed": False,
                }
                results.append(result)
                if self._rollback_on_failure and rollback_stack:
                    results.extend(await self._rollback(rollback_stack, trace_id))
                break

            normalized_call = ToolCall(tool_name=call.tool_name, arguments=valid_args)
            result = await self._execute_single(normalized_call, trace_id)
            results.append(result)

            if result.get("success"):
                rollback_call = self._catalog.build_rollback_call(normalized_call)
                if rollback_call is not None:
                    rollback_stack.append(rollback_call)
                continue

            if self._rollback_on_failure and rollback_stack:
                results.extend(await self._rollback(rollback_stack, trace_id))
            break

        return results

    def _is_high_risk_confirmed(self, metadata: dict[str, Any]) -> bool:
        for key in ("confirm_high_risk", "high_risk_confirmed", "confirmed"):
            value = metadata.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "y", "ok"}:
                return True
        return False

    async def _execute_single(self, tool_call: ToolCall, trace_id: str) -> dict[str, Any]:
        payload = {
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "trace_id": trace_id,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                resp = await client.post(f"{self._bridge_url}/v1/tools/call", json=payload)
                if resp.status_code >= 400:
                    detail = self._extract_bridge_error_detail(resp)
                    message = f"bridge_error:{resp.status_code}"
                    if detail:
                        message = f"{message}:{detail}"
                    return {
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "success": False,
                        "message": message,
                        "trace_id": trace_id,
                        "executed": True,
                    }
                data = resp.json()
        except Exception as ex:
            return {
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "success": False,
                "message": f"bridge_unreachable:{ex}",
                "trace_id": trace_id,
                "executed": True,
            }

        if not isinstance(data, dict):
            return {
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "success": False,
                "message": "invalid_bridge_payload",
                "trace_id": trace_id,
                "executed": True,
            }
        return {
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "success": bool(data.get("success", False)),
            "message": str(data.get("message", "")),
            "trace_id": trace_id,
            "executed": True,
            "data": data,
        }

    def _extract_bridge_error_detail(self, response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except Exception:
            text = response.text.strip()
            return text or None

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail is not None:
                text = str(detail).strip()
                if text:
                    return text
            message = payload.get("message")
            if message is not None:
                text = str(message).strip()
                if text:
                    return text
        return None

    async def _rollback(self, stack: list[ToolCall], trace_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        while stack:
            call = stack.pop()
            result = await self._execute_single(call, trace_id)
            result["rollback"] = True
            results.append(result)
        return results
