from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from app.core.models import ToolCall
from app.core.text_codec import EncodingNormalizationError, normalize_payload, normalize_text


LIGHT_CHAR = "灯"
SCENE_CINEMA_WORD = "影院模式"
SCENE_HOME_WORD = "回家模式"
SCENE_GOOD_NIGHT_WORD = "晚安模式"
CLIMATE_WORD = "空调"

LIGHT_ON_KEYWORDS = (
    "开灯",
    "打开灯",
    "开一下灯",
    "把灯打开",
    "帮我打开灯",
    "turn on",
    "lights on",
)
LIGHT_OFF_KEYWORDS = (
    "关灯",
    "关闭灯",
    "关掉灯",
    "把灯关掉",
    "帮我关灯",
    "turn off",
    "lights off",
)
LIGHT_ON_VERBS = ("打开", "开启", "开")
LIGHT_OFF_VERBS = ("关闭", "关掉", "关")
LIGHT_STATE_ON_PHRASES = (
    "灯是开启",
    "灯开启了",
    "灯已经开启",
    "灯开着",
    "灯是开的",
    "灯亮着",
    "light is on",
    "lights are on",
)
LIGHT_STATE_OFF_PHRASES = (
    "灯是关闭",
    "灯关着",
    "灯没开",
    "灯没有开",
    "灯没有开启",
    "灯是灭的",
    "灯是关的",
    "light is off",
    "lights are off",
)
LIGHT_REQUEST_CUES = (
    "帮我",
    "请",
    "麻烦",
    "把",
    "给我",
    "please",
    "can you",
    "turn",
    "switch",
)
ALL_SCOPE_HINTS = (
    "所有",
    "全部",
    "全屋",
    "全家",
    "整屋",
    "整个家",
    "all lights",
    "all the lights",
    "every light",
    "all lamps",
    "all lighting",
    "whole house",
    "entire home",
    "entire house",
)
EXCEPT_SCOPE_HINTS = (
    "except",
    "but not",
    "excluding",
    "exclude",
)
LEAVE_HOME_HINTS = (
    "离开",
    "出门",
    "要走了",
    "外出",
    "leave",
    "going out",
)

CLIMATE_SET_KEYWORDS = ("调到", "设置", "设为", "调成", "调为", "开到")
CLIMATE_ON_KEYWORDS = (
    "开空调",
    "打开空调",
    "开一下空调",
    "开启空调",
    "把空调打开",
    "帮我打开空调",
    "turn on air conditioner",
    "turn on ac",
    "ac on",
)
CLIMATE_OFF_KEYWORDS = (
    "关空调",
    "关闭空调",
    "关掉空调",
    "把空调关掉",
    "帮我关空调",
    "turn off air conditioner",
    "turn off ac",
    "ac off",
)
CLIMATE_ON_VERBS = ("打开", "开启", "开", "启动")
CLIMATE_OFF_VERBS = ("关闭", "关掉", "关", "停止")
CLIMATE_STATE_ON_PHRASES = (
    "空调是开启",
    "空调开启了",
    "空调已经开启",
    "空调开着",
    "空调是开的",
    "ac is on",
    "air conditioner is on",
)
CLIMATE_STATE_OFF_PHRASES = (
    "空调是关闭",
    "空调关着",
    "空调没开",
    "空调没有开",
    "空调是关的",
    "ac is off",
    "air conditioner is off",
)
CLIMATE_REQUEST_CUES = LIGHT_REQUEST_CUES

AREA_WORD = "区域"
AREA_SYNC_HINTS = (
    "区域",
    "房间",
    "区域整理",
    "房间整理",
    "设置区域",
    "设置房间",
    "创建区域",
    "创建房间",
    "area",
    "areas",
)
AREA_SYNC_ACTION_HINTS = (
    "整理",
    "设置",
    "创建",
    "同步",
    "set",
    "setup",
    "create",
    "sync",
)
AREA_DELETE_HINTS = (
    "删掉",
    "删除",
    "清理",
    "无用",
    "delete",
    "remove",
    "cleanup",
)
AREA_AUDIT_HINTS = (
    "检查",
    "排查",
    "审计",
    "归属",
    "归区",
    "未分配",
    "未归属",
    "漏分配",
    "audit",
    "check",
    "inspect",
    "unassigned",
)
AREA_ASSIGN_HINTS = (
    "归类",
    "分配",
    "归位",
    "回写",
    "修复归属",
    "应用建议",
    "批量归类",
    "assign",
    "apply suggestion",
)
AREA_AUDIT_UNAVAILABLE_HINTS = (
    "离线",
    "不可用",
    "offline",
    "unavailable",
)
TARGET_AREA_NAMES = ("玄关", "厨房", "客厅", "主卧", "次卧", "餐厅", "书房", "卫生间", "走廊")
AREA_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "xuan_guan": ("玄关", "xuan_guan", "xuanguan", "entryway", "foyer"),
    "kitchen": ("厨房", "kitchen", "chu_fang", "chufang"),
    "living_room": ("客厅", "living room", "living_room", "livingroom", "ke_ting", "keting"),
    "master_bedroom": ("主卧", "主卧室", "master bedroom", "master_bedroom", "masterbedroom", "zhu_wo", "zhuwo"),
    "guest_bedroom": ("次卧", "次卧室", "guest bedroom", "guest_bedroom", "guestbedroom", "ci_wo", "ciwo"),
    "bedroom": ("卧室", "bedroom"),
    "dining_room": ("餐厅", "dining room", "dining_room", "diningroom", "can_ting", "canting"),
    "study": ("书房", "study", "shu_fang", "shufang"),
    "bathroom": ("卫生间", "浴室", "bathroom", "wc", "toilet", "wei_sheng_jian", "weishengjian"),
    "corridor": ("走廊", "corridor", "hallway", "zou_lang", "zoulang"),
}
AREA_DETECTION_FALLBACKS: dict[str, tuple[str, ...]] = {
    "master_bedroom": ("bedroom",),
    "guest_bedroom": ("bedroom",),
    "bedroom": ("master_bedroom", "guest_bedroom"),
}


@dataclass(frozen=True)
class ToolSpec:
    tool_name: str
    description: str
    strategy: str
    domain: str
    service: str
    enabled: bool
    tool_version: int = 1
    schema_version: str = "1.0"
    permission_level: str = "low"
    environment_tags: tuple[str, ...] = ("home", "prod")
    allowed_agents: tuple[str, ...] = ("home_automation_agent",)
    rollout_percentage: int = 100
    default_arguments: dict[str, Any] = field(default_factory=dict)
    rollback_tool: str | None = None
    parameters_schema: dict[str, Any] = field(default_factory=dict)


class ToolCatalog:
    def __init__(
        self,
        *,
        bridge_url: str,
        timeout_seconds: float,
        refresh_interval_seconds: float = 30.0,
        text_encoding_strict: bool = True,
    ) -> None:
        self._bridge_url = bridge_url.rstrip("/")
        self._timeout_seconds = max(1.0, timeout_seconds)
        self._refresh_interval_seconds = max(5.0, refresh_interval_seconds)
        self._text_encoding_strict = text_encoding_strict
        self._lock = threading.RLock()
        self._last_refresh_at = 0.0
        self._last_refresh_error: str | None = None
        self._catalog_source = "fallback"
        self._api_endpoints: set[tuple[str, str]] = set()
        self._specs: dict[str, ToolSpec] = {}
        self._load_fallback_specs()
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and (now - self._last_refresh_at) < self._refresh_interval_seconds:
                return
            self._last_refresh_at = now

        try:
            tool_rows, api_rows = self._fetch_bridge_catalogs()
            new_specs = self._build_specs_from_rows(tool_rows)
            if not new_specs:
                raise RuntimeError("bridge tool catalog is empty")
            api_endpoints = self._parse_api_endpoints(api_rows)
        except Exception as ex:
            with self._lock:
                self._last_refresh_error = str(ex)
            return

        with self._lock:
            self._specs = new_specs
            self._api_endpoints = api_endpoints
            self._catalog_source = "bridge"
            self._last_refresh_error = None

    def enabled_tool_names(self) -> list[str]:
        with self._lock:
            return sorted(spec.tool_name for spec in self._specs.values() if spec.enabled)

    def list_catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for spec in self._specs.values():
                items.append(
                    {
                        "tool_name": spec.tool_name,
                        "description": spec.description,
                        "strategy": spec.strategy,
                        "enabled": spec.enabled,
                        "tool_version": spec.tool_version,
                        "schema_version": spec.schema_version,
                        "permission_level": spec.permission_level,
                        "environment_tags": list(spec.environment_tags),
                        "allowed_agents": list(spec.allowed_agents),
                        "rollout_percentage": spec.rollout_percentage,
                        "rollback_tool": spec.rollback_tool,
                        "default_arguments": spec.default_arguments,
                        "parameters": spec.parameters_schema,
                    }
                )
            return sorted(items, key=lambda row: str(row["tool_name"]))

    def tool_schemas(
        self,
        *,
        candidate_tool_names: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if candidate_tool_names is not None and not candidate_tool_names:
            return []
        candidate_set = {name.strip() for name in candidate_tool_names or [] if name.strip()}
        max_items = limit if isinstance(limit, int) and limit > 0 else None

        with self._lock:
            result: list[dict[str, Any]] = []
            for spec in sorted(self._specs.values(), key=lambda row: row.tool_name):
                if not spec.enabled:
                    continue
                if candidate_set and spec.tool_name not in candidate_set:
                    continue
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": spec.tool_name,
                            "description": spec.description or spec.tool_name,
                            "parameters": spec.parameters_schema,
                        },
                    }
                )
                if max_items is not None and len(result) >= max_items:
                    break
            return result

    def is_known_tool(self, tool_name: str) -> bool:
        with self._lock:
            spec = self._specs.get(tool_name)
            return bool(spec and spec.enabled)

    def permission_level(self, tool_name: str) -> str:
        with self._lock:
            spec = self._specs.get(tool_name)
            if not spec:
                return "low"
            return spec.permission_level

    def requires_confirmation(self, tool_name: str) -> bool:
        return self.permission_level(tool_name) in {"high", "critical"}

    def detect_explicit_area(self, text: str) -> str | None:
        normalized = text.strip().lower()
        if not normalized:
            return None
        return self._detect_area_explicit(normalized)

    def candidate_tool_names(
        self,
        *,
        route_agent: str,
        role: str,
        runtime_env: str,
        session_id: str,
        user_area: str | None,
        ha_context: dict[str, Any] | None,
        is_role_allowed: Callable[[str, str], bool],
        limit: int,
    ) -> list[str]:
        max_items = max(1, limit)
        normalized_env = runtime_env.strip().lower()
        normalized_agent = route_agent.strip().lower()

        with self._lock:
            specs = sorted(self._specs.values(), key=lambda row: row.tool_name)

        selected: list[str] = []
        for spec in specs:
            if not spec.enabled:
                continue
            if not is_role_allowed(role, spec.tool_name):
                continue
            if not self._agent_match(spec, normalized_agent):
                continue
            if not self._environment_match(spec, normalized_env):
                continue
            if not self._in_rollout(spec, session_id):
                continue
            if not self._context_available(spec, user_area, ha_context):
                continue
            selected.append(spec.tool_name)
            if len(selected) >= max_items:
                break
        return selected

    def validate(self, tool_call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
        with self._lock:
            spec = self._specs.get(tool_call.tool_name)
        if spec is None or not spec.enabled:
            return None, f"unknown_tool:{tool_call.tool_name}"

        merged = dict(spec.default_arguments)
        merged.update(tool_call.arguments)

        strategy = spec.strategy.strip().lower()
        if strategy in {"light_area", "cover_area", "climate_area", "climate_area_temperature"}:
            area = merged.get("area")
            if area is not None:
                try:
                    area_text = normalize_text(
                        str(area),
                        field_path=f"arguments.area.{tool_call.tool_name}",
                        strict=self._text_encoding_strict,
                    )
                except EncodingNormalizationError as ex:
                    return None, f"invalid_text_encoding:{ex.field_path}"
                merged["area"] = area_text.strip().lower()
            elif "entity_id" not in merged:
                merged["area"] = self._default_area()

        if strategy == "scene_id":
            scene_id = str(merged.get("scene_id", "")).strip().lower()
            if not scene_id:
                return None, "invalid_arguments:scene_id is required"
            merged["scene_id"] = scene_id

        if strategy == "climate_area_temperature":
            temp_raw = merged.get("temperature")
            if temp_raw is None:
                return None, "invalid_arguments:temperature is required"
            try:
                temperature = int(temp_raw)
            except (TypeError, ValueError):
                return None, "invalid_arguments:temperature must be an integer"
            if not 16 <= temperature <= 30:
                return None, "invalid_arguments:temperature must be between 16 and 30"
            merged["temperature"] = temperature

        if strategy == "area_sync":
            target_raw = merged.get("target_areas")
            target_values: list[str] = []
            if isinstance(target_raw, str):
                try:
                    normalized_targets = normalize_text(
                        target_raw,
                        field_path=f"arguments.target_areas.{tool_call.tool_name}",
                        strict=self._text_encoding_strict,
                    )
                except EncodingNormalizationError as ex:
                    return None, f"invalid_text_encoding:{ex.field_path}"
                normalized_targets = normalized_targets.replace("，", ",")
                target_values = [item.strip() for item in normalized_targets.split(",") if item.strip()]
            elif isinstance(target_raw, (list, tuple, set)):
                try:
                    normalized_target_raw = normalize_payload(
                        target_raw,
                        field_path=f"arguments.target_areas.{tool_call.tool_name}",
                        strict=self._text_encoding_strict,
                    )
                except EncodingNormalizationError as ex:
                    return None, f"invalid_text_encoding:{ex.field_path}"
                target_values = [str(item).strip() for item in normalized_target_raw if str(item).strip()]
            if not target_values:
                return None, "invalid_arguments:target_areas is required"
            merged["target_areas"] = list(dict.fromkeys(target_values))
            merged["delete_unused"] = self._to_bool(merged.get("delete_unused"), default=True)
            merged["force_delete_in_use"] = self._to_bool(merged.get("force_delete_in_use"), default=False)

        if strategy == "area_audit":
            target_raw = merged.get("target_areas")
            target_values: list[str] = []
            if isinstance(target_raw, str):
                try:
                    normalized_targets = normalize_text(
                        target_raw,
                        field_path=f"arguments.target_areas.{tool_call.tool_name}",
                        strict=self._text_encoding_strict,
                    )
                except EncodingNormalizationError as ex:
                    return None, f"invalid_text_encoding:{ex.field_path}"
                normalized_targets = normalized_targets.replace("，", ",")
                target_values = [item.strip() for item in normalized_targets.split(",") if item.strip()]
            elif isinstance(target_raw, (list, tuple, set)):
                try:
                    normalized_target_raw = normalize_payload(
                        target_raw,
                        field_path=f"arguments.target_areas.{tool_call.tool_name}",
                        strict=self._text_encoding_strict,
                    )
                except EncodingNormalizationError as ex:
                    return None, f"invalid_text_encoding:{ex.field_path}"
                target_values = [str(item).strip() for item in normalized_target_raw if str(item).strip()]
            if not target_values:
                return None, "invalid_arguments:target_areas is required"
            merged["target_areas"] = list(dict.fromkeys(target_values))

            domains_raw = merged.get("domains")
            domains: list[str] = []
            if isinstance(domains_raw, str):
                domains = [item.strip().lower() for item in domains_raw.split(",") if item.strip()]
            elif isinstance(domains_raw, (list, tuple, set)):
                domains = [str(item).strip().lower() for item in domains_raw if str(item).strip()]
            if not domains:
                domains = ["light", "switch", "climate", "cover", "fan"]
            merged["domains"] = list(dict.fromkeys(domains))
            merged["include_unavailable"] = self._to_bool(merged.get("include_unavailable"), default=False)

        if strategy == "area_assign":
            target_raw = merged.get("target_areas")
            target_values: list[str] = []
            if isinstance(target_raw, str):
                try:
                    normalized_targets = normalize_text(
                        target_raw,
                        field_path=f"arguments.target_areas.{tool_call.tool_name}",
                        strict=self._text_encoding_strict,
                    )
                except EncodingNormalizationError as ex:
                    return None, f"invalid_text_encoding:{ex.field_path}"
                normalized_targets = normalized_targets.replace("，", ",")
                target_values = [item.strip() for item in normalized_targets.split(",") if item.strip()]
            elif isinstance(target_raw, (list, tuple, set)):
                try:
                    normalized_target_raw = normalize_payload(
                        target_raw,
                        field_path=f"arguments.target_areas.{tool_call.tool_name}",
                        strict=self._text_encoding_strict,
                    )
                except EncodingNormalizationError as ex:
                    return None, f"invalid_text_encoding:{ex.field_path}"
                target_values = [str(item).strip() for item in normalized_target_raw if str(item).strip()]
            if not target_values:
                return None, "invalid_arguments:target_areas is required"
            merged["target_areas"] = list(dict.fromkeys(target_values))

            domains_raw = merged.get("domains")
            domains: list[str] = []
            if isinstance(domains_raw, str):
                domains = [item.strip().lower() for item in domains_raw.split(",") if item.strip()]
            elif isinstance(domains_raw, (list, tuple, set)):
                domains = [str(item).strip().lower() for item in domains_raw if str(item).strip()]
            if not domains:
                domains = ["light", "switch", "climate", "cover", "fan"]
            merged["domains"] = list(dict.fromkeys(domains))

            merged["include_unavailable"] = self._to_bool(merged.get("include_unavailable"), default=False)
            merged["only_with_suggestion"] = self._to_bool(merged.get("only_with_suggestion"), default=True)
            merged["max_updates"] = self._clamp_positive_int(merged.get("max_updates"), default=200, minimum=1, maximum=2000)

        return merged, None

    def build_rollback_call(self, tool_call: ToolCall) -> ToolCall | None:
        with self._lock:
            spec = self._specs.get(tool_call.tool_name)
            if not spec or not spec.rollback_tool:
                return None
            rollback_spec = self._specs.get(spec.rollback_tool)
            if not rollback_spec or not rollback_spec.enabled:
                return None
        return ToolCall(tool_name=spec.rollback_tool, arguments=dict(tool_call.arguments))

    def detect_tool_call(self, text: str) -> ToolCall | None:
        self.refresh(force=False)
        with self._lock:
            supports_tool_call = ("POST", "/v1/tools/call") in self._api_endpoints
        if not supports_tool_call:
            return None

        normalized = text.strip()
        if not normalized:
            return None
        lower = normalized.lower()
        explicit_area = self._detect_area_explicit(lower)
        area = explicit_area or self._default_area()
        excluded_areas = self._extract_excluded_areas(lower)
        all_scope_requested = self._is_all_scope_requested(lower)
        all_scope_without_area = explicit_area is None and self._is_all_scope_requested(lower)

        area_sync_call = self._detect_area_sync_tool_call(lower)
        if area_sync_call is not None:
            return area_sync_call
        area_assign_call = self._detect_area_assign_tool_call(lower)
        if area_assign_call is not None:
            return area_assign_call
        area_audit_call = self._detect_area_audit_tool_call(lower)
        if area_audit_call is not None:
            return area_audit_call

        if self._has_light_off_intent(lower):
            target = self._find_light_tool(service="turn_off")
            if target:
                action_area = "all" if (all_scope_without_area or (all_scope_requested and excluded_areas)) else area
                args: dict[str, Any] = {"area": action_area}
                if action_area == "all" and excluded_areas:
                    args["exclude_areas"] = excluded_areas
                return ToolCall(tool_name=target.tool_name, arguments=args)
        if self._has_leave_home_light_off_intent(lower):
            target = self._find_light_tool(service="turn_off")
            if target:
                action_area = "all" if (all_scope_without_area or (all_scope_requested and excluded_areas)) else area
                args: dict[str, Any] = {"area": action_area}
                if action_area == "all" and excluded_areas:
                    args["exclude_areas"] = excluded_areas
                return ToolCall(tool_name=target.tool_name, arguments=args)
        if self._has_light_on_intent(lower):
            target = self._find_light_tool(service="turn_on")
            if target:
                action_area = "all" if (all_scope_without_area or (all_scope_requested and excluded_areas)) else area
                args: dict[str, Any] = {"area": action_area}
                if action_area == "all" and excluded_areas:
                    args["exclude_areas"] = excluded_areas
                return ToolCall(tool_name=target.tool_name, arguments=args)

        if SCENE_CINEMA_WORD in lower or SCENE_HOME_WORD in lower or SCENE_GOOD_NIGHT_WORD in lower:
            target = self._find_scene_tool()
            if target:
                if SCENE_CINEMA_WORD in lower:
                    scene_id = "scene.cinema"
                elif SCENE_HOME_WORD in lower:
                    scene_id = "scene.home"
                else:
                    scene_id = "scene.good_night"
                return ToolCall(tool_name=target.tool_name, arguments={"scene_id": scene_id})

        if self._contains_climate_reference(lower):
            if self._contains_any(lower, CLIMATE_SET_KEYWORDS):
                target = self._find_climate_temperature_tool()
                temperature = self._extract_temperature(lower)
                if target and temperature is not None:
                    return ToolCall(
                        tool_name=target.tool_name,
                        arguments={
                            "area": area,
                            "temperature": temperature,
                        },
                    )
            if self._has_climate_off_intent(lower):
                target = self._find_climate_tool(service="turn_off")
                if target:
                    return ToolCall(tool_name=target.tool_name, arguments={"area": area})
            if self._has_leave_home_climate_off_intent(lower):
                target = self._find_climate_tool(service="turn_off")
                if target:
                    return ToolCall(tool_name=target.tool_name, arguments={"area": area})
            if self._has_climate_on_intent(lower):
                target = self._find_climate_tool(service="turn_on")
                if target:
                    return ToolCall(tool_name=target.tool_name, arguments={"area": area})
        return None

    def health_meta(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source": self._catalog_source,
                "tools": len(self._specs),
                "enabled_tools": sum(1 for row in self._specs.values() if row.enabled),
                "high_risk_tools": sum(
                    1 for row in self._specs.values() if row.enabled and row.permission_level in {"high", "critical"}
                ),
                "api_endpoints": len(self._api_endpoints),
                "tool_call_supported": ("POST", "/v1/tools/call") in self._api_endpoints,
                "last_refresh_error": self._last_refresh_error,
            }

    def _fetch_bridge_catalogs(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        headers = {"X-HA-Bridge-Source": "system"}
        with httpx.Client(timeout=self._timeout_seconds, headers=headers) as client:
            tools_resp = client.get(f"{self._bridge_url}/v1/tools/catalog")
            tools_resp.raise_for_status()
            apis_resp = client.get(f"{self._bridge_url}/v1/apis/catalog")
            apis_resp.raise_for_status()

        tools_body = tools_resp.json()
        apis_body = apis_resp.json()
        tools = tools_body.get("tools", []) if isinstance(tools_body, dict) else []
        apis = apis_body.get("apis", []) if isinstance(apis_body, dict) else []
        if not isinstance(tools, list):
            tools = []
        if not isinstance(apis, list):
            apis = []
        return tools, apis

    def _build_specs_from_rows(self, tool_rows: list[dict[str, Any]]) -> dict[str, ToolSpec]:
        rows: list[dict[str, Any]] = []
        for row in tool_rows:
            if not isinstance(row, dict):
                continue
            tool_name = str(row.get("tool_name", "")).strip()
            if not tool_name:
                continue
            rows.append(row)
        if not rows:
            return {}

        rollback_map = self._infer_rollback_pairs([str(row.get("tool_name", "")) for row in rows])
        specs: dict[str, ToolSpec] = {}
        for row in rows:
            tool_name = str(row.get("tool_name", "")).strip()
            default_arguments = row.get("default_arguments", {})
            if not isinstance(default_arguments, dict):
                default_arguments = {}
            strategy = str(row.get("strategy", "passthrough")).strip().lower() or "passthrough"
            enabled = bool(row.get("enabled", True))
            description = str(row.get("description", "")).strip() or f"Execute {tool_name}"
            domain = str(row.get("domain", "")).strip().lower()
            service = str(row.get("service", "")).strip().lower()
            tool_version = self._to_positive_int(row.get("tool_version"), fallback=1)
            schema_version = str(row.get("schema_version", "1.0")).strip() or "1.0"
            permission_level = self._normalize_permission_level(row.get("permission_level", "low"))
            environment_tags = tuple(
                self._parse_string_list(row.get("environment_tags"), fallback=["home", "prod"])
            )
            allowed_agents = tuple(
                self._parse_string_list(row.get("allowed_agents"), fallback=["home_automation_agent"])
            )
            rollout_percentage = self._clamp_percentage(row.get("rollout_percentage"), fallback=100)
            parameters_schema = self._build_parameters_schema(strategy, default_arguments)
            specs[tool_name] = ToolSpec(
                tool_name=tool_name,
                description=description,
                strategy=strategy,
                domain=domain,
                service=service,
                enabled=enabled,
                tool_version=tool_version,
                schema_version=schema_version,
                permission_level=permission_level,
                environment_tags=environment_tags,
                allowed_agents=allowed_agents,
                rollout_percentage=rollout_percentage,
                default_arguments=default_arguments,
                rollback_tool=rollback_map.get(tool_name),
                parameters_schema=parameters_schema,
            )
        return specs

    def _parse_api_endpoints(self, api_rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
        endpoints: set[tuple[str, str]] = set()
        for row in api_rows:
            if not isinstance(row, dict):
                continue
            method = str(row.get("method", "")).strip().upper()
            path = str(row.get("path", "")).strip()
            if method and path:
                endpoints.add((method, path))
        return endpoints

    def _load_fallback_specs(self) -> None:
        fallback_rows = [
            {
                "tool_name": "home.lights.on",
                "description": "Turn on lights by area",
                "strategy": "light_area",
                "domain": "auto",
                "service": "turn_on",
                "enabled": True,
                "default_arguments": {"area": "living_room"},
            },
            {
                "tool_name": "home.lights.off",
                "description": "Turn off lights by area",
                "strategy": "light_area",
                "domain": "auto",
                "service": "turn_off",
                "enabled": True,
                "default_arguments": {"area": "living_room"},
            },
            {
                "tool_name": "home.scene.activate",
                "description": "Activate a scene",
                "strategy": "scene_id",
                "domain": "scene",
                "service": "turn_on",
                "enabled": True,
                "default_arguments": {},
            },
            {
                "tool_name": "home.climate.turn_on",
                "description": "Turn on climate by area",
                "strategy": "climate_area",
                "domain": "climate",
                "service": "turn_on",
                "enabled": True,
                "default_arguments": {"area": "living_room"},
            },
            {
                "tool_name": "home.climate.turn_off",
                "description": "Turn off climate by area",
                "strategy": "climate_area",
                "domain": "climate",
                "service": "turn_off",
                "enabled": True,
                "default_arguments": {"area": "living_room"},
            },
            {
                "tool_name": "home.climate.set_temperature",
                "description": "Set climate temperature by area",
                "strategy": "climate_area_temperature",
                "domain": "climate",
                "service": "set_temperature",
                "enabled": True,
                "default_arguments": {"area": "living_room"},
            },
            {
                "tool_name": "home.areas.sync",
                "description": "Ensure target HA areas and clean unused areas",
                "strategy": "area_sync",
                "domain": "ha",
                "service": "areas_sync",
                "enabled": True,
                "default_arguments": {
                    "target_areas": ["玄关", "厨房", "客厅", "主卧", "次卧", "餐厅", "书房", "卫生间", "走廊"],
                    "delete_unused": True,
                    "force_delete_in_use": False,
                },
            },
            {
                "tool_name": "home.areas.audit",
                "description": "Audit area assignment and suggest target areas for unassigned entities",
                "strategy": "area_audit",
                "domain": "ha",
                "service": "areas_audit",
                "enabled": True,
                "default_arguments": {
                    "target_areas": ["玄关", "厨房", "客厅", "主卧", "次卧", "餐厅", "书房", "卫生间", "走廊"],
                    "domains": ["light", "switch", "climate", "cover", "fan"],
                    "include_unavailable": False,
                },
            },
            {
                "tool_name": "home.areas.assign",
                "description": "Assign unassigned entities to areas based on audit suggestions",
                "strategy": "area_assign",
                "domain": "ha",
                "service": "areas_assign",
                "enabled": True,
                "default_arguments": {
                    "target_areas": ["玄关", "厨房", "客厅", "主卧", "次卧", "餐厅", "书房", "卫生间", "走廊"],
                    "domains": ["light", "switch", "climate", "cover", "fan"],
                    "include_unavailable": False,
                    "only_with_suggestion": True,
                    "max_updates": 200,
                },
            },
        ]
        self._specs = self._build_specs_from_rows(fallback_rows)
        self._api_endpoints = {("POST", "/v1/tools/call")}
        self._catalog_source = "fallback"

    def _build_parameters_schema(self, strategy: str, default_arguments: dict[str, Any]) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        }
        area_enum = self._extract_area_enum(default_arguments)

        if strategy in {"light_area", "cover_area", "climate_area", "climate_area_temperature"}:
            area_schema: dict[str, Any] = {"type": "string"}
            if area_enum:
                area_schema["enum"] = area_enum
            schema["properties"]["area"] = area_schema
            schema["properties"]["entity_id"] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }

        if strategy == "scene_id":
            schema["properties"]["scene_id"] = {"type": "string"}
            schema["required"] = ["scene_id"]

        if strategy == "climate_area_temperature":
            schema["properties"]["temperature"] = {"type": "integer", "minimum": 16, "maximum": 30}
            schema["required"] = ["temperature"]

        if strategy == "area_sync":
            schema["properties"]["target_areas"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
            schema["properties"]["delete_unused"] = {"type": "boolean"}
            schema["properties"]["force_delete_in_use"] = {"type": "boolean"}
            schema["required"] = ["target_areas"]

        if strategy == "area_audit":
            schema["properties"]["target_areas"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
            schema["properties"]["domains"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
            schema["properties"]["include_unavailable"] = {"type": "boolean"}
            schema["required"] = ["target_areas"]

        if strategy == "area_assign":
            schema["properties"]["target_areas"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
            schema["properties"]["domains"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
            schema["properties"]["include_unavailable"] = {"type": "boolean"}
            schema["properties"]["only_with_suggestion"] = {"type": "boolean"}
            schema["properties"]["max_updates"] = {"type": "integer", "minimum": 1, "maximum": 2000}
            schema["required"] = ["target_areas"]

        if strategy == "passthrough":
            return schema
        return schema

    def _extract_area_enum(self, default_arguments: dict[str, Any]) -> list[str]:
        values: list[str] = []
        area = default_arguments.get("area")
        if isinstance(area, str) and area.strip():
            values.append(area.strip().lower())

        area_map = default_arguments.get("area_entity_map")
        if isinstance(area_map, dict):
            for key in area_map.keys():
                normalized = str(key).strip().lower()
                if normalized:
                    values.append(normalized)

        deduped = list(dict.fromkeys(values))
        return deduped

    def _parse_string_list(self, raw: Any, fallback: list[str]) -> list[str]:
        if isinstance(raw, list):
            values = [str(item).strip().lower() for item in raw if str(item).strip()]
            if values:
                return list(dict.fromkeys(values))
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized:
                return [normalized]
        return list(dict.fromkeys(item.strip().lower() for item in fallback if item.strip()))

    def _normalize_permission_level(self, raw: Any) -> str:
        level = str(raw or "low").strip().lower()
        if level in {"low", "medium", "high", "critical"}:
            return level
        return "low"

    def _to_positive_int(self, raw: Any, fallback: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return fallback
        return value if value > 0 else fallback

    def _to_bool(self, raw: Any, *, default: bool) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    def _clamp_positive_int(self, raw: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, value))

    def _clamp_percentage(self, raw: Any, fallback: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return fallback
        return max(0, min(100, value))

    def _agent_match(self, spec: ToolSpec, route_agent: str) -> bool:
        if not spec.allowed_agents:
            return True
        if "*" in spec.allowed_agents:
            return True
        return route_agent in spec.allowed_agents

    def _environment_match(self, spec: ToolSpec, runtime_env: str) -> bool:
        if not spec.environment_tags:
            return True
        if "*" in spec.environment_tags:
            return True
        return runtime_env in spec.environment_tags

    def _in_rollout(self, spec: ToolSpec, session_id: str) -> bool:
        percentage = max(0, min(100, spec.rollout_percentage))
        if percentage >= 100:
            return True
        if percentage <= 0:
            return False
        bucket_source = f"{session_id}:{spec.tool_name}".encode("utf-8")
        digest = hashlib.sha256(bucket_source).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < percentage

    def _context_available(self, spec: ToolSpec, user_area: str | None, ha_context: dict[str, Any] | None) -> bool:
        if not user_area:
            return True
        if spec.strategy not in {"light_area", "cover_area", "climate_area", "climate_area_temperature"}:
            return True
        if not isinstance(ha_context, dict):
            return True
        known_entities = ha_context.get("known_entities")
        target_entities = self._extract_entity_ids_for_area(spec.default_arguments, user_area)
        if not target_entities:
            from_context = self._extract_context_entity_ids(
                known_entities=known_entities,
                strategy=spec.strategy,
                area=user_area,
            )
            if from_context:
                target_entities = from_context

        if not target_entities:
            # No explicit mapping for this area in catalog/context; keep candidate and let bridge
            # resolve entities from HA area metadata at execution time.
            return True
        # Keep action tools visible even if mapped entities look unavailable in context.
        # Runtime execution resolves entities from HA dynamically and is more reliable.
        return True

    def _extract_context_entity_ids(
        self,
        *,
        known_entities: Any,
        strategy: str,
        area: str,
    ) -> list[str] | None:
        if not isinstance(known_entities, dict):
            return None

        entity_type = self._entity_type_for_strategy(strategy)
        if not entity_type:
            return None

        entity_map = known_entities.get(entity_type)
        if not isinstance(entity_map, dict):
            return None

        if area not in entity_map:
            return None
        return self._parse_entity_ids(entity_map.get(area))

    def _entity_type_for_strategy(self, strategy: str) -> str | None:
        mapping = {
            "light_area": "light",
            "cover_area": "cover",
            "climate_area": "climate",
            "climate_area_temperature": "climate",
        }
        return mapping.get(strategy)

    def _extract_entity_ids_for_area(self, default_arguments: dict[str, Any], area: str) -> list[str]:
        area_map = default_arguments.get("area_entity_map")
        if not isinstance(area_map, dict):
            return []
        raw = area_map.get(area)
        return self._parse_entity_ids(raw)

    def _parse_entity_ids(self, raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            values: list[str] = []
            for item in raw:
                values.extend(self._parse_entity_ids(item))
            deduped = [item for item in dict.fromkeys(values) if item]
            return deduped
        text = str(raw).strip()
        if not text:
            return []
        if "," in text:
            values = [item.strip() for item in text.split(",") if item.strip()]
            return list(dict.fromkeys(values))
        return [text]

    def _entity_available(self, state_payload: Any) -> bool:
        if not isinstance(state_payload, dict):
            return True
        if state_payload.get("available") is False:
            return False
        state = str(state_payload.get("state", "")).strip().lower()
        return state not in {"unknown", "unavailable"}

    def _infer_rollback_pairs(self, tool_names: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        names = {name.strip() for name in tool_names if name and name.strip()}
        for name in names:
            if name.endswith(".on"):
                candidate = f"{name[:-3]}.off"
                if candidate in names:
                    mapping[name] = candidate
            elif name.endswith(".off"):
                candidate = f"{name[:-4]}.on"
                if candidate in names:
                    mapping[name] = candidate
            elif name.endswith("turn_on"):
                candidate = f"{name[:-7]}turn_off"
                if candidate in names:
                    mapping[name] = candidate
            elif name.endswith("turn_off"):
                candidate = f"{name[:-8]}turn_on"
                if candidate in names:
                    mapping[name] = candidate
        return mapping

    def _find_light_tool(self, *, service: str) -> ToolSpec | None:
        with self._lock:
            specs = [row for row in self._specs.values() if row.enabled]
        preferred = [
            row
            for row in specs
            if row.strategy == "light_area" and (row.service == service or f"lights.{service.replace('turn_', '')}" in row.tool_name)
        ]
        if preferred:
            return sorted(preferred, key=lambda row: row.tool_name)[0]

        fallback = [row for row in specs if row.strategy == "light_area"]
        if service == "turn_on":
            fallback = [row for row in fallback if ".on" in row.tool_name or "turn_on" in row.tool_name]
        else:
            fallback = [row for row in fallback if ".off" in row.tool_name or "turn_off" in row.tool_name]
        if fallback:
            return sorted(fallback, key=lambda row: row.tool_name)[0]
        return None

    def _find_scene_tool(self) -> ToolSpec | None:
        with self._lock:
            candidates = [row for row in self._specs.values() if row.enabled and row.strategy == "scene_id"]
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: row.tool_name)[0]

    def _find_climate_temperature_tool(self) -> ToolSpec | None:
        with self._lock:
            candidates = [
                row for row in self._specs.values() if row.enabled and row.strategy == "climate_area_temperature"
            ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: row.tool_name)[0]

    def _find_climate_tool(self, *, service: str) -> ToolSpec | None:
        with self._lock:
            specs = [row for row in self._specs.values() if row.enabled]
        preferred = [
            row
            for row in specs
            if row.strategy == "climate_area"
            and (row.service == service or f"climate.{service.replace('turn_', '')}" in row.tool_name)
        ]
        if preferred:
            return sorted(preferred, key=lambda row: row.tool_name)[0]

        fallback = [row for row in specs if row.strategy == "climate_area"]
        if service == "turn_on":
            fallback = [row for row in fallback if ".on" in row.tool_name or "turn_on" in row.tool_name]
        else:
            fallback = [row for row in fallback if ".off" in row.tool_name or "turn_off" in row.tool_name]
        if fallback:
            return sorted(fallback, key=lambda row: row.tool_name)[0]
        return None

    def _find_area_sync_tool(self) -> ToolSpec | None:
        with self._lock:
            candidates = [row for row in self._specs.values() if row.enabled and row.strategy == "area_sync"]
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: row.tool_name)[0]

    def _find_area_audit_tool(self) -> ToolSpec | None:
        with self._lock:
            candidates = [row for row in self._specs.values() if row.enabled and row.strategy == "area_audit"]
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: row.tool_name)[0]

    def _find_area_assign_tool(self) -> ToolSpec | None:
        with self._lock:
            candidates = [row for row in self._specs.values() if row.enabled and row.strategy == "area_assign"]
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: row.tool_name)[0]

    def _detect_area_sync_tool_call(self, text: str) -> ToolCall | None:
        if not self._contains_any(text, AREA_SYNC_HINTS):
            return None
        if not (self._contains_any(text, AREA_SYNC_ACTION_HINTS) or self._contains_any(text, AREA_DELETE_HINTS)):
            return None
        target_areas = self._extract_target_areas(text)
        if not target_areas:
            return None
        target = self._find_area_sync_tool()
        if target is None:
            return None
        return ToolCall(
            tool_name=target.tool_name,
            arguments={
                "target_areas": target_areas,
                "delete_unused": self._contains_any(text, AREA_DELETE_HINTS),
                "force_delete_in_use": False,
            },
        )

    def _detect_area_assign_tool_call(self, text: str) -> ToolCall | None:
        target_areas = self._extract_target_areas(text)
        if not self._contains_any(text, AREA_SYNC_HINTS) and not target_areas:
            return None
        if not self._contains_any(text, AREA_ASSIGN_HINTS):
            return None
        target = self._find_area_assign_tool()
        if target is None:
            return None
        target_areas = target_areas or list(TARGET_AREA_NAMES)
        return ToolCall(
            tool_name=target.tool_name,
            arguments={
                "target_areas": target_areas,
                "domains": self._extract_area_audit_domains(text),
                "include_unavailable": self._contains_any(text, AREA_AUDIT_UNAVAILABLE_HINTS),
                "only_with_suggestion": True,
                "max_updates": 200,
            },
        )

    def _detect_area_audit_tool_call(self, text: str) -> ToolCall | None:
        target_areas = self._extract_target_areas(text)
        if not self._contains_any(text, AREA_SYNC_HINTS) and not target_areas:
            return None
        if not self._contains_any(text, AREA_AUDIT_HINTS):
            return None
        target = self._find_area_audit_tool()
        if target is None:
            return None
        target_areas = target_areas or list(TARGET_AREA_NAMES)
        return ToolCall(
            tool_name=target.tool_name,
            arguments={
                "target_areas": target_areas,
                "domains": self._extract_area_audit_domains(text),
                "include_unavailable": self._contains_any(text, AREA_AUDIT_UNAVAILABLE_HINTS),
            },
        )

    def _extract_target_areas(self, text: str) -> list[str]:
        targets: list[str] = []
        for area in TARGET_AREA_NAMES:
            if area in text:
                targets.append(area)
        return list(dict.fromkeys(targets))

    def _extract_area_audit_domains(self, text: str) -> list[str]:
        domains: list[str] = []
        if LIGHT_CHAR in text or "light" in text or "lights" in text:
            domains.append("light")
        if "开关" in text or "switch" in text:
            domains.append("switch")
        if CLIMATE_WORD in text or "climate" in text or "air conditioner" in text or "ac" in text:
            domains.append("climate")
        if "窗帘" in text or "curtain" in text or "cover" in text:
            domains.append("cover")
        if "风扇" in text or "fan" in text:
            domains.append("fan")
        if domains:
            return list(dict.fromkeys(domains))
        return ["light", "switch", "climate", "cover", "fan"]

    def _has_light_on_intent(self, text: str) -> bool:
        if not self._contains_light_reference(text):
            return False
        if self._contains_any(text, LIGHT_OFF_KEYWORDS):
            return False
        if self._contains_any(text, LIGHT_ON_KEYWORDS):
            return True
        if self._is_light_state_statement(text) and not self._has_light_request_cue(text):
            return False
        if not self._contains_any(text, LIGHT_ON_VERBS):
            return False
        if self._contains_any(text, LIGHT_STATE_ON_PHRASES) and not self._has_light_request_cue(text):
            return False
        return True

    def _has_light_off_intent(self, text: str) -> bool:
        if not self._contains_light_reference(text):
            return False
        if self._contains_any(text, LIGHT_ON_KEYWORDS):
            return False
        if self._contains_any(text, LIGHT_OFF_KEYWORDS):
            return True
        if self._is_light_state_statement(text) and not self._has_light_request_cue(text):
            return False
        if not self._contains_any(text, LIGHT_OFF_VERBS):
            return False
        if self._contains_any(text, LIGHT_STATE_OFF_PHRASES) and not self._has_light_request_cue(text):
            return False
        return True

    def _has_leave_home_light_off_intent(self, text: str) -> bool:
        if not self._contains_light_reference(text):
            return False
        if not self._contains_any(text, LEAVE_HOME_HINTS):
            return False
        if self._contains_any(text, LIGHT_OFF_KEYWORDS):
            return True
        return self._contains_any(text, LIGHT_STATE_ON_PHRASES)

    def _has_climate_on_intent(self, text: str) -> bool:
        if not self._contains_climate_reference(text):
            return False
        if self._contains_any(text, CLIMATE_OFF_KEYWORDS):
            return False
        if self._contains_any(text, CLIMATE_ON_KEYWORDS):
            return True
        if self._is_climate_state_statement(text) and not self._has_climate_request_cue(text):
            return False
        if not self._contains_any(text, CLIMATE_ON_VERBS):
            return False
        if self._contains_any(text, CLIMATE_STATE_ON_PHRASES) and not self._has_climate_request_cue(text):
            return False
        return True

    def _has_climate_off_intent(self, text: str) -> bool:
        if not self._contains_climate_reference(text):
            return False
        if self._contains_any(text, CLIMATE_ON_KEYWORDS):
            return False
        if self._contains_any(text, CLIMATE_OFF_KEYWORDS):
            return True
        if self._is_climate_state_statement(text) and not self._has_climate_request_cue(text):
            return False
        if not self._contains_any(text, CLIMATE_OFF_VERBS):
            return False
        if self._contains_any(text, CLIMATE_STATE_OFF_PHRASES) and not self._has_climate_request_cue(text):
            return False
        return True

    def _has_leave_home_climate_off_intent(self, text: str) -> bool:
        if not self._contains_climate_reference(text):
            return False
        if not self._contains_any(text, LEAVE_HOME_HINTS):
            return False
        if self._contains_any(text, CLIMATE_OFF_KEYWORDS):
            return True
        return self._contains_any(text, CLIMATE_STATE_ON_PHRASES)

    def _is_light_state_statement(self, text: str) -> bool:
        return self._contains_any(text, LIGHT_STATE_ON_PHRASES + LIGHT_STATE_OFF_PHRASES)

    def _has_light_request_cue(self, text: str) -> bool:
        return self._contains_any(text, LIGHT_REQUEST_CUES)

    def _is_climate_state_statement(self, text: str) -> bool:
        return self._contains_any(text, CLIMATE_STATE_ON_PHRASES + CLIMATE_STATE_OFF_PHRASES)

    def _has_climate_request_cue(self, text: str) -> bool:
        return self._contains_any(text, CLIMATE_REQUEST_CUES)

    def _is_all_scope_requested(self, text: str) -> bool:
        if self._contains_any(text, ALL_SCOPE_HINTS):
            return True
        return bool(re.search(r"\ball\b", text))

    def _extract_excluded_areas(self, text: str) -> list[str]:
        segments: list[str] = []
        zh = re.search(r"(?:除(?:了)?)(.+?)(?:之外|外)", text)
        if zh:
            segment = str(zh.group(1)).strip()
            if segment:
                segments.append(segment)

        en = re.search(r"(?:except|but not|excluding|exclude)\s+(.+)", text)
        if en:
            segment = str(en.group(1)).strip()
            if segment:
                segments.append(segment)

        if not segments and not self._contains_any(text, EXCEPT_SCOPE_HINTS):
            return []

        excluded: list[str] = []
        for segment in segments or [text]:
            for area, aliases in AREA_ALIAS_GROUPS.items():
                if any(alias in segment for alias in aliases):
                    excluded.append(area)
            explicit = self._detect_area_explicit(segment)
            if explicit:
                excluded.append(explicit)
        return list(dict.fromkeys(excluded))

    def _contains_light_reference(self, text: str) -> bool:
        return LIGHT_CHAR in text or "light" in text or "lights" in text

    def _contains_climate_reference(self, text: str) -> bool:
        if CLIMATE_WORD in text or "air conditioner" in text:
            return True
        return re.search(r"\bac\b", text) is not None

    def _contains_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _detect_area(self, text: str) -> str:
        explicit = self._detect_area_explicit(text)
        if explicit:
            return explicit
        return self._default_area()

    def _detect_area_explicit(self, text: str) -> str | None:
        known = self._known_areas()
        for area, words in AREA_ALIAS_GROUPS.items():
            if not any(word in text for word in words):
                continue
            if area in known:
                return area
            for fallback in AREA_DETECTION_FALLBACKS.get(area, ()):
                if fallback in known:
                    return fallback
            return area

        for area in sorted(known):
            if area and area in text:
                return area
        return None

    def _known_areas(self) -> set[str]:
        with self._lock:
            specs = [row for row in self._specs.values() if row.enabled]

        areas: set[str] = set()
        areas.update(AREA_ALIAS_GROUPS.keys())
        for spec in specs:
            default_area = spec.default_arguments.get("area")
            if isinstance(default_area, str) and default_area.strip():
                areas.add(default_area.strip().lower())
            area_map = spec.default_arguments.get("area_entity_map")
            if isinstance(area_map, dict):
                for key in area_map.keys():
                    normalized = str(key).strip().lower()
                    if normalized:
                        areas.add(normalized)
        if not areas:
            areas = {"living_room", "bedroom", "study"}
        return areas

    def _default_area(self) -> str:
        known = self._known_areas()
        if "living_room" in known:
            return "living_room"
        return sorted(known)[0]

    def _extract_temperature(self, text: str) -> int | None:
        numbers = re.findall(r"\d+", text)
        if not numbers:
            return None
        try:
            value = int(numbers[0])
        except ValueError:
            return None
        if 16 <= value <= 30:
            return value
        return None

