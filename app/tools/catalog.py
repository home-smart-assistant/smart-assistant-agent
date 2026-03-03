from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from app.core.models import ToolCall
from app.core.text_codec import EncodingNormalizationError, normalize_payload, normalize_text


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
    "balcony": ("阳台", "balcony", "yang_tai", "yangtai"),
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
        self._catalog_source = "unavailable"
        self._api_endpoints: set[tuple[str, str]] = set()
        self._specs: dict[str, ToolSpec] = {}
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
                self._catalog_source = "unavailable"
                self._specs = {}
                self._api_endpoints = set()
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
        route_agent: str | None,
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
        normalized_agent = route_agent.strip().lower() if isinstance(route_agent, str) else None

        with self._lock:
            specs = sorted(self._specs.values(), key=lambda row: row.tool_name)

        selected: list[str] = []
        for spec in specs:
            if not spec.enabled:
                continue
            if not is_role_allowed(role, spec.tool_name):
                continue
            if normalized_agent and not self._agent_match(spec, normalized_agent):
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

    def allowed_agents_for_tools(self, tool_names: list[str]) -> set[str]:
        names = {name.strip() for name in tool_names if isinstance(name, str) and name.strip()}
        agents: set[str] = set()
        with self._lock:
            for name in names:
                spec = self._specs.get(name)
                if not spec or not spec.enabled:
                    continue
                for agent in spec.allowed_agents:
                    value = str(agent).strip().lower()
                    if value and value != "*":
                        agents.add(value)
        return agents

    def hint_domains_for_tools(self, tool_names: list[str]) -> list[str]:
        names = {name.strip() for name in tool_names if isinstance(name, str) and name.strip()}
        domains: list[str] = []
        with self._lock:
            for name in sorted(names):
                spec = self._specs.get(name)
                if not spec or not spec.enabled:
                    continue
                domain = self._hint_domain_for_spec(spec)
                if domain:
                    domains.append(domain)
        return list(dict.fromkeys(domains))

    def hint_domain_for_tool(self, tool_name: str) -> str | None:
        name = tool_name.strip()
        if not name:
            return None
        with self._lock:
            spec = self._specs.get(name)
        if not spec or not spec.enabled:
            return None
        return self._hint_domain_for_spec(spec)

    def validate(self, tool_call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
        with self._lock:
            spec = self._specs.get(tool_call.tool_name)
        if spec is None or not spec.enabled:
            return None, f"unknown_tool:{tool_call.tool_name}"

        unknown_keys = self._find_unknown_argument_keys(spec, tool_call.arguments)
        if unknown_keys:
            return None, f"invalid_arguments:unknown_argument:{unknown_keys[0]}"

        # Strict mode: do not auto-merge catalog defaults into runtime arguments.
        # Tool arguments must be explicitly generated by the LLM and pass schema validation.
        merged = dict(tool_call.arguments)
        delay_raw = merged.get("delay_seconds")
        if delay_raw is not None:
            try:
                delay_seconds = float(delay_raw)
            except (TypeError, ValueError):
                return None, "invalid_arguments:delay_seconds must be a number"
            if delay_seconds < 0 or delay_seconds > 3600:
                return None, "invalid_arguments:delay_seconds must be between 0 and 3600"
            merged["delay_seconds"] = round(delay_seconds, 3)

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
                normalized_area = area_text.strip().lower()
                if not normalized_area:
                    return None, "invalid_arguments:area must not be empty"
                merged["area"] = normalized_area
            elif "entity_id" not in merged:
                return None, "invalid_arguments:area or entity_id is required"

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

    def _find_unknown_argument_keys(self, spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
        if not isinstance(arguments, dict) or not arguments:
            return []
        schema = spec.parameters_schema if isinstance(spec.parameters_schema, dict) else {}
        if schema.get("additionalProperties") is not False:
            return []
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return sorted(str(key) for key in arguments.keys())
        allowed_keys = {str(key) for key in properties.keys()}
        unknown = [str(key) for key in arguments.keys() if str(key) not in allowed_keys]
        return sorted(unknown)

    def build_rollback_call(self, tool_call: ToolCall) -> ToolCall | None:
        with self._lock:
            spec = self._specs.get(tool_call.tool_name)
            if not spec or not spec.rollback_tool:
                return None
            rollback_spec = self._specs.get(spec.rollback_tool)
            if not rollback_spec or not rollback_spec.enabled:
                return None
        return ToolCall(tool_name=spec.rollback_tool, arguments=dict(tool_call.arguments))

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
            parameters_schema = self._build_parameters_schema(strategy)
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

    def _build_parameters_schema(self, strategy: str) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        if strategy in {"light_area", "cover_area", "climate_area", "climate_area_temperature"}:
            schema["properties"]["area"] = {"type": "string"}
            schema["properties"]["entity_id"] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }
            schema["properties"]["delay_seconds"] = {"type": "number", "minimum": 0, "maximum": 3600}

        if strategy == "scene_id":
            schema["properties"]["scene_id"] = {"type": "string"}
            schema["properties"]["delay_seconds"] = {"type": "number", "minimum": 0, "maximum": 3600}
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

    def _hint_domain_for_spec(self, spec: ToolSpec) -> str | None:
        strategy = spec.strategy.strip().lower()
        if strategy == "light_area":
            return "light"
        if strategy == "cover_area":
            return "cover"
        if strategy in {"climate_area", "climate_area_temperature"}:
            return "climate"

        domain = spec.domain.strip().lower()
        if domain in {"light", "cover", "climate"}:
            return domain
        return None

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

    def _detect_area_explicit(self, text: str) -> str | None:
        known = self._known_areas()
        for area, words in AREA_ALIAS_GROUPS.items():
            if not any(word in text for word in words):
                continue
            if area in known:
                return area
            return area

        for area in sorted(known):
            if area and area in text:
                return area
        return None

    def _known_areas(self) -> set[str]:
        with self._lock:
            specs = [row for row in self._specs.values() if row.enabled]

        areas: set[str] = set(AREA_ALIAS_GROUPS.keys())
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
        return areas
