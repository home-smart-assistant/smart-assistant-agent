from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.models import ToolCall
from app.tools.catalog import AREA_ALIAS_GROUPS, ToolCatalog


@dataclass
class FastRouteResult:
    tool_calls: list[ToolCall] = field(default_factory=list)
    reply_text: str | None = None
    matched: bool = False
    reasons: list[str] = field(default_factory=list)
    unresolved_parts: list[str] = field(default_factory=list)
    used_rules: list[str] = field(default_factory=list)
    trace_meta: dict[str, Any] = field(default_factory=dict)


class FastRouter:
    def __init__(self, *, max_calls: int = 12, allow_delay_seconds: bool = True) -> None:
        self.max_calls = max(1, int(max_calls))
        self.allow_delay_seconds = bool(allow_delay_seconds)
        self._area_alias_map = self._build_area_alias_map()

    def route(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
        candidate_tool_names: list[str],
        catalog: ToolCatalog,
    ) -> FastRouteResult:
        started = time.time()
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return FastRouteResult(
                reply_text="Fast mode supports home-control commands only. Try: turn on living room light.",
                reasons=["empty_text"],
                trace_meta={"clauses": [], "latency_ms": 0.0},
            )

        catalog_rows = self._candidate_rows(catalog=catalog, candidate_tool_names=candidate_tool_names)
        clauses = self._split_clauses(normalized_text)
        metadata_area = self._normalize_area(metadata.get("area"))
        calls: list[ToolCall] = []
        reasons: list[str] = []
        unresolved: list[str] = []
        used_rules: list[str] = []
        clause_trace: list[dict[str, Any]] = []

        for clause in clauses:
            if len(calls) >= self.max_calls:
                unresolved.append("Too many actions in one request; truncated.")
                reasons.append("max_calls_exceeded")
                break
            if not clause:
                continue
            parsed = self._parse_clause(
                clause=clause,
                metadata_area=metadata_area,
                catalog_rows=catalog_rows,
            )
            clause_trace.append(parsed.get("trace", {}))
            used_rules.extend(parsed.get("used_rules", []))
            reasons.extend(parsed.get("reasons", []))
            unresolved.extend(parsed.get("unresolved", []))
            for call in parsed.get("tool_calls", []):
                valid_args, error = catalog.validate(call)
                if error or valid_args is None:
                    unresolved.append(f"Invalid args in clause '{clause}': {error or 'unknown'}")
                    reasons.append("validation_failed")
                    continue
                calls.append(ToolCall(tool_name=call.tool_name, arguments=valid_args))
                if len(calls) >= self.max_calls:
                    break

        matched = bool(calls)
        if not matched:
            reply = self._friendly_failure(unresolved)
        elif unresolved:
            reply = "Executed matched actions. Unresolved parts: " + "; ".join(dict.fromkeys(unresolved))[:300]
        else:
            reply = None

        elapsed_ms = round((time.time() - started) * 1000.0, 3)
        return FastRouteResult(
            tool_calls=calls[: self.max_calls],
            reply_text=reply,
            matched=matched,
            reasons=list(dict.fromkeys(reasons)),
            unresolved_parts=list(dict.fromkeys(unresolved)),
            used_rules=list(dict.fromkeys(used_rules)),
            trace_meta={
                "clauses": clauses,
                "clause_details": clause_trace,
                "generated_calls": [row.model_dump(mode="json") for row in calls[: self.max_calls]],
                "latency_ms": elapsed_ms,
            },
        )

    def _parse_clause(
        self,
        *,
        clause: str,
        metadata_area: str | None,
        catalog_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        reasons: list[str] = []
        unresolved: list[str] = []
        used_rules: list[str] = []

        domain = self._detect_domain(clause)
        action = self._detect_action(clause, domain=domain)
        areas = self._detect_areas(clause)
        if not areas and metadata_area:
            areas = [metadata_area]
            used_rules.append("metadata_area_fallback")

        delay_seconds = self._extract_delay_seconds(clause) if self.allow_delay_seconds else None
        brightness = self._extract_percentage(clause)
        temperature = self._extract_temperature(clause)
        scene_id = self._extract_scene_id(clause)

        if not domain:
            unresolved.append(f"Unrecognized domain in clause '{clause}'.")
            reasons.append("domain_not_detected")
        if not action:
            unresolved.append(f"Unrecognized action in clause '{clause}'.")
            reasons.append("action_not_detected")

        tool_calls: list[ToolCall] = []
        if domain and action:
            selected, ambiguous = self._select_tool(catalog_rows=catalog_rows, domain=domain, action=action)
            if ambiguous:
                unresolved.append(f"Ambiguous tool match in clause '{clause}'.")
                reasons.append("ambiguous_tool")
            elif not selected:
                unresolved.append(f"No available tool for clause '{clause}'.")
                reasons.append("no_matching_tool")
            else:
                strategy = str(selected.get("strategy", "")).strip().lower()
                tool_name = str(selected.get("tool_name", "")).strip()
                args_base: dict[str, Any] = {}
                if delay_seconds is not None:
                    args_base["delay_seconds"] = delay_seconds
                if strategy in {
                    "light_area",
                    "cover_area",
                    "climate_area",
                    "climate_area_temperature",
                    "switch_area",
                    "fan_area",
                }:
                    if not areas:
                        unresolved.append(f"Missing area in clause '{clause}'.")
                        reasons.append("area_missing")
                    else:
                        for area in areas:
                            args = dict(args_base)
                            args["area"] = area
                            if strategy == "climate_area_temperature":
                                if temperature is None:
                                    unresolved.append(f"Missing temperature in clause '{clause}'.")
                                    reasons.append("temperature_missing")
                                    continue
                                args["temperature"] = temperature
                            tool_calls.append(ToolCall(tool_name=tool_name, arguments=args))
                elif strategy == "scene_id":
                    if not scene_id:
                        unresolved.append(f"Missing scene id in clause '{clause}'.")
                        reasons.append("scene_missing")
                    else:
                        args = dict(args_base)
                        args["scene_id"] = scene_id
                        tool_calls.append(ToolCall(tool_name=tool_name, arguments=args))
                else:
                    unresolved.append(f"Unsupported strategy in clause '{clause}'.")
                    reasons.append("unsupported_strategy")

                if action == "set_brightness" and brightness is not None:
                    unresolved.append("Brightness value parsed but not supported by current tool schema.")
                    reasons.append("brightness_not_supported")
                if tool_calls:
                    used_rules.append(f"{domain}:{action}")

        return {
            "tool_calls": tool_calls,
            "reasons": reasons,
            "unresolved": unresolved,
            "used_rules": used_rules,
            "trace": {
                "clause": clause,
                "domain": domain,
                "action": action,
                "areas": areas,
                "delay_seconds": delay_seconds,
                "brightness": brightness,
                "temperature": temperature,
                "scene_id": scene_id,
                "tool_calls": [row.model_dump(mode="json") for row in tool_calls],
            },
        }

    def _friendly_failure(self, unresolved: list[str]) -> str:
        if unresolved:
            return "Fast mode could not parse this request: " + "; ".join(dict.fromkeys(unresolved))[:260]
        return "Fast mode supports home-control commands only. Try: turn on living room light."

    def _normalize_text(self, text: str) -> str:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return ""
        normalized = (
            lowered.replace("\uFF0C", ",")
            .replace("\u3002", ",")
            .replace("\uFF1B", ",")
            .replace("\u5E2E\u6211", "")
            .replace("\u8BF7", "")
            .replace("\u4E00\u4E0B", "")
        )
        return re.sub(r"\s+", " ", normalized).strip()

    def _split_clauses(self, text: str) -> list[str]:
        if not text:
            return []
        buffer = text.replace("\u5148", "")
        parts = re.split(r"(?:\u7136\u540E|\u5E76\u4E14|\u540C\u65F6|,|\u518D|\band\b|\bthen\b)", buffer)
        clauses = [part.strip() for part in parts if part and part.strip()]
        return clauses if clauses else [buffer.strip()]

    def _build_area_alias_map(self) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for canonical, aliases in AREA_ALIAS_GROUPS.items():
            alias_map[canonical] = canonical
            for alias in aliases:
                value = str(alias).strip().lower()
                if value:
                    alias_map[value] = canonical
        return alias_map

    def _detect_areas(self, clause: str) -> list[str]:
        found: list[str] = []
        lowered = clause.strip().lower()
        if not lowered:
            return []
        for alias, canonical in self._area_alias_map.items():
            if alias and alias in lowered:
                found.append(canonical)
        return list(dict.fromkeys(found))

    def _normalize_area(self, area: Any) -> str | None:
        if not isinstance(area, str):
            return None
        raw = area.strip().lower()
        if not raw:
            return None
        return self._area_alias_map.get(raw, raw)

    def _detect_domain(self, clause: str) -> str | None:
        if any(
            word in clause
            for word in (
                "\u7A7A\u8C03",
                "\u6E29\u5EA6",
                "climate",
                "temperature",
            )
        ):
            return "climate"
        if any(word in clause for word in ("\u6D74\u9738", "bath_heater", "bath heater")):
            return "switch"
        if any(
            word in clause
            for word in (
                "\u51C0\u5316\u5668",
                "\u7A7A\u6C14\u51C0\u5316",
                "air purifier",
                "purifier",
            )
        ):
            return "fan"
        if any(
            word in clause
            for word in (
                "\u7A97\u5E18",
                "\u7EB1\u5E18",
                "\u5E18\u5B50",
                "\u5377\u5E18",
                "\u767E\u53F6\u5E18",
                "\u906E\u5149\u5E18",
                "cover",
                "curtain",
                "blind",
                "blinds",
                "shade",
                "shades",
            )
        ):
            return "cover"
        if any(word in clause for word in ("\u573A\u666F", "\u6A21\u5F0F", "scene", "mode")):
            return "scene"
        if any(word in clause for word in ("\u706F", "\u4EAE", "light")):
            return "light"
        return None

    def _detect_action(self, clause: str, *, domain: str | None) -> str | None:
        if domain == "scene":
            if any(word in clause for word in ("\u6267\u884C", "\u542F\u52A8", "\u6253\u5F00", "activate")):
                return "activate_scene"
            if "\u6A21\u5F0F" in clause:
                return "activate_scene"
        if domain == "light":
            if any(word in clause for word in ("\u4EAE\u5EA6", "\u8C03\u4EAE", "\u8C03\u6697", "%", "brightness")):
                return "set_brightness"
            if any(word in clause for word in ("\u5173\u95ED", "\u5173\u6389", "\u71C4\u706F", "turn off")):
                return "turn_off"
            if any(word in clause for word in ("\u6253\u5F00", "\u5F00\u542F", "\u5F00\u706F", "turn on", "\u5F00")):
                return "turn_on"
        if domain == "climate":
            if any(
                word in clause
                for word in (
                    "\u6E29\u5EA6",
                    "\u8BBE\u5230",
                    "\u8C03\u5230",
                    "\u5EA6",
                    "temperature",
                    "set",
                )
            ):
                return "set_temperature"
            if any(word in clause for word in ("\u5173\u95ED", "\u5173\u6389", "turn off")):
                return "turn_off"
            if any(word in clause for word in ("\u6253\u5F00", "\u5F00\u542F", "turn on", "\u5F00")):
                return "turn_on"
        if domain == "cover":
            if any(word in clause for word in ("\u5173\u95ED", "\u5173\u4E0A", "\u62C9\u4E0A", "\u964D\u4E0B", "close")):
                return "turn_off"
            if any(word in clause for word in ("\u6253\u5F00", "\u5F00\u542F", "\u62C9\u5F00", "\u5347\u8D77", "open", "\u5F00")):
                return "turn_on"
        if domain in {"switch", "fan"}:
            if any(word in clause for word in ("\u5173\u95ED", "\u5173\u6389", "turn off", "\u5173")):
                return "turn_off"
            if any(word in clause for word in ("\u6253\u5F00", "\u5F00\u542F", "turn on", "\u5F00")):
                return "turn_on"
        return None

    def _extract_percentage(self, clause: str) -> int | None:
        match = re.search(r"(\d{1,3})\s*%", clause)
        if not match:
            return None
        return max(0, min(100, int(match.group(1))))

    def _extract_temperature(self, clause: str) -> int | None:
        match = re.search(r"(\d{2})\s*(?:\u5EA6)", clause)
        if not match:
            match = re.search(r"(?:\u8BBE\u5230)\s*(\d{2})", clause)
        if not match:
            match = re.search(r"(?:set|to)\s*(\d{2})\b", clause)
        if not match:
            match = re.search(r"\b(\d{2})\s*(?:c|celsius)\b", clause)
        if not match:
            return None
        value = int(match.group(1))
        if 16 <= value <= 30:
            return value
        return None

    def _extract_delay_seconds(self, clause: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:\u79D2)\u540E", clause)
        if not match:
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\s*(?:later|after)?", clause)
        if not match:
            match = re.search(r"(?:after)\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)", clause)
        if not match:
            return None
        value = float(match.group(1))
        if value < 0:
            return None
        return round(value, 3)

    def _extract_scene_id(self, clause: str) -> str | None:
        match = re.search(r"([\u4E00-\u9FFF_a-z0-9]{1,24})(?:\u573A\u666F|\u6A21\u5F0F|scene|mode)", clause)
        if not match:
            return None
        raw = match.group(1).strip().lower()
        if not raw:
            return None
        return raw.replace(" ", "_")

    def _candidate_rows(self, *, catalog: ToolCatalog, candidate_tool_names: list[str]) -> list[dict[str, Any]]:
        names = {row.strip() for row in candidate_tool_names if isinstance(row, str) and row.strip()}
        rows = catalog.list_catalog()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tool_name = str(row.get("tool_name", "")).strip()
            if not tool_name or tool_name not in names:
                continue
            if not bool(row.get("enabled", False)):
                continue
            filtered.append(row)
        return filtered

    def _select_tool(
        self,
        *,
        catalog_rows: list[dict[str, Any]],
        domain: str,
        action: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for row in catalog_rows:
            score = self._score_tool(row=row, domain=domain, action=action)
            if score <= 0:
                continue
            tool_name = str(row.get("tool_name", "")).strip()
            scored.append((score, tool_name, row))

        if not scored:
            return None, False
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score = scored[0][0]
        ambiguous = len(scored) > 1 and scored[1][0] == best_score
        return scored[0][2], ambiguous

    def _score_tool(self, *, row: dict[str, Any], domain: str, action: str) -> int:
        tool_name = str(row.get("tool_name", "")).strip().lower()
        strategy = str(row.get("strategy", "")).strip().lower()
        service = str(row.get("service", "")).strip().lower()
        row_domain = str(row.get("domain", "")).strip().lower()

        score = 0
        if row_domain == domain:
            score += 30

        if action == "turn_on":
            if strategy in {"light_area", "cover_area", "climate_area", "switch_area", "fan_area"}:
                score += 20
            if any(word in service for word in ("turn_on", "open")) or any(word in tool_name for word in (".on", ".open")):
                score += 30
        elif action == "turn_off":
            if strategy in {"light_area", "cover_area", "climate_area", "switch_area", "fan_area"}:
                score += 20
            if any(word in service for word in ("turn_off", "close")) or any(word in tool_name for word in (".off", ".close")):
                score += 30
        elif action == "set_temperature":
            if strategy == "climate_area_temperature":
                score += 60
            if "temperature" in service or "set_temperature" in tool_name:
                score += 20
        elif action == "activate_scene":
            if strategy == "scene_id":
                score += 60
            if "activate" in service or "scene" in tool_name:
                score += 20
        elif action == "set_brightness":
            if strategy == "light_area":
                score += 20
            if "brightness" in service or "brightness" in tool_name:
                score += 10
        return score
