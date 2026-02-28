from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.action import ActionExecutor
from app.context import HaContextService
from app.core.config import AppConfig
from app.core.models import AgentRespondRequest, AgentRespondResponse, PlanStepView, ToolCall
from app.core.observability import MetricsStore, TraceStore
from app.core.security import PermissionManager, PromptInjectionGuard
from app.llm import build_llm_provider
from app.llm.providers import estimate_message_tokens, estimate_text_tokens
from app.memory import LongTermMemoryService, ShortTermMemory, build_embedding_provider
from app.memory.long_term import MemoryDocument
from app.planning import AgentPlan, PlanStep, Planner
from app.tools import ToolCatalog


logger = logging.getLogger("smart_assistant_agent")
HOME_AUTOMATION_ROUTE = "home_automation"
KNOWLEDGE_QA_ROUTE = "knowledge_qa"
DEVICE_MAINTENANCE_ROUTE = "device_maintenance"
FAMILY_SCHEDULE_ROUTE = "family_schedule"
ROUTE_LABELS = {
    HOME_AUTOMATION_ROUTE,
    KNOWLEDGE_QA_ROUTE,
    DEVICE_MAINTENANCE_ROUTE,
    FAMILY_SCHEDULE_ROUTE,
}
ROUTE_TO_AGENT = {
    HOME_AUTOMATION_ROUTE: "home_automation_agent",
    KNOWLEDGE_QA_ROUTE: "knowledge_agent",
    DEVICE_MAINTENANCE_ROUTE: "device_maintenance_agent",
    FAMILY_SCHEDULE_ROUTE: "family_schedule_agent",
}
HOME_AUTOMATION_KEYWORDS = (
    "\u706f",
    "lights",
    "light",
    "\u7a97\u5e18",
    "curtain",
    "scene",
    "\u573a\u666f",
    "\u7a7a\u8c03",
    "climate",
    "turn on",
    "turn off",
    "\u6253\u5f00",
    "\u5173\u95ed",
)
DEVICE_MAINTENANCE_KEYWORDS = (
    "\u79bb\u7ebf",
    "\u6545\u969c",
    "\u7535\u6c60",
    "\u7ef4\u62a4",
    "\u8bbe\u5907\u72b6\u6001",
    "offline",
    "maintenance",
)
FAMILY_SCHEDULE_KEYWORDS = (
    "\u65e5\u7a0b",
    "\u63d0\u9192",
    "\u95f9\u949f",
    "\u884c\u7a0b",
    "schedule",
    "calendar",
    "remind",
)


@dataclass
class IntentRouteDecision:
    route: str
    confidence: float
    reason: str
    source: str = "rule"


class AgentService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.catalog = ToolCatalog(
            bridge_url=config.ha_bridge_url,
            timeout_seconds=min(config.agent_action_timeout_seconds, 6.0),
        )
        self.short_memory = ShortTermMemory(
            max_turns=config.agent_memory_max_turns,
            token_budget=config.agent_token_budget,
        )

        embedding_provider = build_embedding_provider(config)
        self.long_memory = LongTermMemoryService(config=config, provider=embedding_provider)

        self.context_service = HaContextService(config)
        self.permissions = PermissionManager(
            whitelist=config.agent_tool_whitelist,
            default_role=config.agent_default_role,
        )
        self.guard = PromptInjectionGuard(
            enabled=config.prompt_injection_guard_enabled,
            patterns=config.prompt_injection_patterns,
        )
        self.planner = Planner(
            catalog=self.catalog,
            multi_step_enabled=config.agent_multi_step_enabled,
            max_plan_steps=config.agent_max_plan_steps,
        )
        self.executor = ActionExecutor(
            bridge_url=config.ha_bridge_url,
            timeout_seconds=config.agent_action_timeout_seconds,
            auto_execute=config.agent_tool_auto_execute,
            rollback_on_failure=config.agent_rollback_on_failure,
            catalog=self.catalog,
            permission_manager=self.permissions,
        )
        self.llm_provider = build_llm_provider(config)
        self.traces = TraceStore(max_items=config.agent_trace_max_items)
        self.metrics = MetricsStore()
        self.permissions.set_whitelist(self.catalog.enabled_tool_names())

    async def respond(self, req: AgentRespondRequest) -> AgentRespondResponse:
        session_id = req.session_id or uuid.uuid4().hex
        text = req.text.strip()
        trace_id = uuid.uuid4().hex
        role = self.permissions.resolve_role(req.metadata)

        self.traces.start_trace(trace_id, session_id=session_id, user_text=text)
        self.traces.add_event(trace_id, "perceive.input", {"role": role, "metadata_keys": sorted(req.metadata.keys())})
        self.short_memory.add_turn(session_id, "user", text)
        await self._remember_turn(session_id, text, {"role": "user", **req.metadata}, trace_id)

        blocked, block_reason = self.guard.inspect(text)
        if blocked:
            reply_text = "Request blocked by prompt-injection guard. Please send a direct home-control command."
            self.short_memory.add_turn(session_id, "assistant", reply_text)
            self.metrics.record_request("security_blocked")
            self.metrics.record_error()
            self.traces.add_event(trace_id, "security.blocked", {"reason": block_reason})
            self.traces.finish_trace(trace_id, source="security_blocked", error=block_reason)
            return AgentRespondResponse(
                session_id=session_id,
                reply_text=reply_text,
                source="security_blocked",
                trace_id=trace_id,
                security={"blocked": True, "reason": block_reason, "role": role},
            )

        self.catalog.refresh(force=False)
        self.permissions.set_whitelist(self.catalog.enabled_tool_names())
        self.traces.add_event(trace_id, "catalog.refresh", self.catalog.health_meta())

        ha_context, context_error, from_cache = await self.context_service.fetch(force_refresh=False)
        self.traces.add_event(
            trace_id,
            "context.fetch",
            {"error": context_error, "from_cache": from_cache, "available": isinstance(ha_context, dict)},
        )

        recalled_docs = await self._recall_memory(session_id, text, trace_id)
        recalled_snippets = [doc.text for doc in recalled_docs]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        routing_error: str | None = None
        router_text: str | None = None
        routed_calls: list[ToolCall] = []
        intent_error: str | None = None

        intent_decision, intent_error, intent_prompt_tokens, intent_completion_tokens = await self._decide_intent_route(
            session_id=session_id,
            text=text,
            metadata=req.metadata,
        )
        total_prompt_tokens += intent_prompt_tokens
        total_completion_tokens += intent_completion_tokens

        route_agent = ROUTE_TO_AGENT.get(intent_decision.route, "knowledge_agent")
        user_area = self._resolve_user_area(text, req.metadata)
        candidate_limit = max(1, self.config.agent_candidate_tool_limit)
        candidate_tool_names = self.catalog.candidate_tool_names(
            route_agent=route_agent,
            role=role,
            runtime_env=self.config.agent_runtime_env,
            session_id=session_id,
            user_area=user_area,
            ha_context=ha_context,
            is_role_allowed=self.permissions.is_allowed,
            limit=candidate_limit,
        )
        candidate_tool_schemas = self.catalog.tool_schemas(
            candidate_tool_names=candidate_tool_names,
            limit=candidate_limit,
        )

        self.traces.add_event(
            trace_id,
            "routing.intent",
            {
                "route": intent_decision.route,
                "agent": route_agent,
                "confidence": round(intent_decision.confidence, 4),
                "reason": intent_decision.reason,
                "source": intent_decision.source,
                "error": intent_error,
            },
        )
        self.traces.add_event(
            trace_id,
            "routing.candidates",
            {
                "count": len(candidate_tool_names),
                "limit": candidate_limit,
                "runtime_env": self.config.agent_runtime_env,
                "user_area": user_area,
                "tool_names": candidate_tool_names,
            },
        )

        if self.config.agent_use_llm_tool_router and self.llm_provider and candidate_tool_schemas:
            routed_calls, router_text, routing_error, prompt_used, completion_used = await self._route_tools_with_llm(
                session_id=session_id,
                ha_context=ha_context,
                recalled_snippets=recalled_snippets,
                candidate_tool_schemas=candidate_tool_schemas,
                allowed_tool_names=candidate_tool_names,
            )
            total_prompt_tokens += prompt_used
            total_completion_tokens += completion_used
            self.traces.add_event(
                trace_id,
                "routing.llm",
                {
                    "tool_calls": [row.tool_name for row in routed_calls],
                    "has_text": bool(router_text),
                    "error": routing_error,
                },
            )

        candidate_set = set(candidate_tool_names)
        if routed_calls:
            filtered_calls = [call for call in routed_calls if call.tool_name in candidate_set]
            if len(filtered_calls) != len(routed_calls):
                self.traces.add_event(
                    trace_id,
                    "routing.candidate_reject",
                    {
                        "source": "llm_tool_router",
                        "requested": [call.tool_name for call in routed_calls],
                        "accepted": [call.tool_name for call in filtered_calls],
                    },
                )
            routed_calls = filtered_calls

        if routed_calls:
            plan = self.planner.plan_with_tool_calls(tool_calls=routed_calls)
            planning_source = "llm_tool_router"
        else:
            base_plan = self.planner.plan(text)
            if base_plan.tool_calls:
                filtered_rule_calls = [call for call in base_plan.tool_calls if call.tool_name in candidate_set]
                if len(filtered_rule_calls) != len(base_plan.tool_calls):
                    self.traces.add_event(
                        trace_id,
                        "routing.candidate_reject",
                        {
                            "source": "rule_router",
                            "requested": [call.tool_name for call in base_plan.tool_calls],
                            "accepted": [call.tool_name for call in filtered_rule_calls],
                        },
                    )
                    unresolved = list(base_plan.unresolved_queries)
                    unresolved.append(text)
                    plan = self.planner.plan_with_tool_calls(
                        tool_calls=filtered_rule_calls,
                        unresolved_queries=unresolved,
                    )
                else:
                    plan = base_plan
            else:
                plan = base_plan
            planning_source = "rule_router"

        self.traces.add_event(
            trace_id,
            "planning.done",
            {
                "source": planning_source,
                "tool_calls": [call.tool_name for call in plan.tool_calls],
                "unresolved": len(plan.unresolved_queries),
            },
        )

        tool_results: list[dict[str, Any]] = []
        reply_text: str | None = None
        source = "rule_chat"
        llm_error: str | None = routing_error or intent_error

        if plan.tool_calls:
            tool_results = await self.executor.execute(
                plan.tool_calls,
                trace_id=trace_id,
                role=role,
                metadata=req.metadata,
            )
            self._mark_act_step_status(plan, tool_results)
            self.traces.add_event(
                trace_id,
                "action.executed",
                {
                    "count": len(tool_results),
                    "success": sum(1 for row in tool_results if row.get("success")),
                },
            )
            for row in tool_results:
                if row.get("rollback"):
                    continue
                self.metrics.record_tool(str(row.get("tool_name", "unknown")), bool(row.get("success")))

            llm_reply, llm_error_next, prompt_used, completion_used = await self._chat_with_llm(
                session_id=session_id,
                ha_context=ha_context,
                recalled_snippets=recalled_snippets,
                plan=plan,
                tool_results=tool_results,
            )
            total_prompt_tokens += prompt_used
            total_completion_tokens += completion_used
            if llm_reply:
                reply_text = llm_reply
                source = f"{self.llm_provider.name}_chat" if self.llm_provider else "rule_tool"
            else:
                reply_text = self._render_tool_reply(tool_results)
                source = "rule_tool"
            llm_error = llm_error_next or llm_error

        if reply_text is None:
            if router_text:
                reply_text = router_text
                source = f"{self.llm_provider.name}_chat" if self.llm_provider else "rule_chat"
            else:
                llm_reply, llm_error_next, prompt_used, completion_used = await self._chat_with_llm(
                    session_id=session_id,
                    ha_context=ha_context,
                    recalled_snippets=recalled_snippets,
                    plan=plan,
                    tool_results=tool_results,
                )
                total_prompt_tokens += prompt_used
                total_completion_tokens += completion_used
                if llm_reply:
                    reply_text = llm_reply
                    source = f"{self.llm_provider.name}_chat" if self.llm_provider else "rule_chat"
                elif tool_results:
                    reply_text = self._render_tool_reply(tool_results)
                    source = "rule_tool"
                else:
                    reply_text = self._fallback_chat_reply(session_id, text)
                    source = "rule_chat"
                llm_error = llm_error_next or llm_error

        if total_prompt_tokens > 0 or total_completion_tokens > 0:
            self.metrics.record_tokens(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )
            self.traces.add_event(
                trace_id,
                "llm.tokens",
                {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                },
            )

        if llm_error:
            self.metrics.record_error()
            self.traces.add_event(trace_id, "llm.error", {"error": llm_error})

        self._mark_feedback_step_completed(plan)
        self.short_memory.add_turn(session_id, "assistant", reply_text)
        await self._remember_turn(session_id, reply_text, {"role": "assistant"}, trace_id)

        first_tool_result = next((item for item in tool_results if not item.get("rollback")), None)
        first_tool_call = plan.tool_calls[0] if plan.tool_calls else None
        self.metrics.record_request(source)

        final_error = llm_error or (context_error if context_error not in {None, "disabled"} else None)
        self.traces.finish_trace(trace_id, source=source, error=final_error)
        return AgentRespondResponse(
            session_id=session_id,
            reply_text=reply_text,
            source=source,
            trace_id=trace_id,
            tool_call=first_tool_call,
            tool_result=first_tool_result,
            tool_results=tool_results,
            plan=[self._to_plan_view(step) for step in plan.steps],
            security={
                "blocked": False,
                "role": role,
                "context_error": context_error,
                "route": intent_decision.route,
                "route_confidence": round(intent_decision.confidence, 4),
                "route_agent": route_agent,
                "candidate_tools": candidate_tool_names,
            },
        )

    async def get_ha_context(self, force_refresh: bool) -> dict[str, Any]:
        context, error, from_cache = await self.context_service.fetch(force_refresh=force_refresh)
        return {"enabled": self.config.ha_context_enabled, "from_cache": from_cache, "error": error, "data": context}

    def get_session_memory(self, session_id: str) -> dict[str, Any]:
        turns = [{"role": turn.role, "content": turn.content} for turn in self.short_memory.get_session(session_id)]
        return {"session_id": session_id, "turns": turns}

    def clear_session_memory(self, session_id: str) -> dict[str, Any]:
        self.short_memory.clear_session(session_id)
        return {"session_id": session_id, "cleared": True}

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self.traces.get_trace(trace_id)

    def list_latest_traces(self, limit: int) -> list[dict[str, Any]]:
        return self.traces.latest(limit=limit)

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    def tool_catalog(self) -> list[dict[str, Any]]:
        return self.catalog.list_catalog()

    def health(self) -> dict[str, Any]:
        return {
            "service": self.config.app_name,
            "status": "ok",
            "memory_sessions": self.short_memory.session_count(),
            "ha_bridge_url": self.config.ha_bridge_url,
            "ha_context": self.context_service.health_meta(),
            "llm": {
                "enabled": bool(self.llm_provider),
                "provider": self.llm_provider.name if self.llm_provider else "disabled",
                "config": self.llm_provider.health_meta() if self.llm_provider else {},
            },
            "memory": {
                "short_term": {
                    "max_turns": self.config.agent_memory_max_turns,
                    "token_budget": self.config.agent_token_budget,
                },
                "long_term": self.long_memory.health_meta(),
            },
            "planning": {
                "multi_step_enabled": self.config.agent_multi_step_enabled,
                "max_plan_steps": self.config.agent_max_plan_steps,
                "use_llm_tool_router": self.config.agent_use_llm_tool_router,
                "use_llm_intent_router": self.config.agent_use_llm_intent_router,
                "intent_min_confidence": self.config.agent_intent_min_confidence,
                "candidate_tool_limit": self.config.agent_candidate_tool_limit,
                "runtime_env": self.config.agent_runtime_env,
            },
            "catalog": self.catalog.health_meta(),
            "action": {
                "auto_execute": self.config.agent_tool_auto_execute,
                "rollback_on_failure": self.config.agent_rollback_on_failure,
            },
            "security": self.permissions.health_meta(),
            "observability": {
                "trace_items": self.traces.size(),
                "metrics": self.metrics.snapshot(),
            },
        }

    async def _route_tools_with_llm(
        self,
        session_id: str,
        ha_context: dict[str, Any] | None,
        recalled_snippets: list[str],
        candidate_tool_schemas: list[dict[str, Any]],
        allowed_tool_names: list[str],
    ) -> tuple[list[ToolCall], str | None, str | None, int, int]:
        if not self.llm_provider:
            return [], None, "llm_disabled", 0, 0

        messages = self._build_router_messages(
            session_id=session_id,
            ha_context=ha_context,
            recalled_snippets=recalled_snippets,
        )
        response = await self.llm_provider.chat(messages, tools=candidate_tool_schemas)
        prompt_tokens, completion_tokens = self._token_usage_from_response(messages, response)
        if response.error:
            return [], None, response.error, prompt_tokens, 0

        allowed = {name.strip() for name in allowed_tool_names if name.strip()}
        calls: list[ToolCall] = []
        for call in response.tool_calls:
            if call.tool_name not in allowed or not self.catalog.is_known_tool(call.tool_name):
                continue
            args: dict[str, Any] = {}
            for key, value in call.arguments.items():
                if isinstance(key, str):
                    args[key] = value
            calls.append(ToolCall(tool_name=call.tool_name, arguments=args))
        return calls, response.text, None, prompt_tokens, completion_tokens

    async def _decide_intent_route(
        self,
        session_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> tuple[IntentRouteDecision, str | None, int, int]:
        rule_decision = self._rule_route_decision(text, metadata)
        if not self.config.agent_use_llm_intent_router or not self.llm_provider:
            return rule_decision, None, 0, 0

        llm_decision, llm_error, prompt_tokens, completion_tokens = await self._classify_route_with_llm(
            session_id=session_id,
            text=text,
        )
        if llm_decision is None:
            return rule_decision, llm_error, prompt_tokens, completion_tokens

        min_confidence = max(0.0, min(1.0, float(self.config.agent_intent_min_confidence)))
        if llm_decision.confidence >= min_confidence:
            if self._should_prefer_rule_route(rule_decision, llm_decision, text):
                return rule_decision, llm_error, prompt_tokens, completion_tokens
            return llm_decision, llm_error, prompt_tokens, completion_tokens
        return rule_decision, llm_error, prompt_tokens, completion_tokens

    def _should_prefer_rule_route(
        self,
        rule_decision: IntentRouteDecision,
        llm_decision: IntentRouteDecision,
        text: str,
    ) -> bool:
        if rule_decision.source == "rule_metadata":
            return True

        if rule_decision.route == llm_decision.route:
            return False

        # Keep deterministic home-automation signals from being overridden by noisy LLM classifications.
        if rule_decision.route == HOME_AUTOMATION_ROUTE:
            lower = text.strip().lower()
            if self.catalog.detect_tool_call(lower) is not None:
                return True
            if any(keyword in lower for keyword in HOME_AUTOMATION_KEYWORDS):
                return True

        return False

    def _rule_route_decision(self, text: str, metadata: dict[str, Any]) -> IntentRouteDecision:
        metadata_route = str(metadata.get("route", "")).strip().lower()
        if metadata_route in ROUTE_LABELS:
            return IntentRouteDecision(
                route=metadata_route,
                confidence=1.0,
                reason="route set by metadata",
                source="rule_metadata",
            )

        lower = text.strip().lower()
        if not lower:
            return IntentRouteDecision(
                route=KNOWLEDGE_QA_ROUTE,
                confidence=0.6,
                reason="empty input fallback",
            )

        if self.catalog.detect_tool_call(lower) is not None or any(keyword in lower for keyword in HOME_AUTOMATION_KEYWORDS):
            return IntentRouteDecision(
                route=HOME_AUTOMATION_ROUTE,
                confidence=0.82,
                reason="home automation keywords detected",
            )
        if any(keyword in lower for keyword in DEVICE_MAINTENANCE_KEYWORDS):
            return IntentRouteDecision(
                route=DEVICE_MAINTENANCE_ROUTE,
                confidence=0.74,
                reason="device maintenance keywords detected",
            )
        if any(keyword in lower for keyword in FAMILY_SCHEDULE_KEYWORDS):
            return IntentRouteDecision(
                route=FAMILY_SCHEDULE_ROUTE,
                confidence=0.72,
                reason="family schedule keywords detected",
            )
        return IntentRouteDecision(
            route=KNOWLEDGE_QA_ROUTE,
            confidence=0.66,
            reason="default to knowledge qa",
        )

    async def _classify_route_with_llm(
        self,
        session_id: str,
        text: str,
    ) -> tuple[IntentRouteDecision | None, str | None, int, int]:
        if not self.llm_provider:
            return None, "llm_disabled", 0, 0

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Classify the user intent into exactly one route label. "
                    "Allowed labels: home_automation, knowledge_qa, device_maintenance, family_schedule. "
                    'Return strict JSON only: {"route":"...", "confidence":0.0, "reason":"..."}'
                ),
            },
        ]
        turns = self.short_memory.get_window(session_id, max_history_turns=2)
        for turn in turns:
            if turn.role not in {"user", "assistant"} or not turn.content:
                continue
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": text})

        response = await self.llm_provider.chat(messages)
        prompt_tokens, completion_tokens = self._token_usage_from_response(messages, response)
        if response.error:
            return None, response.error, prompt_tokens, 0
        parsed = self._parse_intent_route_json(response.text or "")
        if parsed is None:
            return None, "intent_route_parse_failed", prompt_tokens, completion_tokens
        return parsed, None, prompt_tokens, completion_tokens

    def _parse_intent_route_json(self, text: str) -> IntentRouteDecision | None:
        payload = self._extract_first_json_object(text)
        if payload is None:
            return None
        route = str(payload.get("route", "")).strip().lower()
        if route not in ROUTE_LABELS:
            return None
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = str(payload.get("reason", "")).strip() or "classified by llm"
        return IntentRouteDecision(
            route=route,
            confidence=confidence,
            reason=reason[:160],
            source="llm",
        )

    def _extract_first_json_object(self, text: str) -> dict[str, Any] | None:
        raw = text.strip()
        if not raw:
            return None
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _resolve_user_area(self, text: str, metadata: dict[str, Any]) -> str | None:
        area_raw = metadata.get("area")
        if isinstance(area_raw, str) and area_raw.strip():
            return area_raw.strip().lower()
        return self.catalog.detect_explicit_area(text)

    async def _chat_with_llm(
        self,
        session_id: str,
        ha_context: dict[str, Any] | None,
        recalled_snippets: list[str],
        plan: AgentPlan,
        tool_results: list[dict[str, Any]],
    ) -> tuple[str | None, str | None, int, int]:
        if not self.llm_provider:
            return None, "llm_disabled", 0, 0

        messages = self._build_llm_messages(
            session_id=session_id,
            ha_context=ha_context,
            recalled_snippets=recalled_snippets,
            plan=plan,
            tool_results=tool_results,
        )
        response = await self.llm_provider.chat(messages)
        prompt_tokens, completion_tokens = self._token_usage_from_response(messages, response)
        if response.error:
            return None, response.error, prompt_tokens, 0
        return response.text, None, prompt_tokens, completion_tokens

    def _build_router_messages(
        self,
        session_id: str,
        ha_context: dict[str, Any] | None,
        recalled_snippets: list[str],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.config.llm_system_prompt}]
        messages.append(
            {
                "role": "system",
                "content": (
                    "If the request is an actionable home control command, call tools. "
                    "If no tool is needed, answer briefly in natural language."
                ),
            }
        )
        context_prompt = self.context_service.build_prompt(ha_context)
        if context_prompt:
            messages.append({"role": "system", "content": context_prompt})
        if recalled_snippets:
            limited = recalled_snippets[: self.config.long_term_memory_top_k]
            memory_prompt = "Long-term memory hints:\n" + "\n".join(f"- {item}" for item in limited)
            messages.append({"role": "system", "content": memory_prompt})
        turns = self.short_memory.get_window(session_id, max_history_turns=self.config.llm_max_history_turns)
        for turn in turns:
            role = turn.role if turn.role in {"user", "assistant"} else "user"
            if turn.content:
                messages.append({"role": role, "content": turn.content})
        return messages

    def _build_llm_messages(
        self,
        session_id: str,
        ha_context: dict[str, Any] | None,
        recalled_snippets: list[str],
        plan: AgentPlan,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.config.llm_system_prompt}]
        messages.append(
            {
                "role": "system",
                "content": "Follow tool whitelist constraints. Explain failures directly when a tool fails.",
            }
        )

        context_prompt = self.context_service.build_prompt(ha_context)
        if context_prompt:
            messages.append({"role": "system", "content": context_prompt})

        if recalled_snippets:
            limited = recalled_snippets[: self.config.long_term_memory_top_k]
            memory_prompt = "Long-term memory hints:\n" + "\n".join(f"- {item}" for item in limited)
            messages.append({"role": "system", "content": memory_prompt})

        if tool_results:
            compact = []
            for row in tool_results[: self.config.agent_max_plan_steps]:
                compact.append(
                    {
                        "tool_name": row.get("tool_name"),
                        "success": row.get("success"),
                        "message": row.get("message"),
                        "rollback": bool(row.get("rollback", False)),
                    }
                )
            messages.append({"role": "system", "content": "Tool execution result:\n" + json.dumps(compact, ensure_ascii=False)})

        if plan.unresolved_queries:
            unresolved = "\n".join(f"- {item}" for item in plan.unresolved_queries)
            messages.append({"role": "system", "content": "Unresolved sub-queries:\n" + unresolved})

        turns = self.short_memory.get_window(session_id, max_history_turns=self.config.llm_max_history_turns)
        for turn in turns:
            role = turn.role if turn.role in {"user", "assistant"} else "user"
            if turn.content:
                messages.append({"role": role, "content": turn.content})
        return messages

    def _token_usage_from_response(
        self,
        messages: list[dict[str, str]],
        response: Any,
    ) -> tuple[int, int]:
        prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response, "completion_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = estimate_message_tokens(messages)
        text = getattr(response, "text", None)
        if completion_tokens <= 0 and isinstance(text, str) and text:
            completion_tokens = estimate_text_tokens(text)
        return prompt_tokens, completion_tokens

    async def _remember_turn(
        self,
        session_id: str,
        text: str,
        metadata: dict[str, Any],
        trace_id: str,
    ) -> None:
        try:
            await self.long_memory.remember(session_id, text, metadata)
        except Exception as ex:
            logger.warning("long-term remember failed: %s", ex)
            self.traces.add_event(trace_id, "memory.remember.failures", {"error": str(ex)})
            self.metrics.record_memory_remember_failure()
            self.metrics.record_error()

    async def _recall_memory(self, session_id: str, query: str, trace_id: str) -> list[MemoryDocument]:
        started = time.perf_counter()
        try:
            docs = await self.long_memory.recall(session_id, query)
        except Exception as ex:
            logger.warning("long-term recall failed: %s", ex)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.traces.add_event(trace_id, "memory.recall.error", {"error": str(ex)})
            self.traces.add_event(trace_id, "memory.recall.hits", {"value": 0})
            self.traces.add_event(trace_id, "memory.recall.latency_ms", {"value": round(latency_ms, 3)})
            self.metrics.record_memory_recall(hits=0, latency_ms=latency_ms)
            self.metrics.record_error()
            return []

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.traces.add_event(trace_id, "memory.recall.hits", {"value": len(docs)})
        self.traces.add_event(trace_id, "memory.recall.latency_ms", {"value": round(latency_ms, 3)})
        self.metrics.record_memory_recall(hits=len(docs), latency_ms=latency_ms)
        return docs

    def _mark_act_step_status(self, plan: AgentPlan, tool_results: list[dict[str, Any]]) -> None:
        direct_results = [item for item in tool_results if not item.get("rollback")]
        act_steps = [step for step in plan.steps if step.stage == "act"]
        for index, step in enumerate(act_steps):
            if index >= len(direct_results):
                step.status = "skipped"
                continue
            result = direct_results[index]
            executed = bool(result.get("executed", True))
            success = bool(result.get("success", False))
            if not executed:
                step.status = "pending"
            else:
                step.status = "completed" if success else "failed"

    def _mark_feedback_step_completed(self, plan: AgentPlan) -> None:
        for step in plan.steps:
            if step.stage == "feedback":
                step.status = "completed"
            elif step.stage == "decide" and step.status == "pending":
                step.status = "completed"

    def _render_tool_reply(self, tool_results: list[dict[str, Any]]) -> str:
        if not tool_results:
            return "Tool command detected, waiting for execution."

        direct_results = [row for row in tool_results if not row.get("rollback")]
        if not direct_results:
            direct_results = tool_results

        if len(direct_results) == 1:
            row = direct_results[0]
            if row.get("success"):
                return f"Done: {row.get('tool_name', 'tool')}."
            return f"Tool execution failed: {row.get('message', 'unknown error')}"

        success_rows = [row for row in direct_results if row.get("success")]
        failed_rows = [row for row in direct_results if not row.get("success")]
        if not failed_rows:
            return f"Executed {len(direct_results)} actions, all succeeded."
        failed_names = ", ".join(str(row.get("tool_name", "unknown")) for row in failed_rows)
        return (
            f"Executed {len(direct_results)} actions, succeeded {len(success_rows)}, "
            f"failed {len(failed_rows)} ({failed_names})."
        )

    def _fallback_chat_reply(self, session_id: str, text: str) -> str:
        turns = len(self.short_memory.get_session(session_id))
        lower = text.lower()
        if any(k in lower for k in ("who are you", "what are you")):
            return "I am your home assistant for conversation and device control."
        if any(k in lower for k in ("network", "online")):
            return "Agent service is online and responding."
        if any(k in lower for k in ("hello", "are you there")):
            return "I am here."
        return f"Received. Current session has {turns} short-term memory turns."

    def _to_plan_view(self, step: PlanStep) -> PlanStepView:
        return PlanStepView(
            step_id=step.step_id,
            stage=step.stage,
            summary=step.summary,
            tool_name=step.tool_call.tool_name if step.tool_call else None,
            status=step.status,
        )
