from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.core.config import AppConfig
from app.core.models import AgentRespondRequest, ToolCall
from app.llm.providers import LlmResponse, LlmToolCall
from app.runtime.agent_service import AgentRuntimeError, AgentService
from app.tools.catalog import ToolCatalog


class _HappyPathLlmProvider:
    name = "ollama"

    async def chat(self, messages, tools=None, model=None):
        if tools:
            return LlmResponse(
                text="",
                prompt_tokens=12,
                completion_tokens=8,
                tool_calls=[LlmToolCall(tool_name="home.lights.on", arguments={"area": "dining_room"})],
            )
        return LlmResponse(
            text='{"route":"home_automation","confidence":0.93,"reason":"home control"}',
            prompt_tokens=6,
            completion_tokens=5,
        )

    def health_meta(self):
        return {"provider": self.name}


class _TimeoutLlmProvider:
    name = "ollama"

    async def chat(self, messages, tools=None, model=None):
        return LlmResponse(text=None, error="ollama_timeout: simulated")

    def health_meta(self):
        return {"provider": self.name}


class _TextOnlyToolRouterProvider:
    name = "ollama"

    async def chat(self, messages, tools=None, model=None):
        if tools:
            return LlmResponse(text="我认为可以执行", prompt_tokens=9, completion_tokens=5, tool_calls=[])
        return LlmResponse(
            text='{"route":"home_automation","confidence":0.91,"reason":"home control"}',
            prompt_tokens=6,
            completion_tokens=5,
        )

    def health_meta(self):
        return {"provider": self.name}


class _KnowledgeRouteProvider:
    name = "ollama"

    async def chat(self, messages, tools=None, model=None):
        if tools:
            return LlmResponse(text="", prompt_tokens=4, completion_tokens=2, tool_calls=[])
        return LlmResponse(
            text='{"route":"knowledge_qa","confidence":0.90,"reason":"question answering"}',
            prompt_tokens=6,
            completion_tokens=5,
        )

    def health_meta(self):
        return {"provider": self.name}


class _UnknownArgsToolProvider:
    name = "ollama"

    async def chat(self, messages, tools=None, model=None):
        if tools:
            return LlmResponse(
                text="",
                prompt_tokens=11,
                completion_tokens=7,
                tool_calls=[LlmToolCall(tool_name="home.lights.on", arguments={"area": "dining_room", "foo": 1})],
            )
        return LlmResponse(
            text='{"route":"home_automation","confidence":0.92,"reason":"home control"}',
            prompt_tokens=6,
            completion_tokens=5,
        )

    def health_meta(self):
        return {"provider": self.name}


class _RepairableUnknownArgsToolProvider:
    name = "ollama"

    def __init__(self) -> None:
        self._tool_route_count = 0

    async def chat(self, messages, tools=None, model=None):
        if tools:
            self._tool_route_count += 1
            if self._tool_route_count == 1:
                return LlmResponse(
                    text="",
                    prompt_tokens=10,
                    completion_tokens=6,
                    tool_calls=[
                        LlmToolCall(
                            tool_name="home.lights.on",
                            arguments={
                                "area": "dining_room",
                                "area_entity_map": {"living_room": "light.living_room"},
                            },
                        )
                    ],
                )
            return LlmResponse(
                text="",
                prompt_tokens=9,
                completion_tokens=5,
                tool_calls=[LlmToolCall(tool_name="home.lights.on", arguments={"area": "dining_room"})],
            )
        return LlmResponse(
            text='{"route":"home_automation","confidence":0.92,"reason":"home control"}',
            prompt_tokens=6,
            completion_tokens=5,
        )

    def health_meta(self):
        return {"provider": self.name}


class _TimedSequenceRepairProvider:
    name = "ollama"

    def __init__(self) -> None:
        self._tool_route_count = 0

    async def chat(self, messages, tools=None, model=None):
        if tools:
            self._tool_route_count += 1
            if self._tool_route_count == 1:
                return LlmResponse(
                    text="",
                    prompt_tokens=10,
                    completion_tokens=6,
                    tool_calls=[LlmToolCall(tool_name="home.lights.on", arguments={"area": "living_room", "delay_seconds": 3})],
                )
            return LlmResponse(
                text="",
                prompt_tokens=12,
                completion_tokens=8,
                tool_calls=[
                    LlmToolCall(tool_name="home.lights.on", arguments={"area": "living_room"}),
                    LlmToolCall(tool_name="home.lights.off", arguments={"area": "living_room", "delay_seconds": 3}),
                ],
            )
        return LlmResponse(
            text='{"route":"home_automation","confidence":0.93,"reason":"home control"}',
            prompt_tokens=6,
            completion_tokens=5,
        )

    def health_meta(self):
        return {"provider": self.name}


class _MustNotCallLlmProvider:
    name = "ollama"

    async def chat(self, messages, tools=None, model=None):
        raise AssertionError("LLM chat must not be called in fast mode")

    def health_meta(self):
        return {"provider": self.name}


def _test_config() -> AppConfig:
    return AppConfig(
        ha_bridge_url="http://127.0.0.1:1",
        ha_context_enabled=False,
        llm_enabled=True,
        llm_provider="ollama",
        llm_model="deepseek-r1:1.5b",
        llm_timeout_seconds=1.0,
        agent_action_timeout_seconds=0.2,
        agent_max_plan_steps=3,
        agent_candidate_tool_limit=20,
        agent_trace_max_items=50,
    )


def _inject_catalog_with_light_on(service: AgentService) -> None:
    rows = [
        {
            "tool_name": "home.lights.on",
            "description": "Turn on lights by area",
            "strategy": "light_area",
            "domain": "light",
            "service": "turn_on",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        }
    ]
    service.catalog._specs = service.catalog._build_specs_from_rows(rows)
    service.catalog._catalog_source = "bridge"
    service.catalog._api_endpoints = {("POST", "/v1/tools/call")}
    service.catalog.refresh = lambda force=False: None
    service.permissions.set_whitelist(service.catalog.enabled_tool_names())


def _inject_catalog_with_light_on_off(service: AgentService) -> None:
    rows = [
        {
            "tool_name": "home.lights.on",
            "description": "Turn on lights by area",
            "strategy": "light_area",
            "domain": "light",
            "service": "turn_on",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
        {
            "tool_name": "home.lights.off",
            "description": "Turn off lights by area",
            "strategy": "light_area",
            "domain": "light",
            "service": "turn_off",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
    ]
    service.catalog._specs = service.catalog._build_specs_from_rows(rows)
    service.catalog._catalog_source = "bridge"
    service.catalog._api_endpoints = {("POST", "/v1/tools/call")}
    service.catalog.refresh = lambda force=False: None
    service.permissions.set_whitelist(service.catalog.enabled_tool_names())


class TestAgentLlmStrictMode(unittest.IsolatedAsyncioTestCase):
    async def test_agent_mode_llm_disabled_returns_503(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = None
        _inject_catalog_with_light_on(service)

        req = AgentRespondRequest(text="打开餐厅灯", metadata={"interaction_mode": "agent"})
        with self.assertRaises(AgentRuntimeError) as cm:
            await service.respond(req)

        self.assertEqual(503, cm.exception.status_code)
        self.assertEqual("llm_unreachable", cm.exception.payload.get("error_code"))
        self.assertEqual("tool_router", cm.exception.payload.get("stage"))

    async def test_chat_mode_timeout_returns_504(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _TimeoutLlmProvider()

        req = AgentRespondRequest(text="你好", metadata={"interaction_mode": "chat"})
        with self.assertRaises(AgentRuntimeError) as cm:
            await service.respond(req)

        self.assertEqual(504, cm.exception.status_code)
        self.assertEqual("llm_timeout", cm.exception.payload.get("error_code"))
        self.assertEqual("chat", cm.exception.payload.get("stage"))

    async def test_agent_mode_uses_llm_tool_router(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _HappyPathLlmProvider()
        _inject_catalog_with_light_on(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {
                    "tool_name": "home.lights.on",
                    "arguments": {"area": "dining_room"},
                    "success": True,
                    "message": "HA call succeeded",
                    "trace_id": "test-trace",
                    "executed": True,
                }
            ]
        )

        req = AgentRespondRequest(text="打开餐厅灯", metadata={"interaction_mode": "agent"})
        resp = await service.respond(req)

        self.assertEqual("llm_tool_router", resp.source)
        self.assertIsNotNone(resp.tool_call)
        assert resp.tool_call is not None
        self.assertEqual("home.lights.on", resp.tool_call.tool_name)
        self.assertEqual("dining_room", resp.tool_call.arguments.get("area"))
        self.assertEqual("llm_tool_router", resp.source)

    async def test_agent_mode_text_only_router_returns_error(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _TextOnlyToolRouterProvider()
        _inject_catalog_with_light_on(service)

        req = AgentRespondRequest(text="打开餐厅灯", metadata={"interaction_mode": "agent"})
        with self.assertRaises(AgentRuntimeError) as cm:
            await service.respond(req)

        self.assertEqual(502, cm.exception.status_code)
        self.assertEqual("llm_routing_failed", cm.exception.payload.get("error_code"))
        self.assertEqual("tool_router", cm.exception.payload.get("stage"))

    async def test_bridge_catalog_unavailable_returns_no_candidate_error(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _HappyPathLlmProvider()
        service.catalog._specs = {}
        service.catalog._catalog_source = "unavailable"
        service.catalog._last_refresh_error = "bridge_unreachable"
        service.catalog.refresh = lambda force=False: None
        service.permissions.set_whitelist([])

        req = AgentRespondRequest(text="打开餐厅灯", metadata={"interaction_mode": "agent"})
        with self.assertRaises(AgentRuntimeError) as cm:
            await service.respond(req)

        self.assertEqual(502, cm.exception.status_code)
        self.assertEqual("llm_routing_failed", cm.exception.payload.get("error_code"))
        self.assertEqual("tool_router", cm.exception.payload.get("stage"))
        self.assertEqual("no_candidate_tools", cm.exception.payload.get("message"))

    async def test_prevalidation_blocks_unknown_tool_arguments_before_execution(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _UnknownArgsToolProvider()
        _inject_catalog_with_light_on(service)
        service.executor.execute = AsyncMock(return_value=[])

        req = AgentRespondRequest(text="打开餐厅灯", metadata={"interaction_mode": "agent"})
        with self.assertRaises(AgentRuntimeError) as cm:
            await service.respond(req)

        self.assertEqual(502, cm.exception.status_code)
        self.assertEqual("llm_routing_failed", cm.exception.payload.get("error_code"))
        self.assertEqual("tool_router", cm.exception.payload.get("stage"))
        self.assertIn("invalid_tool_call:home.lights.on", cm.exception.payload.get("message", ""))
        service.executor.execute.assert_not_called()

    async def test_prevalidation_repair_reroutes_and_executes(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _RepairableUnknownArgsToolProvider()
        _inject_catalog_with_light_on(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {
                    "tool_name": "home.lights.on",
                    "arguments": {"area": "dining_room"},
                    "success": True,
                    "message": "HA call succeeded",
                    "trace_id": "test-trace",
                    "executed": True,
                }
            ]
        )

        req = AgentRespondRequest(text="打开餐厅灯", metadata={"interaction_mode": "agent"})
        resp = await service.respond(req)

        self.assertEqual("llm_tool_router", resp.source)
        assert resp.tool_call is not None
        self.assertEqual("home.lights.on", resp.tool_call.tool_name)
        self.assertEqual({"area": "dining_room"}, resp.tool_call.arguments)
        service.executor.execute.assert_called_once()

    async def test_timed_sequence_repair_requires_opposite_actions(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _TimedSequenceRepairProvider()
        _inject_catalog_with_light_on_off(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {
                    "tool_name": "home.lights.on",
                    "arguments": {"area": "living_room"},
                    "success": True,
                    "message": "HA call succeeded",
                    "trace_id": "test-trace",
                    "executed": True,
                },
                {
                    "tool_name": "home.lights.off",
                    "arguments": {"area": "living_room", "delay_seconds": 3},
                    "success": True,
                    "message": "HA call succeeded",
                    "trace_id": "test-trace",
                    "executed": True,
                },
            ]
        )

        req = AgentRespondRequest(text="打开客厅的灯，3秒后关闭", metadata={"interaction_mode": "agent"})
        resp = await service.respond(req)
        self.assertEqual("llm_tool_router", resp.source)
        self.assertEqual(2, len(resp.tool_results))
        calls = service.executor.execute.call_args.args[0]
        self.assertEqual(2, len(calls))
        self.assertEqual("home.lights.on", calls[0].tool_name)
        self.assertEqual("home.lights.off", calls[1].tool_name)
        self.assertNotIn("delay_seconds", calls[0].arguments)
        self.assertEqual(3, calls[1].arguments.get("delay_seconds"))

    async def test_metadata_route_is_ignored(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _KnowledgeRouteProvider()
        decision, error, _, _ = await service._decide_intent_route(
            session_id="s1",
            text="这个问题是什么",
            metadata={"route": "home_automation"},
        )
        self.assertIsNone(error)
        assert decision is not None
        self.assertEqual("knowledge_qa", decision.route)
        self.assertEqual("llm", decision.source)

    async def test_fast_mode_skips_llm_and_returns_fast_source(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog_with_light_on(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {
                    "tool_name": "home.lights.on",
                    "arguments": {"area": "dining_room"},
                    "success": True,
                    "message": "HA call succeeded",
                    "trace_id": "test-trace",
                    "executed": True,
                }
            ]
        )

        req = AgentRespondRequest(text="打开餐厅灯", metadata={"interaction_mode": "fast"})
        resp = await service.respond(req)
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual("fast", resp.security.get("interaction_mode"))


class TestCatalogBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ToolCatalog(
            bridge_url="http://127.0.0.1:1",
            timeout_seconds=0.2,
            refresh_interval_seconds=3600.0,
        )

    def test_rule_tool_detection_api_removed(self) -> None:
        self.assertFalse(hasattr(self.catalog, "detect_tool_call"))

    def test_detect_explicit_area_with_chinese(self) -> None:
        self.assertEqual("dining_room", self.catalog.detect_explicit_area("我在餐厅"))

    def test_validate_rejects_unknown_argument_keys(self) -> None:
        self.catalog._specs = self.catalog._build_specs_from_rows(
            [
                {
                    "tool_name": "home.lights.on",
                    "description": "Turn on lights by area",
                    "strategy": "light_area",
                    "domain": "light",
                    "service": "turn_on",
                    "enabled": True,
                    "default_arguments": {},
                }
            ]
        )
        args, error = self.catalog.validate(ToolCall(tool_name="home.lights.on", arguments={"area": "study", "foo": 1}))
        self.assertIsNone(args)
        self.assertEqual("invalid_arguments:unknown_argument:foo", error)

    def test_validate_no_default_area_fallback(self) -> None:
        self.catalog._specs = self.catalog._build_specs_from_rows(
            [
                {
                    "tool_name": "home.lights.off",
                    "description": "Turn off lights by area",
                    "strategy": "light_area",
                    "domain": "light",
                    "service": "turn_off",
                    "enabled": True,
                    "default_arguments": {},
                }
            ]
        )
        args, error = self.catalog.validate(ToolCall(tool_name="home.lights.off", arguments={}))
        self.assertIsNone(args)
        self.assertEqual("invalid_arguments:area or entity_id is required", error)

    def test_validate_does_not_merge_internal_default_arguments(self) -> None:
        self.catalog._specs = self.catalog._build_specs_from_rows(
            [
                {
                    "tool_name": "home.lights.on",
                    "description": "Turn on lights by area",
                    "strategy": "light_area",
                    "domain": "light",
                    "service": "turn_on",
                    "enabled": True,
                    "default_arguments": {
                        "area": "living_room",
                        "area_entity_map": {"living_room": "light.living_room"},
                    },
                }
            ]
        )
        args, error = self.catalog.validate(ToolCall(tool_name="home.lights.on", arguments={"area": "dining_room"}))
        self.assertIsNone(error)
        assert args is not None
        self.assertEqual("dining_room", args.get("area"))
        self.assertNotIn("area_entity_map", args)

    def test_validate_accepts_delay_seconds_for_light_area(self) -> None:
        self.catalog._specs = self.catalog._build_specs_from_rows(
            [
                {
                    "tool_name": "home.lights.off",
                    "description": "Turn off lights by area",
                    "strategy": "light_area",
                    "domain": "light",
                    "service": "turn_off",
                    "enabled": True,
                    "default_arguments": {},
                }
            ]
        )
        args, error = self.catalog.validate(
            ToolCall(tool_name="home.lights.off", arguments={"area": "living_room", "delay_seconds": 3})
        )
        self.assertIsNone(error)
        assert args is not None
        self.assertEqual(3.0, args.get("delay_seconds"))


class TestHttpSemantics(unittest.TestCase):
    def test_http_error_mapping(self) -> None:
        from app.main import app, service

        async def _fake_respond(_req):
            raise AgentRuntimeError(
                status_code=503,
                payload={
                    "error_code": "llm_unreachable",
                    "message": "ollama_unreachable: simulated",
                    "stage": "tool_router",
                    "trace_id": "trace-x",
                    "provider": "ollama",
                    "model": "deepseek-r1:1.5b",
                },
            )

        original = service.respond
        service.respond = _fake_respond
        try:
            client = TestClient(app)
            resp = client.post("/v1/agent/respond", json={"text": "打开餐厅灯", "metadata": {"interaction_mode": "agent"}})
            self.assertEqual(503, resp.status_code)
            body = resp.json()
            self.assertEqual("llm_unreachable", body["detail"]["error_code"])
            self.assertEqual("tool_router", body["detail"]["stage"])
        finally:
            service.respond = original


if __name__ == "__main__":
    unittest.main()
