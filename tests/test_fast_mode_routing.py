from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from app.core.config import AppConfig
from app.core.models import AgentRespondRequest
from app.runtime.agent_service import AgentService


class _MustNotCallLlmProvider:
    name = "ollama"

    async def chat(self, messages, tools=None, model=None):
        raise AssertionError("LLM must not be called in fast mode")

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
        agent_max_plan_steps=5,
        agent_candidate_tool_limit=50,
        agent_trace_max_items=50,
        agent_fast_mode_enabled=True,
        agent_fast_mode_allow_delay_seconds=True,
        agent_fast_mode_max_calls=12,
    )


def _test_config_fast_disabled() -> AppConfig:
    return AppConfig(
        ha_bridge_url="http://127.0.0.1:1",
        ha_context_enabled=False,
        llm_enabled=True,
        llm_provider="ollama",
        llm_model="deepseek-r1:1.5b",
        llm_timeout_seconds=1.0,
        agent_action_timeout_seconds=0.2,
        agent_max_plan_steps=5,
        agent_candidate_tool_limit=50,
        agent_trace_max_items=50,
        agent_fast_mode_enabled=False,
        agent_fast_mode_allow_delay_seconds=True,
        agent_fast_mode_max_calls=12,
    )


def _inject_catalog(service: AgentService) -> None:
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
        {
            "tool_name": "home.climate.set_temperature",
            "description": "Set temperature",
            "strategy": "climate_area_temperature",
            "domain": "climate",
            "service": "set_temperature",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
        {
            "tool_name": "home.covers.open",
            "description": "Open covers by area",
            "strategy": "cover_area",
            "domain": "cover",
            "service": "open_cover",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
        {
            "tool_name": "home.covers.close",
            "description": "Close covers by area",
            "strategy": "cover_area",
            "domain": "cover",
            "service": "close_cover",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
        {
            "tool_name": "home.scene.activate",
            "description": "Activate a scene by scene_id",
            "strategy": "scene_id",
            "domain": "scene",
            "service": "activate",
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


def _inject_ambiguous_light_on_catalog(service: AgentService) -> None:
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
            "tool_name": "home.lights.power_on",
            "description": "Turn on lights by area alt",
            "strategy": "light_area",
            "domain": "light",
            "service": "turn_on",
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


class TestFastModeRouting(unittest.IsolatedAsyncioTestCase):
    async def test_fast_mode_disabled_returns_friendly_response(self) -> None:
        service = AgentService(_test_config_fast_disabled())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(return_value=[])

        resp = await service.respond(
            AgentRespondRequest(text="turn on living room light", metadata={"interaction_mode": "fast"})
        )
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual(0, len(resp.tool_results))
        service.executor.execute.assert_not_called()

    async def test_fast_mode_does_not_call_llm(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(return_value=[])

        await service.respond(AgentRespondRequest(text="turn on living room light", metadata={"interaction_mode": "fast"}))

    async def test_single_light_on(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {
                    "tool_name": "home.lights.on",
                    "arguments": {"area": "living_room"},
                    "success": True,
                    "message": "ok",
                    "trace_id": "trace-x",
                    "executed": True,
                }
            ]
        )

        resp = await service.respond(
            AgentRespondRequest(text="turn on living room light", metadata={"interaction_mode": "fast"})
        )
        self.assertEqual("fast_rule_router", resp.source)
        assert resp.tool_call is not None
        self.assertEqual("home.lights.on", resp.tool_call.tool_name)
        self.assertEqual("living_room", resp.tool_call.arguments.get("area"))

    async def test_batch_multi_area(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {"tool_name": "home.lights.on", "arguments": {"area": "living_room"}, "success": True, "message": "ok"},
                {"tool_name": "home.lights.on", "arguments": {"area": "dining_room"}, "success": True, "message": "ok"},
            ]
        )

        resp = await service.respond(
            AgentRespondRequest(text="turn on living room and dining room light", metadata={"interaction_mode": "fast"})
        )
        self.assertEqual(2, len(resp.tool_results))

    async def test_multi_clause(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {"tool_name": "home.lights.on", "arguments": {"area": "living_room"}, "success": True, "message": "ok"},
                {"tool_name": "home.lights.off", "arguments": {"area": "dining_room"}, "success": True, "message": "ok"},
            ]
        )

        resp = await service.respond(
            AgentRespondRequest(text="turn on living room light, turn off dining room light", metadata={"interaction_mode": "fast"})
        )
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual(2, len(resp.tool_results))

    async def test_temperature_extract(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {
                    "tool_name": "home.climate.set_temperature",
                    "arguments": {"area": "living_room", "temperature": 26},
                    "success": True,
                    "message": "ok",
                }
            ]
        )

        resp = await service.respond(
            AgentRespondRequest(text="set living room climate to 26", metadata={"interaction_mode": "fast"})
        )
        assert resp.tool_call is not None
        self.assertEqual("home.climate.set_temperature", resp.tool_call.tool_name)
        self.assertEqual(26, resp.tool_call.arguments.get("temperature"))

    async def test_delay_extract(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {
                    "tool_name": "home.lights.off",
                    "arguments": {"area": "living_room", "delay_seconds": 3},
                    "success": True,
                    "message": "ok",
                }
            ]
        )

        resp = await service.respond(
            AgentRespondRequest(text="3 seconds later turn off living room light", metadata={"interaction_mode": "fast"})
        )
        assert resp.tool_call is not None
        self.assertEqual("home.lights.off", resp.tool_call.tool_name)
        self.assertEqual(3.0, resp.tool_call.arguments.get("delay_seconds"))

    async def test_missing_area_returns_200_and_no_execute(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(return_value=[])

        resp = await service.respond(AgentRespondRequest(text="turn on light", metadata={"interaction_mode": "fast"}))
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual(0, len(resp.tool_results))
        service.executor.execute.assert_not_called()

    async def test_partial_success(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(
            return_value=[
                {"tool_name": "home.lights.on", "arguments": {"area": "living_room"}, "success": True, "message": "ok"},
            ]
        )

        resp = await service.respond(
            AgentRespondRequest(text="turn on living room light, what is weather", metadata={"interaction_mode": "fast"})
        )
        parse = resp.security.get("fast_parse", {})
        self.assertTrue(bool(parse.get("unresolved_parts")))
        self.assertEqual(1, len(resp.tool_results))

    async def test_ambiguous_tool_returns_200_and_no_execute(self) -> None:
        service = AgentService(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_ambiguous_light_on_catalog(service)
        service.executor.execute = AsyncMock(return_value=[])

        resp = await service.respond(
            AgentRespondRequest(text="turn on living room light", metadata={"interaction_mode": "fast"})
        )
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual(0, len(resp.tool_results))
        service.executor.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
