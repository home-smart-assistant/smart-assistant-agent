from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

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
        {
            "tool_name": "home.bath_heater.on",
            "description": "Turn on bath heater by area",
            "strategy": "light_area",
            "domain": "switch",
            "service": "turn_on",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
        {
            "tool_name": "home.bath_heater.off",
            "description": "Turn off bath heater by area",
            "strategy": "light_area",
            "domain": "switch",
            "service": "turn_off",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
        {
            "tool_name": "home.air_purifier.on",
            "description": "Turn on purifier by area",
            "strategy": "light_area",
            "domain": "fan",
            "service": "turn_on",
            "enabled": True,
            "default_arguments": {},
            "allowed_agents": ["home_automation_agent"],
            "environment_tags": ["home", "prod"],
        },
        {
            "tool_name": "home.air_purifier.off",
            "description": "Turn off purifier by area",
            "strategy": "light_area",
            "domain": "fan",
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


def _build_service(config: AppConfig) -> AgentService:
    with patch("app.tools.catalog.ToolCatalog.refresh", autospec=True, return_value=None):
        return AgentService(config)


async def _echo_execute(tool_calls, trace_id=None, role=None, metadata=None):
    rows = []
    for call in tool_calls:
        rows.append(
            {
                "tool_name": call.tool_name,
                "arguments": dict(call.arguments),
                "success": True,
                "message": "ok",
                "trace_id": trace_id or "trace-x",
                "executed": True,
            }
        )
    return rows


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
        service = _build_service(_test_config_fast_disabled())
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
        service = _build_service(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(return_value=[])

        await service.respond(AgentRespondRequest(text="turn on living room light", metadata={"interaction_mode": "fast"}))

    async def test_single_light_on(self) -> None:
        service = _build_service(_test_config())
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
        service = _build_service(_test_config())
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
        service = _build_service(_test_config())
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
        service = _build_service(_test_config())
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
        service = _build_service(_test_config())
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
        service = _build_service(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(return_value=[])

        resp = await service.respond(AgentRespondRequest(text="turn on light", metadata={"interaction_mode": "fast"}))
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual(0, len(resp.tool_results))
        service.executor.execute.assert_not_called()

    async def test_partial_success(self) -> None:
        service = _build_service(_test_config())
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
        service = _build_service(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_ambiguous_light_on_catalog(service)
        service.executor.execute = AsyncMock(return_value=[])

        resp = await service.respond(
            AgentRespondRequest(text="turn on living room light", metadata={"interaction_mode": "fast"})
        )
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual(0, len(resp.tool_results))
        service.executor.execute.assert_not_called()

    async def test_cn_area_matrix_light_cases_no_bridge(self) -> None:
        service = _build_service(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(side_effect=_echo_execute)
        cases = [
            ("\u536B\u751F\u95F4", "bathroom"),
            ("\u53A8\u623F", "kitchen"),
            ("\u9910\u5385", "dining_room"),
            ("\u7384\u5173", "xuan_guan"),
            ("\u4E66\u623F", "study"),
            ("\u5BA2\u5385", "living_room"),
            ("\u4E3B\u5367", "master_bedroom"),
            ("\u6B21\u5367", "guest_bedroom"),
            ("\u9633\u53F0", "balcony"),
            ("\u8D70\u5ECA", "corridor"),
        ]
        for area_cn, area_key in cases:
            with self.subTest(area=area_cn):
                req = AgentRespondRequest(
                    text=f"\u6253\u5F00{area_cn}\u706F",
                    metadata={"interaction_mode": "fast"},
                )
                resp = await service.respond(req)
                self.assertEqual("fast_rule_router", resp.source)
                self.assertTrue(resp.tool_results)
                self.assertEqual("home.lights.on", resp.tool_results[0]["tool_name"])
                self.assertEqual(area_key, resp.tool_results[0]["arguments"].get("area"))

    async def test_cn_device_cases_bath_heater_and_purifier(self) -> None:
        service = _build_service(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(side_effect=_echo_execute)
        cases = [
            ("\u6253\u5F00\u536B\u751F\u95F4\u6D74\u9738", "home.bath_heater.on", "bathroom"),
            ("\u5173\u95ED\u536B\u751F\u95F4\u6D74\u9738", "home.bath_heater.off", "bathroom"),
            ("\u6253\u5F00\u5BA2\u5385\u7A7A\u6C14\u51C0\u5316\u5668", "home.air_purifier.on", "living_room"),
            ("\u5173\u95ED\u4E66\u623F\u7A7A\u6C14\u51C0\u5316\u5668", "home.air_purifier.off", "study"),
            ("\u6253\u5F00\u4E3B\u5367\u7A97\u5E18", "home.covers.open", "master_bedroom"),
            ("\u5173\u95ED\u6B21\u5367\u7A97\u5E18", "home.covers.close", "guest_bedroom"),
            ("\u628A\u9910\u5385\u7A7A\u8C03\u8BBE\u523026\u5EA6", "home.climate.set_temperature", "dining_room"),
        ]
        for text, tool_name, area in cases:
            with self.subTest(text=text):
                resp = await service.respond(AgentRespondRequest(text=text, metadata={"interaction_mode": "fast"}))
                self.assertEqual("fast_rule_router", resp.source)
                self.assertTrue(resp.tool_results)
                self.assertEqual(tool_name, resp.tool_results[0]["tool_name"])
                self.assertEqual(area, resp.tool_results[0]["arguments"].get("area"))

    async def test_unsupported_device_returns_friendly_unresolved(self) -> None:
        service = _build_service(_test_config())
        service.llm_provider = _MustNotCallLlmProvider()
        _inject_catalog(service)
        service.executor.execute = AsyncMock(side_effect=_echo_execute)

        resp = await service.respond(
            AgentRespondRequest(text="\u6253\u5F00\u5BA2\u5385\u7535\u89C6", metadata={"interaction_mode": "fast"})
        )
        self.assertEqual("fast_rule_router", resp.source)
        self.assertEqual(0, len(resp.tool_results))
        parse = resp.security.get("fast_parse", {})
        self.assertFalse(parse.get("matched"))
        self.assertTrue(parse.get("unresolved_parts"))


if __name__ == "__main__":
    unittest.main()
