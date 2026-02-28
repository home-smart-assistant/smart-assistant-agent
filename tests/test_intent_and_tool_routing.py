from __future__ import annotations

import unittest

from app.runtime.agent_service import (
    DEVICE_MAINTENANCE_ROUTE,
    HOME_AUTOMATION_ROUTE,
    AgentService,
    IntentRouteDecision,
)
from app.tools.catalog import ToolCatalog


class TestToolCatalogClimateIntent(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ToolCatalog(
            bridge_url="http://127.0.0.1:1",
            timeout_seconds=1.0,
            refresh_interval_seconds=3600.0,
        )

    def test_detect_turn_on_climate_in_study(self) -> None:
        text = "\u5e2e\u6211\u6253\u5f00\u4e66\u623f\u7684\u7a7a\u8c03"
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.climate.turn_on", call.tool_name)
        self.assertIn(call.arguments.get("area"), {"study", "living_room"})

    def test_detect_leave_home_prefers_turn_off_climate(self) -> None:
        text = "\u6211\u5728\u9910\u5385\uff0c\u7a7a\u8c03\u662f\u5f00\u542f\u7684\uff0c\u6211\u73b0\u5728\u8981\u79bb\u5f00\u4e86"
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.climate.turn_off", call.tool_name)
        self.assertEqual("living_room", call.arguments.get("area"))


class TestIntentRouteGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ToolCatalog(
            bridge_url="http://127.0.0.1:1",
            timeout_seconds=1.0,
            refresh_interval_seconds=3600.0,
        )
        self.service = object.__new__(AgentService)
        self.service.catalog = self.catalog

    def test_rule_home_automation_not_overridden_by_llm_misroute(self) -> None:
        text = "\u5e2e\u6211\u6253\u5f00\u4e66\u623f\u7684\u7a7a\u8c03"
        rule = IntentRouteDecision(
            route=HOME_AUTOMATION_ROUTE,
            confidence=0.82,
            reason="home automation keywords detected",
            source="rule",
        )
        llm = IntentRouteDecision(
            route=DEVICE_MAINTENANCE_ROUTE,
            confidence=0.95,
            reason="misclassified by llm",
            source="llm",
        )
        keep_rule = self.service._should_prefer_rule_route(rule, llm, text)
        self.assertTrue(keep_rule)

    def test_rule_router_marks_aircon_open_as_home_automation(self) -> None:
        text = "\u5e2e\u6211\u6253\u5f00\u4e66\u623f\u7684\u7a7a\u8c03"
        decision = self.service._rule_route_decision(text, {})
        self.assertEqual(HOME_AUTOMATION_ROUTE, decision.route)


class TestCandidateFilteringWithContext(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ToolCatalog(
            bridge_url="http://127.0.0.1:1",
            timeout_seconds=1.0,
            refresh_interval_seconds=3600.0,
        )

    def test_exclude_climate_tools_when_context_has_no_climate_entities(self) -> None:
        ha_context = {
            "known_entities": {
                "light": {"study": "switch.shu_fang_deng"},
                "climate": {},
                "cover": {"study": "cover.study"},
            },
            "entity_states": {
                "switch.shu_fang_deng": {"available": True, "state": "off"},
            },
        }
        names = self.catalog.candidate_tool_names(
            route_agent="home_automation_agent",
            role="operator",
            runtime_env="home",
            session_id="s1",
            user_area="study",
            ha_context=ha_context,
            is_role_allowed=lambda _role, _tool: True,
            limit=20,
        )
        self.assertTrue(any(name.startswith("home.lights.") for name in names))
        self.assertFalse(any(name.startswith("home.climate.") for name in names))


if __name__ == "__main__":
    unittest.main()
