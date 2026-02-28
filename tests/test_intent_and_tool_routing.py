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
        text = "帮我打开书房的空调"
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.climate.turn_on", call.tool_name)
        self.assertIn(call.arguments.get("area"), {"study", "living_room"})

    def test_detect_turn_on_light_in_dining_room(self) -> None:
        text = "打开餐厅灯"
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.lights.on", call.tool_name)
        self.assertEqual("dining_room", call.arguments.get("area"))

    def test_detect_leave_home_prefers_turn_off_climate(self) -> None:
        text = "我在餐厅，空调是开启的，我现在要离开了"
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.climate.turn_off", call.tool_name)
        self.assertEqual("dining_room", call.arguments.get("area"))

    def test_detect_leave_home_prefers_turn_off_lights(self) -> None:
        text = "我在餐厅，灯是开启的，我现在要离开了"
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.lights.off", call.tool_name)
        self.assertEqual("dining_room", call.arguments.get("area"))


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
        text = "帮我打开书房的空调"
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
        text = "帮我打开书房的空调"
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


class TestAreaSyncIntent(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ToolCatalog(
            bridge_url="http://127.0.0.1:1",
            timeout_seconds=1.0,
            refresh_interval_seconds=3600.0,
        )
        self.service = object.__new__(AgentService)
        self.service.catalog = self.catalog

    def test_detect_area_sync_tool_call(self) -> None:
        text = "设置区域：玄关，厨房，客厅，主卧，次卧，餐厅，书房，卫生间，走廊，删除无用区域"
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.areas.sync", call.tool_name)
        self.assertTrue(call.arguments.get("delete_unused"))
        self.assertEqual(
            call.arguments.get("target_areas"),
            ["玄关", "厨房", "客厅", "主卧", "次卧", "餐厅", "书房", "卫生间", "走廊"],
        )

    def test_area_phrase_routes_to_home_automation(self) -> None:
        decision = self.service._rule_route_decision("帮我整理一下区域", {})
        self.assertEqual(HOME_AUTOMATION_ROUTE, decision.route)

    def test_detect_area_audit_tool_call(self) -> None:
        text = (
            "检查一下区域设备归属，"
            "玄关，厨房，客厅，主卧，次卧，餐厅，书房，卫生间，走廊"
        )
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.areas.audit", call.tool_name)
        self.assertEqual(
            call.arguments.get("target_areas"),
            ["玄关", "厨房", "客厅", "主卧", "次卧", "餐厅", "书房", "卫生间", "走廊"],
        )

    def test_detect_area_audit_without_area_keyword(self) -> None:
        text = (
            "检查一下玄关，厨房，客厅，主卧，"
            "次卧，餐厅，书房，卫生间，走廊的设备归属"
        )
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.areas.audit", call.tool_name)

    def test_detect_area_assign_tool_call(self) -> None:
        text = (
            "把玄关、厨房、客厅、主卧、次卧、"
            "餐厅、书房、卫生间、走廊的未分配设备按建议批量归类"
        )
        call = self.catalog.detect_tool_call(text)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual("home.areas.assign", call.tool_name)
        self.assertTrue(call.arguments.get("only_with_suggestion"))


if __name__ == "__main__":
    unittest.main()
