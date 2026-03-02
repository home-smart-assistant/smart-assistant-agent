from __future__ import annotations

from dataclasses import dataclass, field

from app.core.models import ToolCall


@dataclass
class PlanStep:
    step_id: int
    stage: str
    summary: str
    status: str = "pending"
    tool_call: ToolCall | None = None


@dataclass
class AgentPlan:
    steps: list[PlanStep] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    unresolved_queries: list[str] = field(default_factory=list)

    @property
    def needs_llm(self) -> bool:
        return bool(self.unresolved_queries) or not self.tool_calls


class Planner:
    def __init__(self, catalog: object, multi_step_enabled: bool, max_plan_steps: int) -> None:
        self._max_plan_steps = max(1, max_plan_steps)

    def plan(self, text: str) -> AgentPlan:
        normalized = text.strip()
        unresolved = [normalized] if normalized else []
        steps: list[PlanStep] = [
            PlanStep(step_id=1, stage="perceive", summary="接收用户输入并提取关键意图", status="completed")
        ]
        steps.append(
            PlanStep(
                step_id=2,
                stage="decide",
                summary="等待 LLM 生成工具调用",
                status="pending",
            )
        )
        steps.append(PlanStep(step_id=3, stage="feedback", summary="生成反馈并写入记忆", status="pending"))
        return AgentPlan(steps=steps, tool_calls=[], unresolved_queries=unresolved)

    def plan_with_tool_calls(
        self,
        tool_calls: list[ToolCall],
        unresolved_queries: list[str] | None = None,
    ) -> AgentPlan:
        unresolved = list(unresolved_queries or [])
        steps: list[PlanStep] = [
            PlanStep(step_id=1, stage="perceive", summary="接收用户输入并提取关键意图", status="completed")
        ]
        next_id = 2
        for call in tool_calls[: self._max_plan_steps]:
            steps.append(
                PlanStep(
                    step_id=next_id,
                    stage="decide",
                    summary=f"LLM 路由工具调用: {call.tool_name}",
                    status="completed",
                    tool_call=call,
                )
            )
            next_id += 1
            steps.append(
                PlanStep(
                    step_id=next_id,
                    stage="act",
                    summary=f"执行工具: {call.tool_name}",
                    status="pending",
                    tool_call=call,
                )
            )
            next_id += 1

        if unresolved:
            steps.append(
                PlanStep(
                    step_id=next_id,
                    stage="decide",
                    summary="存在补充问题，转交 LLM 生成最终答复",
                    status="pending",
                )
            )
            next_id += 1

        steps.append(PlanStep(step_id=next_id, stage="feedback", summary="生成反馈并写入记忆", status="pending"))
        return AgentPlan(
            steps=steps,
            tool_calls=tool_calls[: self._max_plan_steps],
            unresolved_queries=unresolved,
        )
