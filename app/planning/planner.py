from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.models import ToolCall
from app.tools.catalog import ToolCatalog


CLAUSE_SPLIT_PATTERN = re.compile(r"[，,。；;]+")
CONNECTOR_PATTERN = re.compile(r"(然后|接着|再|并且|同时|先|最后)")


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
    def __init__(self, catalog: ToolCatalog, multi_step_enabled: bool, max_plan_steps: int) -> None:
        self._catalog = catalog
        self._multi_step_enabled = multi_step_enabled
        self._max_plan_steps = max(1, max_plan_steps)

    def plan(self, text: str) -> AgentPlan:
        steps: list[PlanStep] = [
            PlanStep(step_id=1, stage="perceive", summary="接收用户输入并提取关键意图", status="completed")
        ]
        tool_calls: list[ToolCall] = []
        unresolved_queries: list[str] = []

        clauses = self._split_to_clauses(text)
        next_id = 2
        for clause in clauses:
            tool_call = self._catalog.detect_tool_call(clause)
            if tool_call is None:
                unresolved_queries.append(clause)
                continue
            tool_calls.append(tool_call)
            steps.append(
                PlanStep(
                    step_id=next_id,
                    stage="decide",
                    summary=f"识别工具调用: {tool_call.tool_name}",
                    status="completed",
                    tool_call=tool_call,
                )
            )
            next_id += 1
            steps.append(
                PlanStep(
                    step_id=next_id,
                    stage="act",
                    summary=f"执行工具: {tool_call.tool_name}",
                    status="pending",
                    tool_call=tool_call,
                )
            )
            next_id += 1

        # Fallback: when per-clause parsing misses, run a holistic detection once.
        if not tool_calls:
            holistic_call = self._catalog.detect_tool_call(text)
            if holistic_call is not None:
                tool_calls.append(holistic_call)
                unresolved_queries = []
                steps.append(
                    PlanStep(
                        step_id=next_id,
                        stage="decide",
                        summary=f"璇嗗埆鏁翠綋璇彞宸ュ叿璋冪敤: {holistic_call.tool_name}",
                        status="completed",
                        tool_call=holistic_call,
                    )
                )
                next_id += 1
                steps.append(
                    PlanStep(
                        step_id=next_id,
                        stage="act",
                        summary=f"鎵ц宸ュ叿: {holistic_call.tool_name}",
                        status="pending",
                        tool_call=holistic_call,
                    )
                )
                next_id += 1

        if unresolved_queries:
            steps.append(
                PlanStep(
                    step_id=next_id,
                    stage="decide",
                    summary="存在无法直接映射工具的需求，转交 LLM 决策/回复",
                    status="pending",
                )
            )
            next_id += 1

        steps.append(
            PlanStep(step_id=next_id, stage="feedback", summary="生成反馈并写入记忆", status="pending")
        )

        return AgentPlan(
            steps=steps,
            tool_calls=tool_calls,
            unresolved_queries=unresolved_queries,
        )

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

        steps.append(
            PlanStep(step_id=next_id, stage="feedback", summary="生成反馈并写入记忆", status="pending")
        )
        return AgentPlan(
            steps=steps,
            tool_calls=tool_calls[: self._max_plan_steps],
            unresolved_queries=unresolved,
        )

    def _split_to_clauses(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        if not self._multi_step_enabled:
            return [normalized]
        normalized = CONNECTOR_PATTERN.sub("。", normalized)
        clauses = [part.strip() for part in CLAUSE_SPLIT_PATTERN.split(normalized) if part.strip()]
        if not clauses:
            clauses = [normalized]
        return clauses[: self._max_plan_steps]
