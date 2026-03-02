from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class PlanStepView(BaseModel):
    step_id: int
    stage: str
    summary: str
    tool_name: str | None = None
    status: str = "pending"


class AgentRespondRequest(BaseModel):
    session_id: str | None = None
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRespondResponse(BaseModel):
    session_id: str
    reply_text: str
    source: str = "llm_tool_router"
    trace_id: str | None = None
    tool_call: ToolCall | None = None
    tool_result: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[PlanStepView] = Field(default_factory=list)
    security: dict[str, Any] = Field(default_factory=dict)


class SessionMemoryResponse(BaseModel):
    session_id: str
    turns: list[dict[str, str]]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool
