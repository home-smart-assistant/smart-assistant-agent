import os
import uuid
from collections import defaultdict, deque
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field


APP_NAME = "smart_assistant_agent"
HA_BRIDGE_URL = os.getenv("HA_BRIDGE_URL", "http://localhost:8092")
MEMORY_MAX_TURNS = int(os.getenv("AGENT_MEMORY_MAX_TURNS", "12"))
TOOL_AUTO_EXECUTE = os.getenv("AGENT_TOOL_AUTO_EXECUTE", "true").lower() == "true"

app = FastAPI(title=APP_NAME, version="0.1.0")

# 进程内短期记忆。后续可切 Redis。
session_memory: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=MEMORY_MAX_TURNS))


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentRespondRequest(BaseModel):
    session_id: str | None = None
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRespondResponse(BaseModel):
    session_id: str
    reply_text: str
    source: str = "rule"
    tool_call: ToolCall | None = None
    tool_result: dict[str, Any] | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "status": "ok",
        "memory_sessions": len(session_memory),
        "ha_bridge_url": HA_BRIDGE_URL,
    }


@app.get("/v1/agent/session/{session_id}")
async def get_session_memory(session_id: str) -> dict[str, Any]:
    memory = list(session_memory.get(session_id, []))
    return {
        "session_id": session_id,
        "turns": memory,
    }


@app.delete("/v1/agent/session/{session_id}")
async def clear_session_memory(session_id: str) -> dict[str, Any]:
    session_memory.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}


@app.post("/v1/agent/respond", response_model=AgentRespondResponse)
async def respond(req: AgentRespondRequest) -> AgentRespondResponse:
    session_id = req.session_id or uuid.uuid4().hex
    text = req.text.strip()

    session_memory[session_id].append(f"user: {text}")

    tool_call = detect_tool_call(text)
    trace_id = uuid.uuid4().hex

    if tool_call is not None:
        tool_result = None
        if TOOL_AUTO_EXECUTE:
            tool_result = await invoke_tool(tool_call, trace_id)

        reply_text = render_tool_reply(tool_call, tool_result)
        session_memory[session_id].append(f"assistant: {reply_text}")
        return AgentRespondResponse(
            session_id=session_id,
            reply_text=reply_text,
            source="rule_tool",
            tool_call=tool_call,
            tool_result=tool_result,
        )

    reply_text = render_chat_reply(session_id, text)
    session_memory[session_id].append(f"assistant: {reply_text}")

    return AgentRespondResponse(
        session_id=session_id,
        reply_text=reply_text,
        source="rule_chat",
    )


async def invoke_tool(tool_call: ToolCall, trace_id: str) -> dict[str, Any]:
    payload = {
        "tool_name": tool_call.tool_name,
        "arguments": tool_call.arguments,
        "trace_id": trace_id,
    }

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(f"{HA_BRIDGE_URL}/v1/tools/call", json=payload)
            if resp.status_code >= 400:
                return {
                    "success": False,
                    "message": f"bridge error: {resp.status_code}",
                    "trace_id": trace_id,
                }
            data = resp.json()
            if isinstance(data, dict):
                return data
            return {"success": False, "message": "invalid bridge payload", "trace_id": trace_id}
    except Exception as ex:
        return {
            "success": False,
            "message": f"bridge unreachable: {ex}",
            "trace_id": trace_id,
        }


def detect_tool_call(text: str) -> ToolCall | None:
    area = detect_area(text)

    if any(k in text for k in ["开灯", "打开灯", "开一下灯"]):
        return ToolCall(tool_name="home.lights.on", arguments={"area": area})

    if any(k in text for k in ["关灯", "关闭灯", "把灯关了"]):
        return ToolCall(tool_name="home.lights.off", arguments={"area": area})

    if any(k in text for k in ["影院模式", "回家模式", "晚安模式"]):
        scene = "scene.cinema" if "影院" in text else ("scene.home" if "回家" in text else "scene.good_night")
        return ToolCall(tool_name="home.scene.activate", arguments={"scene_id": scene})

    if "空调" in text and any(k in text for k in ["调到", "设置", "设为"]):
        temp = extract_temperature(text)
        if temp is not None:
            return ToolCall(
                tool_name="home.climate.set_temperature",
                arguments={"area": area, "temperature": temp},
            )

    return None


def detect_area(text: str) -> str:
    if "卧室" in text:
        return "bedroom"
    if "书房" in text:
        return "study"
    if "客厅" in text:
        return "living_room"
    return "living_room"


def extract_temperature(text: str) -> int | None:
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    if not digits:
        return None
    try:
        value = int(digits[0])
    except ValueError:
        return None
    if 16 <= value <= 30:
        return value
    return None


def render_tool_reply(tool_call: ToolCall, tool_result: dict[str, Any] | None) -> str:
    if not tool_result:
        return f"已识别到指令 {tool_call.tool_name}，等待执行。"

    success = bool(tool_result.get("success", False))
    if success:
        return f"好的，{tool_call.tool_name} 已执行。"
    return f"我识别到了指令，但执行失败：{tool_result.get('message', 'unknown error')}"


def render_chat_reply(session_id: str, text: str) -> str:
    turns = len(session_memory.get(session_id, []))
    if any(k in text for k in ["你是谁", "你叫什么"]):
        return "我是你的家庭语音助手，负责对话和家居控制。"

    if any(k in text for k in ["网络", "在线"]):
        return "我在线，当前由 Agent 服务响应。"

    if any(k in text for k in ["你好", "在吗"]):
        return "在的主人。"

    return f"收到。当前会话已累计 {turns} 条短期记忆。"

