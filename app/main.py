from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from app.core.config import AppConfig
from app.core.models import AgentRespondRequest, AgentRespondResponse
from app.core.text_codec import EncodingNormalizationError
from app.runtime import AgentService


config = AppConfig.from_env()
app = FastAPI(title=config.app_name, version=config.app_version)
logger = logging.getLogger(config.app_name)
service = AgentService(config)


@app.post("/v1/agent/respond", response_model=AgentRespondResponse)
async def respond(req: AgentRespondRequest) -> AgentRespondResponse:
    try:
        return await service.respond(req)
    except EncodingNormalizationError as ex:
        raise HTTPException(status_code=400, detail=ex.to_error_detail()) from ex


@app.get("/v1/agent/session/{session_id}")
async def get_session_memory(session_id: str) -> dict[str, Any]:
    return service.get_session_memory(session_id)


@app.delete("/v1/agent/session/{session_id}")
async def clear_session_memory(session_id: str) -> dict[str, Any]:
    return service.clear_session_memory(session_id)


@app.get("/v1/agent/context/ha")
async def get_ha_context(force_refresh: bool = False) -> dict[str, Any]:
    return await service.get_ha_context(force_refresh=force_refresh)


@app.get("/v1/agent/tools")
async def get_tool_catalog() -> dict[str, Any]:
    return {"tools": service.tool_catalog()}


@app.get("/v1/agent/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    trace = service.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace


@app.get("/v1/agent/traces")
async def list_traces(limit: int = 20) -> dict[str, Any]:
    return {"items": service.list_latest_traces(limit=limit)}


@app.get("/v1/agent/metrics")
async def get_metrics() -> dict[str, Any]:
    return service.metrics_snapshot()
