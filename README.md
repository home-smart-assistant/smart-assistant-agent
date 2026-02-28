# smart_assistant_agent

FastAPI-based smart home agent with a non-framework loop:

1. LLM (with tool schema) decides tool calls
2. Agent validates and executes tools
3. Tool result is fed back to LLM for final response
4. Short-term + long-term memory are updated

## Tech Stack
- Python
- FastAPI / Uvicorn
- httpx
- pydantic
- FAISS (fixed long-term memory backend)
- SQLite (metadata + vector backup)

## Architecture Layers
- LLM layer: `app/llm/`
- Memory layer: `app/memory/`
  - short-term window memory
  - long-term memory: FAISS (`IndexFlatIP + IndexIDMap2`) + SQLite
- Tool layer: `app/tools/`
- Planning layer: `app/planning/`
- Action layer: `app/action/`
- Runtime orchestration: `app/runtime/agent_service.py`

## APIs
- `POST /v1/agent/respond`
- `GET /v1/agent/session/{session_id}`
- `DELETE /v1/agent/session/{session_id}`
- `GET /v1/agent/context/ha`
- `GET /v1/agent/tools`
- `GET /v1/agent/trace/{trace_id}`
- `GET /v1/agent/traces`
- `GET /v1/agent/metrics`

## Long-term Memory (FAISS fixed)
- No vector DB backend switching
- Index file: `AGENT_FAISS_INDEX_PATH` (default `./data/memory/faiss.index`)
- Metadata DB: `AGENT_FAISS_META_DB_PATH` (default `./data/memory/memory.db`)
- Similarity threshold: `AGENT_FAISS_MIN_SCORE`
- Session-priority recall: same session first, then global fill
- If FAISS import/load fails: long-term memory is disabled, service still runs

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8091
```

## Docker
- Image installs all dependencies from `requirements.txt`
- Data directory is prepared at `/app/data/memory` for index/db persistence

## Docs
- `docs/openapi/agent.openapi.yaml`
- `docs/llm-provider-architecture.md`
- `docs/agent-architecture.md`
